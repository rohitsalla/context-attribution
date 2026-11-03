"""Analyze complete or partial real-model traces without changing the run."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root):
    rows = [json.loads(line) for line in (root/'traces.jsonl').read_text().splitlines() if line.strip()]
    manifest = json.loads((root/'manifest.json').read_text())
    output = {"backend": manifest['backend_name'], "research_evidence": manifest['research_evidence'],
              "completed_conditions": len(rows), "completed_items": len(set(r['item_id'] for r in rows)),
              "caps": [], "counterexamples": [],
              "caveats": ["Exploratory single-template fictional pilot, not a generalization benchmark.",
                          "Sparse-surrogate approximation, not official ContextCite.",
                          "Costs estimated from shared offline traces; no production latency claim."]}
    rng = np.random.default_rng(819)
    for cap in sorted(set(r['cap'] for r in rows)):
        subset = [r for r in rows if r['cap'] == cap]
        entry = {"cap": cap, "n": len(subset), "accuracy": float(np.mean([r['correct'] for r in subset])),
                 "raw_accuracy": float(np.mean([r['raw_context_correct'] for r in subset])),
                 "summary_length_mean": float(np.mean([r['original']['summary_info']['output_tokens'] for r in subset])),
                 "summary_truncations": sum(r['original']['summary_info']['truncated'] for r in subset),
                 "max_replay_noise": max(r['replay_noise'] for r in subset),
                 "max_effect_mean_nats": float(np.mean([max(map(abs,r['effects'].values())) for r in subset])),
                 "irrelevant_abs_effect_mean": float(np.mean([abs(r['irrelevant_effect']) for r in subset])),
                 "rank_correlation_mean": {}, "methods": []}
        for name in ['recursive_surrogate', 'lexical', 'source_id']:
            vals = [r['rank_correlation'][name] for r in subset if r['rank_correlation'][name] is not None]
            entry['rank_correlation_mean'][name] = float(np.mean(vals)) if vals else None
        for budget in [1,2,3]:
            for method in ['recursive_surrogate','lexical','source_id','random_exact_expectation','oracle']:
                hits, differences, costs, tokens, calls, effect_ratios, base_ids = [], [], [], [], [], [], []
                for row in subset:
                    c = next(c for c in row['comparisons'] if c['method']==method and c['budget']==budget)
                    rand = next(c for c in row['comparisons'] if c['method']=='random_exact_expectation' and c['budget']==budget)
                    maximum = max(map(abs,row['effects'].values()))
                    if c['hit'] is not None:
                        hits.append(float(c['hit']))
                        differences.append(float(c['hit'])-rand['hit'])
                        effect_ratios.append(c['discovered_effect']/maximum)
                        base_ids.append(row['base_id'])
                    overhead = c.get('selection_overhead') or {}
                    full = {k:sum(b['trials'][0]['cost'][k] for b in row['reference'].values())
                            for k in ['seconds','calls','input_tokens','output_tokens']}
                    if method=='random_exact_expectation':
                        replay = {k: v*budget/6 for k,v in full.items()}
                    else:
                        replay = c['one_replay_per_source_cost']
                    # Oracle requires the complete reference to choose its subset.
                    if method=='oracle':
                        replay = full
                    costs.append((overhead.get('wall_seconds',0)+replay['seconds'])/max(full['seconds'],1e-12))
                    tokens.append((overhead.get('input_tokens',0)+overhead.get('output_tokens',0)+replay['input_tokens']+replay['output_tokens'])/
                                  max(full['input_tokens']+full['output_tokens'],1))
                    calls.append(overhead.get('calls',0)+replay['calls'])
                    if method=='recursive_surrogate' and budget==1 and c['hit'] is False:
                        output['counterexamples'].append({"item_id":row['item_id'],"cap":cap,"correct":row['correct'],
                            "answer":row['original']['fixed_answer'],"summary":row['original']['summary'],
                            "selected":c['selected'],"effects":row['effects'],"scores":row['candidate_scores'][method]})
                clusters = {b:[] for b in base_ids}
                for b,d in zip(base_ids,differences): clusters[b].append(d)
                vals = [np.mean(v) for v in clusters.values()]
                boot = [np.mean(rng.choice(vals,len(vals))) for _ in range(2000)] if vals else []
                entry['methods'].append({"method":method,"budget":budget,"resolved_conditions":len(hits),
                    "hit_rate":float(np.mean(hits)) if hits else None,
                    "paired_discovery_gain_over_random":float(np.mean(differences)) if differences else None,
                    "paired_gain_cluster_bootstrap_95ci":np.quantile(boot,[.025,.975]).tolist() if boot else None,
                    "discovered_fraction_of_max_effect":float(np.mean(effect_ratios)) if effect_ratios else None,
                    "time_ratio_to_full_replay":float(np.mean(costs)),
                    "token_ratio_to_full_replay":float(np.mean(tokens)),"mean_model_calls":float(np.mean(calls))})
        output['caps'].append(entry)
    (root/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    result=analyze(args.directory)
    print(json.dumps({k:v for k,v in result.items() if k!='counterexamples'},indent=2))
    print('B=1 local-ranking counterexamples:',len(result['counterexamples']))


if __name__=='__main__':
    main()
