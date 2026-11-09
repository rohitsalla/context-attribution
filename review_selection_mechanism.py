"""Review the three selection contracts; retain incomplete status."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np
from context_attribution.v2.metrics import effect_comparison


def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261109)
    samples=a[rng.integers(len(a),size=(5000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'question_bootstrap_95':np.quantile(samples,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    report={'scope':'adaptive four-study follow-up; unadjusted question intervals, no locked-test claim','studies':{}}
    methods={'frozen':'frozen_effects','fixed_rank_repack':'effects','refitted':'recomputed_effects'}
    for name,n in [('hotpot_qwen7',96),('hotpot_llama8',96),('musique_qwen7',32),('musique_llama8',32)]:
        folder=args.root/name;trace=folder/'traces.jsonl'
        rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
        status={'complete':(folder/'completed.json').exists(),'available_questions':len(rows)};report['studies'][name]=status
        if not status['complete']:continue
        done=json.loads((folder/'completed.json').read_text());assert done['complete'] and done['n']==n==len(rows)
        assert len({r['item_id'] for r in rows})==n
        for r in rows:
            assert max(r['reference_factual_mean_logp_error'],r['reference_raw_mean_logp_error'])<=1e-6
            assert r['repeat_mean_logp_error']<=1e-7
            assert all(set(r[field])==set(r['raw_effects']) for field in methods.values())
        status['trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest();status['methods']={};errors={}
        for name_method,field in methods.items():
            errors[name_method]={r['item_id']:st.mean(abs(r[field][k]-v) for k,v in r['raw_effects'].items()) for r in rows}
            numerator=sum(abs(r[field][k]-v) for r in rows for k,v in r['raw_effects'].items())
            denominator=sum(abs(v) for r in rows for v in r['raw_effects'].values())
            hits=[effect_comparison(r['raw_effects'],r[field])['top1_hit'] for r in rows];hits=[h for h in hits if h is not None]
            status['methods'][name_method]={'mean_question_mae':st.mean(errors[name_method].values()),
                'pooled_nmae':numerator/max(denominator,1e-12),'top_source_hit_rate':st.mean(hits) if hits else None}
        status['paired_contrasts']={}
        for left,right in [('fixed_rank_repack','refitted'),('frozen','fixed_rank_repack')]:
            ids=sorted(errors[left]);correct={r['item_id'] for r in rows if r['qa']['em']==1}
            status['paired_contrasts'][left+'-minus-'+right]={'all':{'n':len(ids),'mae':interval([errors[left][q]-errors[right][q] for q in ids])}}
            if correct:status['paired_contrasts'][left+'-minus-'+right]['factual_exactly_correct']={'n':len(correct),
                'mae':interval([errors[left][q]-errors[right][q] for q in sorted(correct)])}
        status['fixed_rank_mean_intervened_positions']=st.mean(st.mean(r['intervention_positions'].values()) for r in rows)
        status['refitted_vs_fixed_repack_token_change_count']=sum(c['refitted_vs_fixed_repack_tokens_differ'] for r in rows for c in r['token_change_checks'].values())
        status['fixed_repack_vs_frozen_token_change_count']=sum(c['fixed_repack_vs_frozen_tokens_differ'] for r in rows for c in r['token_change_checks'].values())
    args.root.mkdir(parents=True,exist_ok=True);(args.root/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
