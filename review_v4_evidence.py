"""Paired same-question checks for a completed evidence study, no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not (args.folder/'completed.json').exists():raise ValueError('Complete study required')
    trace=args.folder/'traces.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
    groups={method:{r['item_id']:r for r in rows if r['method']==method}
            for method in ['raw','local_lexical','global_lexical','oracle_full']}
    ids=set(groups['raw'])
    if any(set(g)!=ids for g in groups.values()):raise ValueError('Methods have unequal question sets')
    for qid in ids:
        for g in groups.values():
            if g[qid]['common_target_tokens']!=groups['raw'][qid]['common_target_tokens']:raise ValueError('Target mismatch')
    def totals(row):
        return sum(abs(row['effects'][k]-v) for k,v in row['raw_effects'].items()),sum(abs(v) for v in row['raw_effects'].values())
    report={'scope':'one completed exploratory model/benchmark; no prevalence or generality claim',
        'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'methods':{},'raw_reference_outside_support':{}}
    for method,g in groups.items():
        values=list(g.values());pairs=[totals(r) for r in values]
        report['methods'][method]={'n':len(values),'f1':st.mean(r['qa']['f1'] for r in values),
            'mean_positions':st.mean(r['memory_positions'] for r in values),
            'pooled_nmae':sum(a for a,b in pairs)/max(sum(b for a,b in pairs),1e-8)}
    outside=total=0.
    for qid,row in groups['raw'].items():
        oracle=groups['oracle_full'][qid];unsupported=[k for k in row['effects'] if k not in row['support_ids']]
        outside+=sum(abs(row['effects'][k]) for k in unsupported);total+=sum(abs(v) for v in row['effects'].values())
        if any(abs(oracle['effects'][k])>1e-4 for k in unsupported):raise ValueError('Oracle unexpectedly depends on excluded source')
    report['raw_reference_outside_support']={'pooled_l1_fraction':outside/max(total,1e-8),
        'interpretation':'outside annotated support is not equivalent to irrelevant; support-only memory has zero deletion effects for excluded sources',
        'oracle_error_lower_bound_holds':report['methods']['oracle_full']['pooled_nmae']+1e-6>=outside/max(total,1e-8)}
    cohort=sorted(q for q in ids if all(groups[m][q]['qa']['em']==1 for m in ['raw','global_lexical','oracle_full']))
    if cohort:
        differences=[];slices={}
        for method in ['global_lexical','oracle_full']:
            values=[groups[method][q] for q in cohort];pairs=[totals(r) for r in values]
            slices[method]={'pooled_nmae':sum(a for a,b in pairs)/max(sum(b for a,b in pairs),1e-8),
                'rank_misses':sum(r['comparison']['top1_hit'] is False for r in values)}
        for q in cohort:
            o=groups['oracle_full'][q];g=groups['global_lexical'][q]
            differences.append((totals(o)[0]-totals(g)[0])/len(o['effects']))
        a=np.asarray(differences);rng=np.random.default_rng(20261108)
        means=a[rng.integers(len(a),size=(5000,len(a)))].mean(axis=1)
        report['same_questions_all_three_exactly_correct']={'n':len(cohort),'question_ids':cohort,'methods':slices,
            'oracle_minus_global_mean_effect_mae':float(a.mean()),'question_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),
            'selection':'post-outcome both-correct diagnostic, not population inference or causal effect of QA correctness'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
