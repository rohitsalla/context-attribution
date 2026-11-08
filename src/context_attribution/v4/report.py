"""Aggregate exact-state branch suites and evidence oracles, never hide failures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import numpy as np
from ..v3.report import summarize,paired_interval,rows


def evidence_summary(values):
    if not values:return None
    errors=[abs(r['effects'][k]-v) for r in values for k,v in r['raw_effects'].items()]
    scales=[abs(v) for r in values for v in r['raw_effects'].values()]
    both=[r for r in values if r['qa']['em']==1 and r['raw_qa']['em']==1]
    return {'n':len(values),'f1':st.mean(r['qa']['f1'] for r in values),'em':st.mean(r['qa']['em'] for r in values),
        'pooled_nmae':sum(errors)/max(sum(scales),1e-8),'mean_positions':st.mean(r['memory_positions'] for r in values),
        'both_correct_n':len(both),'both_correct_rank_misses':sum(r['comparison']['top1_hit'] is False for r in both)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    report={'scope':'adaptive development; no locked-test claim','suites':{},'evidence':{},'generative':{},'paired_comparisons':{}}
    suites=[]
    for folder in sorted(args.root.glob('task*')):
        done=folder/'completed.json';complete=done.exists();summary={'complete':complete,'branches':{}}
        manifest=json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {}
        summary['architecture']=manifest.get('architecture');summary['seed']=manifest.get('seed')
        for branch in sorted(folder.glob('*')):
            if branch.is_dir():summary['branches'][branch.name]={'complete':(branch/'completed.json').exists(),'evaluation':summarize(rows(branch/'dev.jsonl'))}
        summary['lexical32']=summarize(rows(folder/'lexical32.jsonl'))
        if complete:
            completion=json.loads(done.read_text());summary['all_branch_start_states_equal']=completion['all_branch_start_states_equal']
            if not completion['all_branch_start_states_equal']:raise ValueError('Exact-state suite has mismatched starts')
            if folder.name.startswith('task0_'):summary['scope']='smoke only'
            else:suites.append((folder,summary))
        report['suites'][folder.name]=summary
    for architecture in ['pure','anchored']:
        selected=[(f,s) for f,s in suites if s['architecture']==architecture]
        for weight in ['0.1','1']:
            for control in ['factual_0','unpaired_'+weight]:
                differences=defaultdict(list);per_seed=[]
                for folder,suite in selected:
                    left={r['item_id']:r for r in rows(folder/f'paired_{weight}/dev.jsonl')}
                    right={r['item_id']:r for r in rows(folder/control/'dev.jsonl')}
                    if set(left)!=set(right) or not left:raise ValueError('Nonmatching branch cases')
                    ds=[]
                    for key,r in left.items():
                        d=r['single_comparison']['mean_absolute_error']-right[key]['single_comparison']['mean_absolute_error']
                        q=r['qa']['f1']-right[key]['qa']['f1'];differences[key].append((d,q));ds.append(d)
                    per_seed.append({'seed':suite['seed'],'effect_mae_difference':st.mean(ds)})
                if differences:
                    report['paired_comparisons'][f'{architecture}/paired_{weight}-vs-{control}']={
                        'per_seed':per_seed,'effect_mae':paired_interval([st.mean(v[0] for v in ds) for ds in differences.values()]),
                        'qa_f1':paired_interval([st.mean(v[1] for v in ds) for ds in differences.values()])}
    for folder in sorted(args.root.glob('evidence_*')):
        values=rows(folder/'traces.jsonl');groups=defaultdict(list)
        for row in values:groups[row['method']].append(row)
        report['evidence'][folder.name]={'complete':(folder/'completed.json').exists(),
            'methods':{k:evidence_summary(v) for k,v in groups.items()}}
    for folder in sorted(args.root.glob('strong_global_*')):
        values=rows(folder/'traces.jsonl');groups=defaultdict(list)
        for row in values:groups[row['method']].append(row)
        report['generative'][folder.name]={'complete':(folder/'completed.json').exists(),
            'qa':{k:{'n':len(v),'f1':st.mean(r['qa']['f1'] for r in v)} for k,v in groups.items()}}
    (args.root/'review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
