"""Completed-seed-only anchored comparisons; no model calls or winner selection."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
from context_attribution.v3.report import summarize,paired_interval

LABELS=['factual_0','unpaired_0.1','paired_0.1','unpaired_1','paired_1','lexical32']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();studies={};report={'scope':'adaptive development; question-bootstrap intervals do not capture full seed uncertainty',
        'completed_seeds':[],'expected_seeds':[1729,1730,1731],'exact_state_verified':True,'means':{},'comparisons':{},'trace_sha256':{}}
    for folder in sorted(args.root.glob('task*_anchored_*')):
        if folder.name.startswith('task0_') or not (folder/'completed.json').exists():continue
        done=json.loads((folder/'completed.json').read_text());seed=done['seed']
        if not done['all_branch_start_states_equal']:raise ValueError('Branch state mismatch')
        study={}
        for label in LABELS:
            path=folder/(label+'.jsonl') if label=='lexical32' else folder/label/'dev.jsonl'
            if label!='lexical32':
                receipt=json.loads((folder/label/'completed.json').read_text())
                if receipt['initial_shared_state']!=done['shared_hashes']:raise ValueError('Invalid shared-state receipt')
            values=[json.loads(s) for s in path.read_text().splitlines()]
            study[label]={r['item_id']:r for r in values}
            report['trace_sha256'][f'{seed}/{label}']=hashlib.sha256(path.read_bytes()).hexdigest()
        ids=set(study['lexical32'])
        if any(set(v)!=ids for v in study.values()):raise ValueError('Nonmatched question sets')
        for q in ids:
            for values in study.values():
                if values[q]['raw_effects']!=study['lexical32'][q]['raw_effects']:raise ValueError('Reference changed between branches')
        studies[seed]=study;report['completed_seeds'].append(seed)
    if not studies:raise ValueError('No completed anchored suite')
    report['all_three_seeds_complete']=report['completed_seeds']==report['expected_seeds']
    for label in LABELS:
        by_seed={seed:summarize(list(study[label].values())) for seed,study in studies.items()}
        report['means'][label]={'by_seed':by_seed,
            'qa_f1':st.mean(v['f1'] for v in by_seed.values()),
            'singleton_mae':st.mean(v['singleton_mean_logprob_effect_mae'] for v in by_seed.values()),
            'singleton_pooled_nmae':st.mean(v['singleton_pooled_nmae'] for v in by_seed.values()),
            'joint_mae':st.mean(v['joint_mean_logprob_effect_mae'] for v in by_seed.values())}
    ids=sorted(next(iter(studies.values()))['lexical32'])
    if any(set(study['lexical32'])!=set(ids) for study in studies.values()):raise ValueError('Seed question sets differ')
    for weight in ['0.1','1']:
        for control in ['factual_0','unpaired_'+weight,'lexical32']:
            effect=[];joint=[];quality=[]
            for q in ids:
                pairs=[(s['paired_'+weight][q],s[control][q]) for s in studies.values()]
                effect.append(st.mean(a['single_comparison']['mean_absolute_error']-b['single_comparison']['mean_absolute_error'] for a,b in pairs))
                joint.append(st.mean(a['joint_comparison']['mean_absolute_error']-b['joint_comparison']['mean_absolute_error'] for a,b in pairs))
                quality.append(st.mean(a['qa']['f1']-b['qa']['f1'] for a,b in pairs))
            report['comparisons'][f'paired_{weight}-vs-{control}']={'effect_mae':paired_interval(effect),
                'joint_effect_mae':paired_interval(joint),'qa_f1':paired_interval(quality)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['completed_seeds','all_three_seeds_complete','means','comparisons']},indent=2))


if __name__=='__main__':main()
