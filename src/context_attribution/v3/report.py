"""Question-level development analysis; absent/incomplete runs are never successes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import numpy as np


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(values):
    if not values:return None
    result={'n':len(values),'f1':st.mean(r['qa']['f1'] for r in values),'em':st.mean(r['qa']['em'] for r in values)}
    if 'raw_effects' in values[0]:
        for name,joint in [('singleton',False),('joint',True)]:
            errors=[];scales=[]
            for r in values:
                keys=[k for k in r['effects'] if (',' in k)==joint]
                errors.extend(abs(r['effects'][k]-r['raw_effects'][k]) for k in keys)
                scales.extend(abs(r['raw_effects'][k]) for k in keys)
            result[name+'_mean_logprob_effect_mae']=st.mean(errors)
            result[name+'_pooled_nmae']=sum(errors)/max(sum(scales),1e-8)
            result[name+'_zero_predictor_mae']=st.mean(scales)
    return result


def paired_interval(differences):
    a=np.asarray(differences,float);rng=np.random.default_rng(20261108)
    means=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(axis=1)
    return {'difference':float(a.mean()),'question_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),'n_questions':len(a)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--v2',type=Path,required=True);p.add_argument('--v3',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report={'scope':'development only; exploratory follow-up, not confirmation','v2':{},'v3':{},'paired_comparisons':{}}
    lines=['# Development hypothesis review','',
      'These are development observations. Eight-question pilot means are unstable; they are not powered paper results. The locked test remains untouched.','',
      '| Answerer/profile | System | Budget | QA F1 | Effect MAE (sum log p) | Audit time / raw |','|---|---|---:|---:|---:|---:|']
    for folder in sorted(args.v2.glob('profile*')):
        values=rows(folder/'traces.jsonl')
        if not (folder/'completed.json').exists():
            report['v2'][folder.name]={'complete':False,'records':len(values)};continue
        raw={r['item_id']:r for r in values if r['method']=='raw'};conditions=[]
        rawf1=st.mean(r['qa']['f1'] for r in raw.values())
        lines.append(f'| {folder.name} | raw | full | {100*rawf1:.1f} | 0 | 1 |')
        for method in ['global','source','group4','lead','lexical']:
            for budget in [128,256,512]:
                selected=[r for r in values if r['method']==method and r['budget']==budget]
                mae=st.mean(r['effect_distortion_vs_raw']['mean_absolute_error'] for r in selected)
                ratio=st.mean(r['production_seconds']/raw[r['item_id']]['seconds'] for r in selected)
                correct=[r for r in selected if r['qa']['em']==1 and raw[r['item_id']]['qa']['em']==1]
                failed=sum(r['effect_distortion_vs_raw']['top1_hit'] is False for r in correct)
                conditions.append({'method':method,'budget':budget,'n':len(selected),'f1':st.mean(r['qa']['f1'] for r in selected),
                    'effect_mae':mae,'time_ratio':ratio,'both_exactly_correct_n':len(correct),'both_correct_rank_misses':failed,
                    'sampled_cold_checks':sum(len(r['sampled_uncached_validation']) for r in selected),
                    'max_cold_error':max(c['effect_error'] for r in selected for c in r['sampled_uncached_validation'])})
                if budget==512:lines.append(f'| {folder.name} | {method} | {budget} | {100*conditions[-1]["f1"]:.1f} | {mae:.3f} | {ratio:.2f} |')
        report['v2'][folder.name]={'complete':True,'records':len(values),'conditions':conditions,
            'no_context_f1':st.mean(r['no_context']['qa']['f1'] for r in raw.values())}
    for folder in sorted(args.v3.glob('*')):
        if not folder.is_dir():continue
        if folder.name=='canary_eval':
            report['v3'][folder.name]={'scope':'debugging only; inherited repetition-penalty mismatch; excluded from efficacy'}
            continue
        complete=(folder/'completed.json').exists()
        if folder.name.startswith('cheap_profile'):
            values=rows(folder/'traces.jsonl');report['v3'][folder.name]={'complete':complete,'records':len(values)};continue
        data={f.stem:summarize(rows(f)) for f in folder.glob('*.jsonl') if f.name!='training.jsonl'}
        report['v3'][folder.name]={'complete':complete,'evaluations':data}
    # Average paired differences over seeds BEFORE question bootstrap. Seeds are
    # not treated as independent question-level replicates.
    for control in ['factual','unpaired']:
        per_question=defaultdict(list);seed_diffs=[];warmups_match=True
        for paired_folder in sorted(args.v3.glob('paired_*')):
            seed=paired_folder.name.split('_')[-1];other=args.v3/f'{control}_{seed}'
            if not all((f/'completed.json').exists() for f in [paired_folder,other]):continue
            a={r['item_id']:r for r in rows(paired_folder/'dev.jsonl')};b={r['item_id']:r for r in rows(other/'dev.jsonl')}
            if set(a)!=set(b):raise ValueError('Unpaired development questions')
            am=json.loads((paired_folder/'completed.json').read_text());bm=json.loads((other/'completed.json').read_text())
            warmups_match &= am['warmup_pool_sha256']==bm['warmup_pool_sha256']
            differences=[]
            for key,r in a.items():
                d=r['single_comparison']['mean_absolute_error']-b[key]['single_comparison']['mean_absolute_error']
                q=r['qa']['f1']-b[key]['qa']['f1'];per_question[key].append((d,q));differences.append(d)
            seed_diffs.append({'seed':seed,'effect_mae_difference':st.mean(differences)})
        if per_question:
            report['paired_comparisons'][control]={'seed_differences':seed_diffs,'matched_warmup_parameters':warmups_match,
                'effect_mae':paired_interval([st.mean(d[0] for d in ds) for ds in per_question.values()]),
                'qa_f1':paired_interval([st.mean(d[1] for d in ds) for ds in per_question.values()])}
    lines+=['','## Interpretation','',
        'The caching contract is passing sampled replay checks. The initial source-local abstractive method does not yet win the quality/effect/cost comparison against lexical selection.','',
        'Correct factual answers can coexist with changed intervention rankings. This motivates testing an intervention-preserving objective, but does not establish novelty or effectiveness. The soft-memory canaries establish optimization only. Larger matched factual/unpaired/paired controls and untouched-question evaluation are required.','',
        'Nominal budgets are upper bounds: generated summaries often use fewer positions than extractive memory. Do not attribute every observed difference to source locality or compare summed-log-probability MAE across answerer families as if target lengths were equal. V3 adds mean-log-probability effects, a zero-effect predictor, joint deletions and position-matched lexical controls.','',
        'Full machine-readable follow-up results, including incomplete flags and question-level paired intervals, are in the adjacent JSON. Training success is not held-out efficacy.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['paired_comparisons'],indent=2))


if __name__=='__main__':main()
