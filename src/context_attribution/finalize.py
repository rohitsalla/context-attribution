"""Aggregate full-study shards, retaining an explicit completeness audit."""
import argparse
import json
from pathlib import Path
from .analyze import analyze
from .experiment import write_aggregates


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    root=args.directory;counts={};all_main=[]
    for name,expected in [('main-128',120),('main-256',120),('main-512',120),('stress',180),('stochastic',8),('online',12)]:
        d=root/name;trace=d/'traces.jsonl'
        rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
        keys=[(r['item_id'],r.get('cap')) for r in rows]
        if len(set(keys))!=len(keys): raise ValueError(f'Duplicate conditions in {name}')
        counts[name]={'completed':len(rows),'expected':expected,'complete':len(rows)==expected}
        if name.startswith('main-'): all_main.extend(rows)
        if rows and name!='online': analyze(d)
    merged=root/'main-combined';merged.mkdir(exist_ok=True)
    if all_main:
        original=json.loads((root/'main-128/manifest.json').read_text())
        original['aggregation_only']=True;original['caps']=[128,256,512]
        (merged/'manifest.json').write_text(json.dumps(original,indent=2))
        (merged/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_main))
        write_aggregates(merged,all_main,1729);main_analysis=analyze(merged)
    else: main_analysis=None
    report={'complete':all(c['complete'] for c in counts.values()),'counts':counts,'main_analysis':main_analysis,
            'natural_human_audit_claimed':False,'natural_subset':'official facts-page excerpts, assistant-checked',
            'limitations':['Facts-page examples are not a long-form natural prose benchmark.',
                           'Surrogate approximation is not the official ContextCite implementation.',
                           'Five stochastic replicates are exploratory and do not calibrate a universal noise threshold.']}
    # Online records are actually executed selected interventions. Compare only matching answer spans.
    online=root/'online/traces.jsonl';lookup={(r['item_id'],r['cap']):r for r in all_main}
    comparisons=[]
    if online.exists():
        for line in online.read_text().splitlines():
            row=json.loads(line);ref=lookup.get((row['item_id'],row['cap']))
            if not ref:continue
            match=row['original']['fixed_answer']==ref['original']['fixed_answer']
            for method in row['methods']:
                for budget in method['budgets']:
                    full=sum(b['trials'][0]['cost']['seconds'] for b in ref['reference'].values())
                    maximum=max(map(abs,ref['effects'].values()))
                    hit=any(maximum-abs(ref['effects'][s])<=ref['replay_noise'] for s in budget['selected'])
                    comparisons.append({'item_id':row['item_id'],'method':method['method'],'budget':budget['budget'],
                        'matching_answer_span':match,'hit':hit if match and maximum>ref['replay_noise'] else None,
                        'online_seconds':budget['end_to_end_seconds'],'offline_full_replay_seconds':full,
                        'time_ratio':budget['end_to_end_seconds']/max(full,1e-12)})
    report['online_comparisons']=comparisons
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'complete':report['complete'],'counts':counts},indent=2),flush=True)
    if not report['complete']:raise SystemExit('Incomplete study: see report.json; no complete-study conclusion is claimed.')


if __name__=='__main__':main()
