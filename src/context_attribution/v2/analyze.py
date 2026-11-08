"""Descriptive development summaries; incomplete shards stay explicitly incomplete."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    report={'scope':'development feasibility, not held-out efficacy','profiles':{},'training_canaries':{}}
    for folder in sorted(args.root.glob('profile*')):
        trace=folder/'traces.jsonl'
        if not trace.exists():continue
        rows=[json.loads(s) for s in trace.read_text().splitlines()];groups=defaultdict(list)
        raw={r['item_id']:r for r in rows if r['method']=='raw'}
        for row in rows:groups[(row['method'],row['budget'])].append(row)
        summary=[]
        for (method,budget),values in groups.items():
            row={'method':method,'budget':budget,'n':len(values),
                 'qa_f1':statistics.mean(v['qa']['f1'] for v in values),
                 'qa_em':statistics.mean(v['qa']['em'] for v in values),
                 'mean_memory_tokens':statistics.mean(v['memory_tokens'] for v in values)}
            if method!='raw':
                row.update({'mean_effect_mae':statistics.mean(v['effect_distortion_vs_raw']['mean_absolute_error'] for v in values),
                    'mean_answer_audit_vs_raw_seconds':statistics.mean(v['production_seconds']/raw[v['item_id']]['seconds'] for v in values),
                    'all_sampled_cold_tokens_equal':all(c['identical_tokens'] for v in values for c in v['sampled_uncached_validation']),
                    'max_sampled_cold_effect_error':max((c['effect_error'] for v in values for c in v['sampled_uncached_validation']),default=0)})
            summary.append(row)
        report['profiles'][folder.name]={'complete':(folder/'completed.json').exists(),'records':len(rows),'conditions':summary}
    for folder in sorted(args.root.glob('train_cf*')):
        if (folder/'summary.json').exists():report['training_canaries'][folder.name]=json.loads((folder/'summary.json').read_text())
    canaries=list(report['training_canaries'].values())
    report['matched_initial_training_parameters']=(len(canaries)==2 and len({c['initial_pool_sha256'] for c in canaries})==1)
    destination=args.root/'development_summary.json';destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
