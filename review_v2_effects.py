"""Independent, length-normalized effect sanity check on completed V2 traces."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report={'scope':'eight-question adaptive development; descriptive only',
        'formula':'sum_question,sum_source |raw_effect/n_target - compressed_effect/n_target| / sum |raw_effect/n_target|',
        'zero_effect_predictor_pooled_nmae':1.,'results':[],'trace_sha256':{}}
    for folder in sorted(args.root.glob('profile*')):
        if not (folder/'completed.json').exists():continue
        trace=folder/'traces.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
        report['trace_sha256'][folder.name]=hashlib.sha256(trace.read_bytes()).hexdigest()
        raw={r['item_id']:r for r in rows if r['method']=='raw'}
        for method in ['global','source','lexical']:
            selected=[r for r in rows if r['method']==method and r['budget']==512];error=scale=0.
            for row in selected:
                ref=raw[row['item_id']];n=len(ref['answer']['tokens'])
                error+=sum(abs(effect-row['common_raw_target_effects'][sid])/n for sid,effect in ref['effects'].items())
                scale+=sum(abs(effect)/n for effect in ref['effects'].values())
            report['results'].append({'profile':folder.name,'method':method,'budget':512,'n':len(selected),
                'pooled_nmae':error/scale if scale else None,'reference_l1_mean_logprob':scale,
                'raw_top1_hits':sum(r['effect_distortion_vs_raw']['top1_hit'] is True for r in selected)})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
