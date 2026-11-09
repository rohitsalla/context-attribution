"""Descriptive partition of old BF16 transfer error by factual source coverage."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));args=p.parse_args()
    coverage_path=args.root/'research/v4/SELECTION_COVERAGE.json'
    coverage=json.loads(coverage_path.read_text())
    result={'scope':'post-hoc descriptive partition of prior BF16 error; not a causal decomposition or FP32 finding',
        'coverage_sha256':hashlib.sha256(coverage_path.read_bytes()).hexdigest(),'studies':{}}
    for name,study in coverage['studies'].items():
        trace=args.root/'results/v4'/f'evidence_{name}'/'traces.jsonl'
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==study['trace_sha256']
        rows={r['item_id']:r for r in map(json.loads,trace.read_text().splitlines()) if r['method']=='global_lexical'}
        assert set(rows)=={c['item_id'] for c in study['per_question']}
        absent_error=represented_error=0.
        for c in study['per_question']:
            row=rows[c['item_id']];absent=set(c['sources_without_selected_tokens'])
            for k,v in row['raw_effects'].items():
                error=abs(row['effects'][k]-v)
                if k in absent:absent_error+=error
                else:represented_error+=error
        total=absent_error+represented_error
        result['studies'][name]={'n':96,'total_absolute_effect_error':total,
            'error_on_unrepresented_sources':absent_error,'error_on_partly_or_fully_represented_sources':represented_error,
            'fraction_error_on_unrepresented_sources':absent_error/max(total,1e-12),
            'fraction_error_on_partly_or_fully_represented_sources':represented_error/max(total,1e-12)}
    result['interpretation']='Having any selected tokens is weaker than retaining relevant evidence. Most error occurs on sources with some selected tokens; this does not identify selection-policy changes as the cause. Within-source omission, changed calibration, numerical sensitivity and policy changes remain alternatives. Error location is not an explained variance or causal contribution.'
    out=args.root/'research/v4/COVERAGE_ERROR.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
