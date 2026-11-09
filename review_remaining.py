"""Review completed stronger-compressor and positional precision studies offline."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np


def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261108)
    means=a[rng.integers(len(a),size=(5000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'question_bootstrap_95':np.quantile(means,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result={'scope':'adaptive development; no locked test; intervals unadjusted for multiple comparisons',
            'stronger_compressors':{},'positional':{}}
    for key in ['qwen7','llama8','mistral7']:
        folder=args.root/'v4'/f'strong_global_{key}'
        assert json.loads((folder/'completed.json').read_text())['complete']
        trace=folder/'traces.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
        raw={r['item_id']:r for r in rows if r['method']=='raw'}
        gen={r['item_id']:r for r in rows if r['method']=='global'}
        assert len(raw)==32 and raw.keys()==gen.keys()
        errors=[];scales=[];token_errors=[];token_scales=[];qa_diffs=[];maes=[];checks=[]
        for qid in sorted(raw):
            r,g=raw[qid],gen[qid];target=g['common_raw_target_tokens'];assert target==r['answer']['tokens'] and target
            assert g['common_raw_target_effects'].keys()==r['effects'].keys()
            n=len(target);d=[abs(g['common_raw_target_effects'][k]-v) for k,v in r['effects'].items()]
            errors.extend(v/n for v in d);scales.extend(abs(v)/n for v in r['effects'].values())
            token_errors.extend(d);token_scales.extend(abs(v) for v in r['effects'].values())
            maes.append(st.mean(d)/n);qa_diffs.append(g['qa']['f1']-r['qa']['f1'])
            checks.extend(g['sampled_uncached_validation'])
        result['stronger_compressors'][key]={
            'complete':True,'n':32,'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
            'raw_f1':st.mean(r['qa']['f1'] for r in raw.values()),'generated_f1':st.mean(r['qa']['f1'] for r in gen.values()),
            'generated_minus_raw_f1':interval(qa_diffs),'mean_positions':st.mean(r['memory_tokens'] for r in gen.values()),
            'nominal_cap':512,'pooled_nmae_mean_token_logp':sum(errors)/sum(scales),
            'pooled_nmae_summed_token_logp':sum(token_errors)/sum(token_scales),
            'mean_question_mae_mean_token_logp':st.mean(maes),
            'cached_cold_checks':len(checks),'all_checked_tokens_equal':all(c['identical_tokens'] for c in checks),
            'max_checked_effect_error':max(c['effect_error'] for c in checks),
            'interpretation':'sampled own-pipeline replay is distinct from raw-reference transfer; achieved lengths far below cap; not an equal-achieved-budget comparison'}
    folder=args.root/'precision-probe';trace=folder/'traces.jsonl'
    done=json.loads((folder/'completed.json').read_text());assert done['complete']
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    result['positional']={'completed_selected_cuda':done,'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'full_fp32_status':json.loads((args.root/'position-probe-fp32/skipped.json').read_text()),
        'selected_checks':{},'interpretation':'full positional aggregate remains unvalidated; do not relax the gate after seeing its failure'}
    for precision in ['bfloat16','float32']:
        selected=[r for r in rows if r['precision']==precision];a=np.array([r['dense_sparse_4d_error'] for r in selected])
        result['positional']['selected_checks'][precision]={'n_source_checks':len(selected),
            'median_error':float(np.median(a)),'maximum_error':float(max(a)),
            'above_fixed_1e_minus4_gate':int(sum(a>1e-4)),
            'all_2d_4d_identical':all(r['sparse_2d_4d_error']==0 and r['dense_2d_4d_error']==0 for r in selected)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
