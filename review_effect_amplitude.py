"""Post-hoc cross-fitted oracle effect rescaling, not probability calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np


def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261109)
    means=a[rng.integers(len(a),size=(5000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'conditional_question_bootstrap_95':np.quantile(means,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result={'scope':'post-hoc adaptive diagnostic on completed development studies; uses raw-effect labels from other dev questions, not a deployable baseline',
        'fit':'nonnegative scalar minimizing mean per-question L1 effect error on four hash folds, evaluated on the fifth',
        'fold_rule':'SHA256(effect-amplitude-cv-v1: + question_id), first eight bytes big-endian modulo five',
        'limitations':'effect rescaling is not temperature calibration or a normalized answering distribution. Positive scales retain rankings. Bootstrap conditions on fitted CV scales and is not uncertainty for the full fitting procedure.',
        'studies':{}}
    for folder in sorted(args.root.glob('*_*')):
        if not folder.is_dir() or not (folder/'completed.json').exists():continue
        trace=folder/'traces.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
        methods=sorted({r['method'] for r in rows if r['method']!='raw'})
        groups={m:{r['item_id']:r for r in rows if r['method']==m} for m in methods}
        ids=sorted(groups[methods[0]]);fold={q:int.from_bytes(hashlib.sha256(('effect-amplitude-cv-v1:'+q).encode()).digest()[:8],'big')%5 for q in ids}
        out={'n':len(ids),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'methods':{}}
        errors={}
        for method,group in groups.items():
            assert set(group)==set(ids)
            predictions={};fits=[]
            for held in range(5):
                train=[q for q in ids if fold[q]!=held];test=[q for q in ids if fold[q]==held]
                assert train and test and not set(train)&set(test)
                pairs=[]
                for q in train:
                    r=group[q];n=len(r['effects'])
                    pairs.extend((r['raw_effects'][k]/s,abs(s)/n) for k,s in r['effects'].items() if abs(s)>1e-12)
                pairs.sort();half=sum(w for _,w in pairs)/2;mass=0.;alpha=0.
                for ratio,w in pairs:
                    mass+=w
                    if mass>=half:alpha=max(0.,ratio);break
                def loss(scale):
                    return st.mean(st.mean(abs(scale*s-group[q]['raw_effects'][k]) for k,s in group[q]['effects'].items()) for q in train)
                assert loss(alpha)<=min(loss(0.),loss(1.))+1e-8
                fits.append({'held_out_fold':held,'n_train':len(train),'n_test':len(test),'scale':alpha})
                for q in test:predictions[q]={k:alpha*s for k,s in group[q]['effects'].items()}
            den=sum(abs(v) for q in ids for v in group[q]['raw_effects'].values())
            err=sum(abs(predictions[q][k]-v) for q in ids for k,v in group[q]['raw_effects'].items())
            errors[method]={q:st.mean(abs(predictions[q][k]-v) for k,v in group[q]['raw_effects'].items()) for q in ids}
            out['methods'][method]={'pooled_nmae_after_cross_fitting':err/max(den,1e-12),
                'mean_question_mae_after_cross_fitting':st.mean(errors[method].values()),'fold_fits':fits}
        out['cross_fitted_1024_minus_128_mae']=interval([errors['global_1024_recomputed'][q]-errors['global_128_recomputed'][q] for q in ids])
        result['studies'][folder.name]=out
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
