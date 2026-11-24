"""Evaluate frozen hypotheses with dev-only effect rescaling and fresh test questions."""
import argparse
import json
from pathlib import Path
from statistics import mean
import numpy as np
from context_attribution.generalization import atomic_text, POLICIES, omissions
from context_attribution.heldout_followup import digest
from context_attribution.stability_study import BUDGETS, DATASETS, MODELS, SHARDS, chosen_rows, read_cohort


def fit_scale(rows, policy, budget):
    pairs=[]
    for r in rows:
        values=r['budgets'][str(budget)]['effects'][policy];keys=r['source_ids']
        for k in keys:
            x=values[k]
            if x!=0: pairs.append((r['raw_effects'][k]/x,abs(x)/len(keys)))
    if not pairs:return 0.
    pairs.sort();half=sum(w for _,w in pairs)/2;mass=0.
    for ratio,w in pairs:
        mass+=w
        if mass>=half:return max(0.,ratio)
    raise AssertionError('Weighted median failed')


def question_error(row, policy, budget, alpha=1.):
    estimates=row['budgets'][str(budget)]['effects'][policy]
    return mean(abs(alpha*estimates[k]-row['raw_effects'][k]) for k in row['source_ids'])


def zero_error(row):return mean(abs(row['raw_effects'][k]) for k in row['source_ids'])


def interval(rows, values):
    a=np.asarray(values,dtype=float);assert len(a)==len(rows) and np.isfinite(a).all()
    rng=np.random.default_rng(20261124);samples=np.zeros(5000)
    for stratum in sorted({str(r['stratum']) for r in rows}):
        ix=np.asarray([i for i,r in enumerate(rows) if str(r['stratum'])==stratum])
        samples+=a[ix[rng.integers(len(ix),size=(5000,len(ix))) ]].sum(axis=1)/len(a)
    return {'mean':float(a.mean()),'stratified_question_bootstrap_95':np.quantile(samples,[.025,.975]).tolist()}


def top_credit(row, policy, budget, alpha):
    ref=row['raw_effects'];truth=max(abs(ref[k]) for k in row['source_ids'])
    if truth<=1e-6:return None
    values={k:abs(alpha*row['budgets'][str(budget)]['effects'][policy][k]) for k in row['source_ids']}
    maximum=max(values.values());ties=[k for k,v in values.items() if abs(v-maximum)<=1e-12]
    correct={k for k in row['source_ids'] if abs(abs(ref[k])-truth)<=1e-12}
    return len(correct.intersection(ties))/len(ties)


def load_cell(project, model, dataset):
    items,data_manifest=read_cohort(project,dataset);rows=[];missing=[]
    for shard in range(SHARDS):
        folder=project/'results/stability'/model/dataset/str(shard);done=folder/'completed.json'
        if not done.exists():missing.append(shard);continue
        m=json.loads((folder/'manifest.json').read_text());receipt=json.loads(done.read_text())
        expected=chosen_rows(items,shard);values=[json.loads(s) for s in (folder/'traces.jsonl').read_text().splitlines()]
        assert receipt['complete'] and receipt['n']==len(expected)==len(values)
        assert m['model']==model and m['dataset']==dataset and m['shard']==shard and not m['smoke']
        assert m['data_manifest_sha256']==digest((project/'data/stability/manifest.json').read_bytes())
        assert m['runtime']['dtype']=='torch.float32' and m['python_hash_seed']=='0'
        runtime=m['runtime'];required=json.loads((project/'research/v5/STABILITY_RUNTIME.json').read_text())
        assert {'python_major_minor':'.'.join(runtime['python'].split('.')[:2]),**{k:runtime[k] for k in ('torch','transformers','cuda')}}==required
        registry=json.loads((project/'research/v4/GENERALIZATION_MODELS.json').read_text())
        assert m['runtime']['revision']==registry[model]['revision']
        assert m['budgets']==list(BUDGETS) and m['policies']==list(POLICIES)
        for name,sha in m['source_hashes'].items():assert digest((project/name).read_bytes())==sha
        assert [r['item_id'] for r in values]==[r['id'] for r in expected]
        for row,item in zip(values,expected,strict=True):
            assert row['dataset']==dataset and row['split']==item['split']
            assert row['source_ids']==[s['id'] for s in item['sources']]
            assert row['repeat_error']<=1e-7 and row['raw_endpoint_error']<=1e-7
            keys={','.join(o) for o in omissions(item)};assert set(row['raw_effects'])==keys
            assert row['raw_answer']['tokens']
            for budget in BUDGETS:
                b=row['budgets'][str(budget)];assert set(b['effects'])==set(POLICIES)
                for policy in POLICIES:
                    assert set(b['effects'][policy])==keys
                    assert all(np.isfinite(v) for v in b['effects'][policy].values())
                    assert max(b['intervention_positions'][policy].values())<=budget
                for selector in ('bm25','recomp'):
                    assert b['factual_positions'][selector]<=budget
                    cov=b['coverage_tokens'][selector];assert set(cov)==set(row['source_ids'])
                    for sid in row['source_ids']:
                        if cov[sid]==0:assert b['effects'][selector+'_frozen'][sid]==0.
        rows.extend(values)
    status={'complete':not missing,'available_questions':len(rows),'missing_shards':missing}
    return status,rows


def analyze_cell(rows):
    dev=sorted([r for r in rows if r['split']=='dev'],key=lambda r:r['item_id'])
    test=sorted([r for r in rows if r['split']=='test'],key=lambda r:r['item_id'])
    assert dev and test and not {r['item_id'] for r in dev}&{r['item_id'] for r in test}
    status={'n_dev':len(dev),'n_test':len(test),'zero_question_mae':mean(map(zero_error,test)),'budgets':{}}
    errors={}
    for budget in BUDGETS:
        methods={};errors[budget]={}
        for policy in POLICIES:
            alpha=fit_scale(dev,policy,budget)
            loss=lambda scale:mean(question_error(r,policy,budget,scale) for r in dev)
            assert loss(alpha)<=min(loss(0.),loss(1.))+1e-8
            uncal=[question_error(r,policy,budget) for r in test];cal=[question_error(r,policy,budget,alpha) for r in test]
            errors[budget][policy]={'uncal':uncal,'cal':cal}
            denominator=sum(abs(r['raw_effects'][k]) for r in test for k in r['source_ids'])
            numerator=sum(abs(alpha*r['budgets'][str(budget)]['effects'][policy][k]-r['raw_effects'][k]) for r in test for k in r['source_ids'])
            credit=[x for r in test if (x:=top_credit(r,policy,budget,alpha)) is not None]
            selector=policy.split('_')[0];omitted=[(r,k) for r in test for k in r['source_ids'] if r['budgets'][str(budget)]['coverage_tokens'][selector][k]==0]
            methods[policy]={'dev_fitted_nonnegative_scale':alpha,'uncalibrated_question_mae':mean(uncal),'rescaled_question_mae':mean(cal),
                'rescaled_pooled_nmae_vs_zero':numerator/max(denominator,1e-12),'rescaled_top1_tie_credit':mean(credit) if credit else None,
                'top1_eligible_questions':len(credit),'omitted_source_fraction':len(omitted)/sum(len(r['source_ids']) for r in test),
                'mean_abs_raw_effect_on_omitted_sources':mean(abs(r['raw_effects'][k]) for r,k in omitted) if omitted else None}
        status['budgets'][str(budget)]=methods
    status['replacement_controls']={}
    for selector in ('bm25','recomp'):
        controls=[r['budgets']['512']['length_controls'][selector] for r in test]
        deltas=[]
        for row,c in zip(test,controls,strict=True):
            policy='bm25_fixed_rank' if selector=='bm25' else 'recomp_repacked'
            assert abs(c['actual_effect']-row['budgets']['512']['effects'][policy][c['source']])<=1e-7
            ref=row['raw_effects'][c['source']]
            deltas.append(abs(c['actual_effect']-ref)-abs(c['donor_effect']-ref))
        status['replacement_controls'][selector]={'questions':len(test),'with_new_tokens':sum(c['replacement_tokens']>0 for c in controls),
            'actual_minus_donor_absolute_raw_effect_error':interval(test,deltas),
            'interpretation':'fixed branch length/positions, varied new content; donor is not neutral'}
    return status,errors,test


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();project=a.project
    report={'scope':'prospective new-cohort study; intervals condition on fixed development-fitted scales and fixed readers',
        'source_sha256':digest((project/'src/context_attribution/stability_study.py').read_bytes()),
        'data_manifest_sha256':digest((project/'data/stability/manifest.json').read_bytes()),'cells':{},'dataset_model_average':{}}
    all_errors={};tests={}
    for model in MODELS:
        for dataset in DATASETS:
            key=model+'/'+dataset;status,rows=load_cell(project,model,dataset)
            if status['complete']:
                details,errors,test=analyze_cell(rows);status.update(details);all_errors[(model,dataset)]=errors;tests[(model,dataset)]=test
            report['cells'][key]=status
    complete=all(c['complete'] for c in report['cells'].values())
    for dataset in DATASETS:
        if any((m,dataset) not in all_errors for m in MODELS):
            report['dataset_model_average'][dataset]={'complete':False};continue
        reference=tests[(MODELS[0],dataset)];ids=[r['item_id'] for r in reference]
        assert all([r['item_id'] for r in tests[(m,dataset)]]==ids for m in MODELS)
        contrasts={}
        for budget in BUDGETS:
            out={}
            for selector,left,right in [('bm25','bm25_frozen','bm25_refitted'),('recomp','recomp_frozen','recomp_repacked')]:
                for kind in ('uncal','cal'):
                    values=[mean(all_errors[(m,dataset)][budget][left][kind][i]-all_errors[(m,dataset)][budget][right][kind][i] for m in MODELS) for i in range(len(ids))]
                    out[selector+'_'+kind+'_frozen_minus_recomputed']=interval(reference,values)
                values=[mean(all_errors[(m,dataset)][budget][left]['cal'][i]-zero_error(tests[(m,dataset)][i]) for m in MODELS) for i in range(len(ids))]
                out[selector+'_cal_frozen_minus_zero']=interval(reference,values)
            contrasts[str(budget)]=out
        report['dataset_model_average'][dataset]={'complete':True,'contrasts':contrasts}
    def passed(suffix):
        return complete and all(report['dataset_model_average'][d]['contrasts']['512'][s+suffix]['stratified_question_bootstrap_95'][1]<0 for d in DATASETS for s in ('bm25','recomp'))
    report['complete']=complete
    report['primary_rescaling_robust_gain']='supported' if passed('_cal_frozen_minus_recomputed') else 'criterion unmet' if complete else 'pending'
    report['primary_absolute_usefulness_vs_zero']='supported' if passed('_cal_frozen_minus_zero') else 'criterion unmet' if complete else 'pending'
    report['limits']=['Effect rescaling uses dev raw-effect labels; it is not temperature/probability calibration.',
        'Positive rescaling does not improve rankings. All-zero predictions receive fair fractional tie credit.',
        'Matched-length cross-question donors change content; they are not neutral padding or proof of a pure length effect.',
        'Budgets other than 512 and per-reader results are secondary; test scores are not used to fit scales.']
    out=project/'results/stability';out.mkdir(parents=True,exist_ok=True);atomic_text(out/'review.json',json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
