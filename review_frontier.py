"""Offline review; incomplete experiments cannot enter frontier comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np
from context_attribution.v2.metrics import normalize,effect_comparison


def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20261108)
    return {'mean':float(a.mean()),'question_bootstrap_95':np.quantile(
        a[rng.integers(len(a),size=(5000,len(a)))].mean(axis=1),[.025,.975]).tolist()}


def mae(row):
    return st.mean(abs(row['effects'][k]-v) for k,v in row['raw_effects'].items())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    result={'scope':'adaptive development; two planned contrasts per study; unadjusted exploratory question intervals',
            'studies':{}}
    for name in ['hotpot_qwen7','hotpot_llama8','musique_qwen7','musique_llama8']:
        folder=args.root/name;trace=folder/'traces.jsonl'
        rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
        status={'complete':(folder/'completed.json').exists(),'available_rows':len(rows)}
        result['studies'][name]=status
        if not status['complete']:continue
        expected=32 if name.startswith('musique_') else 96
        done=json.loads((folder/'completed.json').read_text());assert done['complete'] and done['n']==expected
        groups={method:{r['item_id']:r for r in rows if r['method']==method} for method in
                ['raw','global_128_recomputed','global_256_recomputed','global_512_recomputed','global_1024_recomputed','global_512_frozen']}
        ids=sorted(groups['raw']);assert len(ids)==expected and all(set(g)==set(ids) for g in groups.values())
        for qid in ids:
            ref=groups['raw'][qid]
            assert ref['raw_identity_endpoint_error']<=1e-7
            for group in groups.values():
                r=group[qid];assert r['common_target_tokens']==ref['common_target_tokens']
                assert r['raw_effects']==ref['raw_effects'] and r['same_shape_repeat_error']<=1e-7
                assert r['effects'].keys()==r['raw_effects'].keys()
            a,b=groups['global_512_frozen'][qid],groups['global_512_recomputed'][qid]
            assert a['factual_memory_sha256']==b['factual_memory_sha256'] and a['answer']==b['answer']
        status['trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest();status['methods']={}
        for method,group in groups.items():
            values=list(group.values());den=sum(abs(v) for r in values for v in r['raw_effects'].values())
            err=sum(abs(r['effects'][k]-v) for r in values for k,v in r['raw_effects'].items())
            hits=[effect_comparison(r['raw_effects'],r['effects'])['top1_hit'] for r in values]
            valid_hits=[h for h in hits if h is not None]
            status['methods'][method]={'n':len(values),'f1':st.mean(r['qa']['f1'] for r in values),
                'mean_positions':st.mean(r['memory_positions'] for r in values),
                'mean_question_mae':st.mean(mae(r) for r in values),'pooled_nmae':err/max(den,1e-12),
                'raw_effect_l1':den,'top_source_non_negligible_n':len(valid_hits),
                'top_source_hit_rate':st.mean(valid_hits) if valid_hits else None}
        status['contrasts']={}
        for left,right in [('global_1024_recomputed','global_128_recomputed'),('global_512_frozen','global_512_recomputed')]:
            a,b=groups[left],groups[right];cohorts={'all':ids,
                'same_normalized_factual_answer':[q for q in ids if normalize(a[q]['answer']['text'])==normalize(b[q]['answer']['text'])],
                'both_exactly_correct':[q for q in ids if a[q]['qa']['em']==b[q]['qa']['em']==1]}
            status['contrasts'][left+'-minus-'+right]={slice_name:{'n':len(selected),
                'mae':interval([mae(a[q])-mae(b[q]) for q in selected]),
                'qa_f1':interval([a[q]['qa']['f1']-b[q]['qa']['f1'] for q in selected])}
                for slice_name,selected in cohorts.items() if selected}
        # Exact algebra on matched factual memories, not a unique causal decomposition.
        policy_diffs=[];direct_mass=policy_mass=dynamic_mass=0.;excluded_raw=all_raw=0.;excluded_n=0
        flips=0;active=0;raw_omitted_by_question=[]
        for qid in ids:
            frozen=groups['global_512_frozen'][qid];dynamic=groups['global_512_recomputed'][qid]
            diffs=[dynamic['effects'][k]-frozen['effects'][k] for k in frozen['effects']]
            policy_diffs.append(st.mean(abs(v) for v in diffs));policy_mass+=sum(abs(v) for v in diffs)
            direct_mass+=sum(abs(v) for v in frozen['effects'].values())
            dynamic_mass+=sum(abs(v) for v in dynamic['effects'].values())
            omitted=[k for k,v in frozen['source_positions'].items() if v==0]
            assert all(abs(frozen['effects'][k])<=1e-7 for k in omitted)
            omitted_raw=sum(abs(frozen['raw_effects'][k]) for k in omitted)
            excluded_raw+=omitted_raw;all_raw+=sum(abs(v) for v in frozen['raw_effects'].values());excluded_n+=len(omitted)
            raw_omitted_by_question.append({'item_id':qid,'omitted_raw_l1':omitted_raw})
            for k in frozen['effects']:
                f,d=frozen['effects'][k],dynamic['effects'][k]
                if abs(f)>.001 and abs(d)>.001:active+=1;flips+=int(f*d<0)
        status['selection_policy_diagnostic']={'mean_question_absolute_dynamic_minus_frozen_effect':interval(policy_diffs),
            'summed_absolute_frozen_effect':direct_mass,'summed_absolute_policy_difference':policy_mass,
            'summed_absolute_recomputed_effect':dynamic_mass,
            'n_unrepresented_source_deletions':excluded_n,'raw_l1_unrepresented_fraction':excluded_raw/max(all_raw,1e-12),
            'n_both_effects_above_0_001':active,'n_sign_disagreements_in_that_subset':flips,
            'interpretation':'dynamic = frozen + policy difference; magnitude ratios need not sum to one due to cancellation. Factual QA is identical; intervened lengths are not matched. This does not isolate IDF, refill, layout or unique causal mediation.'}
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
