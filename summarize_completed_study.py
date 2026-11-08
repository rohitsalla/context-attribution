"""Validate completed outputs and produce a public summary without private paths."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
COMMIT='40c5476bee6d1635fbb9fa7fbe2700651b9830ac'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'study_results.json')
    args=p.parse_args();root=args.results
    report=json.loads((root/'report.json').read_text())
    assert report['complete'], 'Study is incomplete'
    for name,dataset,expected in [('main-128','main.jsonl',120),('main-256','main.jsonl',120),
        ('main-512','main.jsonl',120),('stress','stress.jsonl',180),('stochastic','stochastic.jsonl',8)]:
        manifest=json.loads((root/name/'manifest.json').read_text())
        assert manifest['backend']=='hf' and manifest['research_evidence'] is True
        assert manifest['likelihood_check']['passed']
        assert manifest['input_dataset_sha256']==hashlib.sha256((ROOT/'data/main-v1'/dataset).read_bytes()).hexdigest()
        for filename,digest in manifest['source_sha256'].items():
            original=subprocess.check_output(['git','show',f'{COMMIT}:src/context_attribution/{filename}'],cwd=ROOT)
            assert hashlib.sha256(original).hexdigest()==digest,filename
        rows=read_rows(root/name/'traces.jsonl')
        assert len(rows)==expected and len({(r['item_id'],r['cap']) for r in rows})==expected
        for row in rows:
            assert len(row['effects'])==6
            for branch in row['reference'].values():
                assert len(branch['trials'])==(5 if name=='stochastic' else 2)
                assert all(t['fixed_answer']==row['original']['fixed_answer'] for t in branch['trials'])
    main_rows={cap:read_rows(root/f'main-{cap}'/'traces.jsonl') for cap in [128,256,512]}
    online=read_rows(root/'online/traces.jsonl');assert len(online)==12
    assert all(not r['reference_lookups_used'] for r in online)
    summary={'complete':True,'source_commit':COMMIT,'counts':report['counts'],
             'bootstrap_seed':20261108,'bootstrap_draws':20000,'bootstrap_unit':'base question',
             'main_results':{},'online_results':[],'human_audit_claimed':False}
    rng=np.random.default_rng(20261108)
    for label,subset in [('all',main_rows[128]),('test',[r for r in main_rows[128] if r['split']=='test'])]:
        def values(method,budget):
            return np.array([next(c['hit'] for c in row['comparisons'] if c['method']==method and c['budget']==budget)
                             for row in subset],dtype=float)
        delta=values('recursive_surrogate',1)-values('source_id',1)
        boot=rng.choice(delta,(20000,len(delta))).mean(1)
        summary['main_results'][label]={'n':len(subset),'accuracy':float(np.mean([r['correct'] for r in subset])),
            'discovery':{m:{str(b):float(values(m,b).mean()) for b in [1,2,3]}
                         for m in ['recursive_surrogate','source_id','random_exact_expectation']},
            'paired_gain_over_source_id_95ci':np.quantile(boot,[.025,.975]).tolist()}
    summary['identical_original_summaries_across_caps']=all(a['original']['summary']==b['original']['summary']
        for cap in [256,512] for a,b in zip(main_rows[128],main_rows[cap]))
    summary['identical_effects_across_caps']=all(a['effects']==b['effects']
        for cap in [256,512] for a,b in zip(main_rows[128],main_rows[cap]))
    summary['identical_original_answers_across_caps']=all(a['original']['fixed_answer']==b['original']['fixed_answer']
        for cap in [256,512] for a,b in zip(main_rows[128],main_rows[cap]))
    summary['mean_original_summary_tokens']=float(np.mean([r['original']['summary_info']['output_tokens'] for r in main_rows[128]]))
    for method in ['recursive_surrogate','source_id','lexical','random']:
        for budget in [1,2,3]:
            subset=[r for r in report['online_comparisons'] if r['method']==method and r['budget']==budget
                    and r['matching_answer_span'] and r['hit'] is not None]
            summary['online_results'].append({'method':method,'budget':budget,'n':len(subset),
                'hit_rate':float(np.mean([r['hit'] for r in subset])),
                'mean_time_ratio_to_full_replay':float(np.mean([r['time_ratio'] for r in subset]))})
    summary['scope']='Frozen 120-item exploratory set: 80 fictional, 40 assistant-checked facts-page excerpts; no human audit or long-form prose benchmark.'
    args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Validated all outputs and wrote',args.output)


if __name__=='__main__':main()
