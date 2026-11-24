import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from context_attribution.stability_study import BUDGETS, POLICIES, chosen_rows, donor_replacement
import review_stability as review


def row(y,x):
    ids=[f'd{i}' for i in range(len(y))]
    return {'source_ids':ids,'raw_effects':dict(zip(ids,y)),
            'budgets':{'512':{'effects':{'bm25_frozen':dict(zip(ids,x))}}}}


class StabilityTests(unittest.TestCase):
    def test_nonnegative_fit_is_question_weighted_l1_optimum(self):
        rng=np.random.default_rng(17)
        for _ in range(30):
            rows=[row(rng.normal(size=n).tolist(),rng.normal(size=n).tolist()) for n in [2,5,10]]
            alpha=review.fit_scale(rows,'bm25_frozen',512)
            loss=lambda a:np.mean([review.question_error(r,'bm25_frozen',512,a) for r in rows])
            candidates=[0.,1.]+[max(0.,r['raw_effects'][k]/x) for r in rows for k,x in r['budgets']['512']['effects']['bm25_frozen'].items() if x]
            self.assertLessEqual(loss(alpha),min(map(loss,candidates))+1e-12)

    def test_fit_degenerate_negative_and_upscale(self):
        self.assertEqual(review.fit_scale([row([2,3],[0,0])],'bm25_frozen',512),0.)
        self.assertEqual(review.fit_scale([row([-2,-4],[1,2])],'bm25_frozen',512),0.)
        self.assertEqual(review.fit_scale([row([4,8],[2,4])],'bm25_frozen',512),2.)

    def test_donor_preserves_survivors_and_exact_length(self):
        factual=[('d0',0,(1,2)),('d1',0,(3,4))]
        branch=[('d1',0,(3,4)),('d2',0,(5,6,7))]
        changed,added=donor_replacement(factual,branch,(20,21),1)
        self.assertEqual(changed,(3,4,21,20,21));self.assertEqual(added,3)
        unchanged,added=donor_replacement(factual,factual,(20,),0)
        self.assertEqual(unchanged,(1,2,3,4));self.assertEqual(added,0)

    def test_stride_shards_partition_both_splits(self):
        rows=[{'id':str(i),'split':s} for s in ['dev','test'] for i in range(20 if s=='dev' else 80)]
        parts=[chosen_rows(rows,k) for k in range(8)]
        keys=[(r['split'],r['id']) for part in parts for r in part]
        self.assertEqual(len(keys),len(set(keys)));self.assertEqual(len(keys),len(rows))

    def test_zero_scale_gets_fair_rank_tie_credit(self):
        r=row([4,1],[3,1]);self.assertEqual(review.top_credit(r,'bm25_frozen',512,0.),.5)

    def test_stratified_bootstrap_preserves_stratum_weights(self):
        rows=[{'stratum':'a'}]*3+[{'stratum':'b'}]*2
        result=review.interval(rows,[1,1,1,4,4]);self.assertAlmostEqual(result['mean'],2.2)
        self.assertTrue(all(abs(x-2.2)<1e-12 for x in result['stratified_question_bootstrap_95']))

    def test_analysis_fits_dev_only_and_handles_all_budgets(self):
        rows=[]
        for split in ('dev','test'):
            for index in range(2):
                effects={p:{'d0':4.,'d1':0.} for p in POLICIES}
                for p in ('bm25_fixed_rank','bm25_refitted','recomp_repacked'):
                    effects[p]={'d0':2.,'d1':0.} if split=='dev' else {'d0':1.,'d1':1.}
                budgets={str(b):{'effects':effects,'coverage_tokens':{'bm25':{'d0':4,'d1':0},'recomp':{'d0':4,'d1':0}},
                    'length_controls':{selector:{'source':'d0','replacement_tokens':2,'actual_effect':effects[policy]['d0'],'donor_effect':0.}
                                      for selector,policy in [('bm25','bm25_fixed_rank'),('recomp','recomp_repacked')]}} for b in BUDGETS}
                rows.append({'item_id':split+str(index),'split':split,'stratum':'a','source_ids':['d0','d1'],
                             'raw_effects':{'d0':2.,'d1':0.},'budgets':budgets})
        status,errors,test=review.analyze_cell(rows)
        self.assertEqual(status['budgets']['512']['bm25_frozen']['dev_fitted_nonnegative_scale'],.5)
        self.assertEqual(status['budgets']['512']['bm25_frozen']['rescaled_question_mae'],0.)
        self.assertEqual(status['budgets']['512']['bm25_refitted']['rescaled_question_mae'],1.)
        for r in rows:
            if r['split']=='test':r['raw_effects']['d0']=200.
        status,_,_=review.analyze_cell(rows)
        self.assertEqual(status['budgets']['512']['bm25_frozen']['dev_fitted_nonnegative_scale'],.5)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'CPU tensor smoke needs torch')
    def test_runner_smoke_and_checkpoint_resume_without_weights(self):
        import torch
        import context_attribution.stability_study as study
        import context_attribution.v2.hf as hf
        from context_attribution.selection_mechanism import rank_chunks,pack_fixed
        project=Path(__file__).resolve().parents[1]
        class Tokenizer:
            def encode(self,text,**kw):
                return [3+sum(word.encode())%28 for word in text.split()]
        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.ones(1));self.generation_config=SimpleNamespace(repetition_penalty=1.)
            def forward(self,input_ids,position_ids,attention_mask,use_cache,logits_to_keep):
                assert attention_mask.ndim==4 and position_ids.shape==input_ids.shape
                x=torch.arange(32,dtype=torch.float32);score=x/20+torch.cos(x)*(int(input_ids.sum())%97)/97
                return SimpleNamespace(logits=score[None,None,:].repeat(1,logits_to_keep,1))
        class Reader:
            def __init__(self,path,revision,model_id=None):
                self.tokenizer=Tokenizer();self.device='cpu';self.model=LM();self.revision=revision
            def runtime(self):return {'dtype':'torch.float32','revision':self.revision,'device':'cpu'}
            def qa_ids(self,question,memory):return [1]+list(memory)+[2]
            def generate(self,prefix):return {'tokens':[5,6],'text':'answer','cap_reached':False}
        class Selector:
            def __init__(self,*args):pass
            def rank(self,tokenizer,sources,question):return list(reversed(rank_chunks(tokenizer,sources,question)))
            def pack(self,rank,sources,budget):return pack_fixed(rank,[s.sid for s in sources],budget)
        def cohort(project,name):
            rows=[{'id':name+str(i),'question':'test query','gold':'answer','answer_aliases':[], 'split':'test','hops':2,
                'sources':[{'id':'d0','title':'first','text':'First passage.'},{'id':'d1','title':'second','text':'Second passage.'}]} for i in [0,1]]
            return rows,{'donor_map':{name:{rows[0]['id']:rows[1]['id'],rows[1]['id']:rows[0]['id']}}}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out';paths=Path(tmp)/'paths.json';paths.write_text(json.dumps({'qwen7':'fake'}))
            argv=['study','--project',str(project),'--paths',str(paths),'--recomp-path','fake','--model','qwen7','--smoke','--output',str(output)]
            with patch.object(hf,'HFModel',Reader),patch.object(study,'RecompSelector',Selector),patch.object(study,'read_cohort',cohort),patch.object(sys,'argv',argv):
                original_generate=Reader.generate;calls=[0]
                def interrupted_generate(self,prefix):
                    calls[0]+=1
                    if calls[0]>9:raise RuntimeError('simulated interruption during second question')
                    return original_generate(self,prefix)
                with patch.object(Reader,'generate',interrupted_generate):
                    with self.assertRaises(RuntimeError):study.main()
                prefix=(output/'traces.jsonl').read_text()
                self.assertEqual(len(prefix.splitlines()),1)
                self.assertFalse((output/'completed.json').exists())
                study.main();saved=(output/'traces.jsonl').read_text()
                self.assertTrue(saved.startswith(prefix))
                rows=[json.loads(s) for s in saved.splitlines()]
                expected=sum(len(chosen_rows(cohort(project,d)[0],smoke=True)) for d in study.DATASETS)
                self.assertEqual(len(rows),expected)
                self.assertEqual(json.loads((output/'completed.json').read_text())['n'],expected)
                self.assertTrue(all(set(r['budgets'])==set(map(str,BUDGETS)) for r in rows))
                with patch.object(Reader,'generate',side_effect=AssertionError('Completed questions must be skipped')):
                    study.main()
                self.assertEqual((output/'traces.jsonl').read_text(),saved)


if __name__=='__main__':unittest.main()
