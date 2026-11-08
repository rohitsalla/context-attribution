import copy
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from context_attribution.v4.branches import snapshot,restore,tree_hash,AnchoredStudy
from context_attribution.v4.evidence import global_lexical,oracle_sources,answer_quality
from context_attribution.v2.memory import Source
from context_attribution.v2.soft import SourceSoftMemory


class Tokens:
    def encode(self,text,add_special_tokens=False):return list(text.encode())


class ExactStateTests(unittest.TestCase):
    def test_optimizer_and_parameters_are_restored_together(self):
        pool=torch.nn.Linear(2,2);opt=torch.optim.AdamW(pool.parameters())
        pool(torch.ones(1,2)).sum().backward();opt.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
            state=snapshot(pool,opt)
            pool(torch.ones(1,2)).sum().backward();opt.step()
            receipt=restore(pool,opt,state)
            self.assertEqual(receipt['pool'],tree_hash(state['pool']))
            self.assertEqual(receipt['optimizer'],tree_hash(state['optimizer']))

    def test_zero_gate_is_exact_lexical_identity(self):
        from types import SimpleNamespace
        study=AnchoredStudy.__new__(AnchoredStudy);study.slots=3;study.literal={};study.rms=.02
        study.decoder=SimpleNamespace(tokenizer=Tokens(),device='cpu')
        study.emb=torch.nn.Embedding(256,4);study.pool=SourceSoftMemory(4,4,3)
        study.pool.register_parameter('residual_gate',torch.nn.Parameter(torch.tensor(0.)))
        feature=torch.randn(1,10,4);study.encode=lambda item:([feature],0.)
        item={'id':'x','question':'q','sources':[{'id':'d0','title':'A','text':'abc'}]}
        actual=study.components(item)[0]
        expected=study.emb(torch.tensor([[65,58,32]]))
        self.assertTrue(torch.equal(actual,expected))

    def test_global_extraction_recomputes_within_budget(self):
        sources=[Source('d0','A','A first fact.'),Source('d1','B','A different bridge fact.')]
        tokens=global_lexical(Tokens(),sources,'bridge',40)
        self.assertLessEqual(len(tokens),40)
        changed=bytes(global_lexical(Tokens(),sources[:1],'bridge',40)).decode()
        self.assertNotIn('[d1]',changed)

    def test_oracle_is_explicit_and_aliases_are_evaluation_only(self):
        item={'gold':'A','answer_aliases':['B'],'support_ids':['d0'],
              'sources':[{'id':'d0','sentences':['keep','remove']},{'id':'d1','sentences':['distractor']}],
              'supporting_facts':[{'source_id':'d0','sentence_index':0}]}
        sources=[Source('d0','A','keep remove'),Source('d1','B','distractor')]
        self.assertEqual(oracle_sources(item,sources),[Source('d0','A','keep')])
        self.assertEqual(answer_quality('B',item)['em'],1.)
