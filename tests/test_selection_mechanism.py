import unittest
from context_attribution.selection_mechanism import rank_chunks,pack_fixed
from context_attribution.frontier_probe import flatten
from context_attribution.v2.memory import Source
from context_attribution.v4.evidence import global_lexical


class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())


class SelectionMechanismTests(unittest.TestCase):
    def test_factual_identity_with_frozen_global_selector_including_fallback(self):
        tok=Tokenizer();sources=[Source('d0','River','Water flows. A bridge crosses it.'),Source('d1','Bridge','A bridge opened in 1932.')]
        for budget in [1,20,64,128,512]:
            ranking=rank_chunks(tok,sources,'When did the bridge open?')
            self.assertEqual(flatten(pack_fixed(ranking,['d0','d1'],budget)),global_lexical(tok,sources,'When did the bridge open?',budget))

    def test_repacking_keeps_original_priority_and_fills_freed_capacity(self):
        ranked=[('d0',0,(1,2,3)),('d1',0,(4,5)),('d2',0,(6,7,8))]
        self.assertEqual(flatten(pack_fixed(ranked,['d0','d1','d2'],5)),(1,2,3,4,5))
        self.assertEqual(flatten(pack_fixed(ranked,['d0','d1','d2'],5,'d0')),(4,5,6,7,8))
        self.assertEqual(pack_fixed([],[],5),[])
