import unittest
from context_attribution.frontier_probe import selected_chunks,flatten
from context_attribution.v4.evidence import global_lexical
from context_attribution.v2.memory import Source


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())


class FrontierControls(unittest.TestCase):
    def test_provenance_matches_frozen_selector_through_fallback_and_deletions(self):
        tok=CharacterTokenizer();sources=[Source('d0','River','Water flows. A bridge crosses it.'),
                                         Source('d1','Bridge','A bridge opened in 1932.'),Source('d2','Empty','')]
        for kept in [sources,sources[1:],sources[:1],[]]:
            for budget in [1,16,50,128,512]:
                chunks=selected_chunks(tok,kept,'When did the bridge open?',budget)
                self.assertEqual(flatten(chunks),global_lexical(tok,kept,'When did the bridge open?',budget))
                self.assertLessEqual(len(flatten(chunks)),budget)

    def test_frozen_selection_drops_only_owned_chunks_without_refill(self):
        chunks=[('d0',0,(1,2)),('d1',0,(3,4)),('d0',1,(5,))]
        self.assertEqual(flatten(chunks,'d0'),(3,4))
        self.assertEqual(flatten(chunks,'d1'),(1,2,5))
        self.assertEqual(flatten(chunks,'absent'),flatten(chunks))
