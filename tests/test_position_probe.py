import unittest
from context_attribution.position_probe import remove_layout,score_inputs


class PositionProtocolTests(unittest.TestCase):
    def test_gapped_positions_always_supply_an_explicit_single_stream_mask(self):
        ids,positions,mask=score_inputs([10,11,12],[0,4,5])
        self.assertEqual(mask.tolist(),[[1,1,1]])
        self.assertEqual(positions.tolist(),[[0,4,5]])
        _,_,dense=score_inputs([10,11,12],[0,1,2],[1,0,1])
        self.assertEqual(dense.tolist(),[[1,0,1]])

    def test_removed_source_does_not_renumber_remaining_positions(self):
        tokens,positions,mask=remove_layout([10,11,12,13,14,15],(1,3))
        self.assertEqual(tokens,[10,13,14,15]);self.assertEqual(positions,[0,3,4,5])
        self.assertEqual(mask,[1,0,0,1,1,1])
        self.assertEqual([t for t,m in zip([10,11,12,13,14,15],mask) if m],tokens)
        with self.assertRaises(ValueError):remove_layout([1,2],(0,3))
