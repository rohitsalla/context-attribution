import unittest
from context_attribution.position_probe import remove_layout


class PositionProtocolTests(unittest.TestCase):
    def test_removed_source_does_not_renumber_remaining_positions(self):
        tokens,positions,mask=remove_layout([10,11,12,13,14,15],(1,3))
        self.assertEqual(tokens,[10,13,14,15]);self.assertEqual(positions,[0,3,4,5])
        self.assertEqual(mask,[1,0,0,1,1,1])
        self.assertEqual([t for t,m in zip([10,11,12,13,14,15],mask) if m],tokens)
        with self.assertRaises(ValueError):remove_layout([1,2],(0,3))
