import unittest
from prepare_v2_hotpot import convert


class HotpotIngestionTests(unittest.TestCase):
    def test_source_duplicates_and_annotation_issues_are_not_silently_dropped(self):
        row={'id':'example','question':'Question','answer':'Answer','type':'bridge','level':'hard',
             'context':{'title':['A','A','B'],'sentences':[['one'],['one'],['two']]},
             'supporting_facts':{'title':['A','B'],'sent_id':[0,4]}}
        r=convert(row,'dev')
        self.assertEqual(len(r['sources']),3)
        self.assertEqual(r['duplicate_title_count'],1)
        self.assertEqual(len(r['annotation_issues']),1)
        self.assertEqual(set(r['support_ids']),{'d0','d1'})
