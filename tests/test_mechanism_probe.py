import copy
import unittest
from context_attribution.mechanism_probe import alias_item


class AliasDiagnosticTests(unittest.TestCase):
    def test_shared_entity_identity_is_preserved_without_mutating_original(self):
        item={'id':'x','split':'dev','question':'Who wrote Book One?','gold':'Anne Smith',
              'sources':[{'id':'d0','title':'Book One','text':'Book One was written by Anne Smith.'},
                         {'id':'d1','title':'Anne Smith','text':'Anne Smith is a writer.'}]}
        snapshot=copy.deepcopy(item);changed=alias_item(item)
        self.assertEqual(item,snapshot)
        self.assertIn(changed['sources'][0]['title'],changed['question'])
        self.assertIn(changed['gold'],changed['sources'][0]['text'])
        self.assertEqual(changed['gold'],changed['sources'][1]['title'])
        self.assertEqual(changed['split'],'dev')

    def test_boolean_answers_remain_boolean(self):
        item={'id':'x','split':'dev','question':'Is Book One old?','gold':'yes',
              'sources':[{'id':'d0','title':'Book One','text':'Book One is old.'}]}
        self.assertEqual(alias_item(item)['gold'],'yes')
