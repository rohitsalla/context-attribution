import importlib.util
import json
from pathlib import Path
import unittest
from context_attribution.data import validate
from context_attribution.experiment import answer_correct

ROOT=Path(__file__).resolve().parents[1]


class MainDatasetTests(unittest.TestCase):
    def setUp(self):
        self.path=ROOT/'data/main-v1'
        self.rows=[json.loads(s) for s in (self.path/'main.jsonl').read_text().splitlines()]

    def test_counts_family_and_base_separation(self):
        validate(self.rows)
        self.assertEqual(len(self.rows),120)
        self.assertEqual(sum(r['split']=='dev' for r in self.rows),40)
        dev={r['family'] for r in self.rows if r['split']=='dev'}
        test={r['family'] for r in self.rows if r['split']=='test'}
        self.assertFalse(dev & test)
        self.assertEqual(len({r['base_id'] for r in self.rows}),120)
        for r in self.rows[:80]: self.assertFalse(any(c.isdigit() for c in r['gold']))

    def test_natural_provenance_and_city_aliases(self):
        natural=[r for r in self.rows if r['data_kind']=='natural_text_facts_page']
        self.assertEqual(len(natural),40)
        for r in natural:
            self.assertFalse(r['human_audited'])
            self.assertTrue(all(s['url'].startswith('https://www.nobelprize.org/') for s in r['sources']))
            self.assertTrue(answer_correct(r,r['gold']))
            for alias in r['gold_aliases']: self.assertTrue(answer_correct(r,alias))

    def test_duplicate_and_replacement_controls(self):
        rows=[json.loads(s) for s in (self.path/'stress.jsonl').read_text().splitlines()]
        validate(rows);self.assertEqual(len(rows),60)
        base={r['id']:r for r in self.rows}
        for r in rows:
            if r['variant']=='duplicate':
                blocks={s['id']:s['text'] for s in r['sources']}
                self.assertEqual(blocks['s2'],blocks['s6'])
                self.assertEqual(r['diagnostic_groups'],[['s2','s6']])
            if r['variant']=='fact_replacement':
                old=base[r['counterfactual_of']]['gold']
                self.assertNotIn(old,' '.join(s['text'] for s in r['sources']))
                self.assertIn(r['gold'],' '.join(s['text'] for s in r['sources']))
