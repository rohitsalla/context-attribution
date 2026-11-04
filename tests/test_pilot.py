import copy
import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from context_attribution.data import make_pilot, validate
from context_attribution.backend import FixtureBackend
from context_attribution.experiment import branch, run_item, discovery
from context_attribution.online import run_online
from context_attribution.attribution import logit_from_logprob, sentences


class PilotTests(unittest.TestCase):
    def test_fact_page_abbreviations_remain_in_sentences(self):
        self.assertEqual(sentences('Dr. A. B. Names works in St. Petersburg [s2]. Project X is led by Names [s1].'),
                         ['Dr. A. B. Names works in St. Petersburg [s2].','Project X is led by Names [s1].'])
    def test_split_and_source_contract(self):
        items = make_pilot()
        validate(items)
        self.assertEqual(len(items), 20)
        self.assertEqual(sum(i['split'] == 'dev' for i in items), 8)
        items[1]['split'] = 'test'
        with self.assertRaises(AssertionError):
            validate(items)

    def test_replay_fixed_answer_noise_and_cost(self):
        item, backend = make_pilot()[0], FixtureBackend()
        row = run_item(backend, item, 128, samples=12)
        self.assertTrue(row['correct'])
        self.assertEqual(row['effects']['s6'], 0)
        self.assertGreater(row['effects']['s1'], 0)
        self.assertGreater(row['effects']['s2'], 0)
        self.assertEqual(row['replay_noise'], 1e-6)
        for ref in row['reference'].values():
            for trial in ref['trials']:
                self.assertEqual(trial['fixed_answer'], row['original']['fixed_answer'])
                self.assertEqual(trial['cost']['calls'], 3)
        selected = next(c for c in row['comparisons'] if c['method'] == 'recursive_surrogate' and c['budget'] == 1)
        self.assertEqual(sum(v is None for v in selected['verified_effects'].values()), 5)
        self.assertGreater(selected['selection_overhead']['calls'], 0)
        self.assertEqual(selected['one_replay_per_source_cost']['calls'], 3)

    def test_duplicate_group_removal(self):
        item, backend = copy.deepcopy(make_pilot()[0]), FixtureBackend()
        duplicate = next(s for s in item['sources'] if s['id'] == 's6')
        duplicate['text'] = next(s['text'] for s in item['sources'] if s['id'] == 's2')
        original = branch(backend, item, item['sources'], 128)
        single = branch(backend, item, [s for s in item['sources'] if s['id'] != 's2'], 128, original['fixed_answer'])
        group = branch(backend, item, [s for s in item['sources'] if s['id'] not in ['s2','s6']], 128, original['fixed_answer'])
        self.assertEqual(single['logprob'], original['logprob'])
        self.assertLess(group['logprob'], original['logprob'])
        item['diagnostic_groups']=[['s2','s6']]
        item['irrelevant_id']='s5';item['support_ids']=['s1','s2','s6']
        row=run_item(backend,item,128,samples=8)
        self.assertEqual(row['effects']['s2'],0)
        self.assertTrue(all(x>0 for x in row['group_diagnostics'][0]['paired_effects']))

    def test_online_selective_replay_keeps_untested_effects_unknown(self):
        row=run_online(FixtureBackend(),make_pilot()[0])
        self.assertFalse(row['reference_lookups_used'])
        for method in row['methods']:
            for result in method['budgets']:
                budget=result['budget']
                self.assertEqual(len(result['untested']),6-budget)
                self.assertEqual(sum(v is not None for v in result['verified_effects'].values()),budget)
                if method['method'] in ['source_id','lexical','random']:
                    self.assertEqual(result['cost']['calls'],3*budget)

    def test_misleading_source_can_support_wrong_answer(self):
        item, backend = copy.deepcopy(make_pilot()[0]), FixtureBackend()
        source = next(s for s in item['sources'] if s['id'] == 's2')
        source['text'] = source['text'].replace(item['gold'], 'Wrongville')
        row = run_item(backend, item, 128, samples=12)
        self.assertFalse(row['correct'])
        self.assertEqual(row['original']['fixed_answer'], 'Wrongville')
        self.assertGreater(row['effects']['s2'], 0)

    def test_negligible_not_counted_as_discovery(self):
        self.assertIsNone(discovery({'s1': 0, 's2': 0}, ['s1'], 1e-6)['hit'])
        self.assertAlmostEqual(logit_from_logprob(-1000), -1000)

    def test_resume_preserves_completed_conditions_and_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            cmd = [sys.executable, '-m', 'context_attribution.experiment', '--backend', 'fixture',
                   '--limit', '1', '--caps', '128', '256', '--samples', '8', '--output', directory]
            subprocess.run(cmd, check=True, capture_output=True)
            path = Path(directory) / 'traces.jsonl'
            original = path.read_text().splitlines()
            path.write_text(original[0]+'\n')  # Emulate interruption after first completed condition.
            subprocess.run(cmd+['--resume'], check=True, capture_output=True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], json.loads(original[0]))
            self.assertEqual({r['cap'] for r in rows}, {128,256})
            cost = json.loads((Path(directory)/'cost.json').read_text())
            self.assertEqual(cost['calls'], sum(r['execution_cost']['calls'] for r in rows))
            (Path(directory)/'summary.json').unlink()
            subprocess.run(cmd+['--resume'], check=True, capture_output=True)
            self.assertTrue((Path(directory)/'summary.json').exists())
            before = path.read_bytes()
            invalid = subprocess.run(cmd+['--resume','--alpha','0.2'], capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
