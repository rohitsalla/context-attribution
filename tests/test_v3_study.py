import json
from pathlib import Path
import tempfile
import unittest
import torch
from context_attribution.v3.soft_study import objective,interventions,read_items


class StudyDesignTests(unittest.TestCase):
    def test_paired_and_unpaired_controls_identify_different_information(self):
        teacher=torch.tensor([[-1.,-3.]])
        student=torch.tensor([[4.,2.]],requires_grad=True)
        self.assertEqual(float(objective(student,teacher,'paired').detach()),0)
        self.assertGreater(float(objective(student,teacher,'unpaired').detach()),0)
        objective(student,teacher,'factual').backward()
        self.assertTrue(torch.equal(student.grad,torch.zeros_like(student)))

    def test_joint_selection_is_stable_and_label_blind(self):
        item={'id':'example','sources':[{} for _ in range(10)],'gold':'A'}
        selected=interventions(item)
        self.assertEqual(len(selected),13)
        item['gold']='B';item['supporting_facts']=['different']
        self.assertEqual(interventions(item),selected)
        self.assertEqual(len(set(selected)),13)
        self.assertEqual(sum(len(s)==2 for s in selected),3)

    def test_locked_examples_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl';path.write_text(json.dumps({'split':'locked_test'})+'\n')
            with self.assertRaises(ValueError):read_items(path,'dev',1)
