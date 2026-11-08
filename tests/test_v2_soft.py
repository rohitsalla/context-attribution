import unittest
import torch
from context_attribution.v2.soft import SourceSoftMemory,paired_effect_loss


class SoftMemoryTests(unittest.TestCase):
    def test_source_independence_and_trainable_gradients(self):
        torch.manual_seed(42);pool=SourceSoftMemory(8,16,slots=2)
        x=torch.randn(3,5,8);mask=torch.ones(3,5,dtype=torch.bool)
        original=pool(x,mask)
        self.assertEqual(tuple(original.shape),(3,2,16))
        self.assertTrue(torch.allclose(original[1:],pool(x[1:],mask[1:]),atol=1e-6))
        changed=x.clone();changed[0]*=4
        self.assertTrue(torch.allclose(original[1:],pool(changed,mask)[1:],atol=1e-6))
        original.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters()))

    def test_paired_loss_preserves_effects_not_baseline_levels(self):
        teacher=torch.tensor([[-1.,-3.]])
        shifted=torch.tensor([[-2.,-4.]],requires_grad=True)
        self.assertAlmostEqual(float(paired_effect_loss(teacher,shifted)),0)
        wrong=torch.tensor([[-1.,-1.]],requires_grad=True)
        loss=paired_effect_loss(teacher,wrong);self.assertEqual(float(loss),4)
        loss.backward();self.assertNotEqual(float(wrong.grad.abs().sum()),0)

    def test_rejects_unpaired_targets(self):
        with self.assertRaises(ValueError):paired_effect_loss(torch.zeros(2,2),torch.zeros(2,3))
