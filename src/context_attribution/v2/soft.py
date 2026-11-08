"""Source-separable soft-memory baseline and paired-effect training objective.

Pooling and projections are established components, not claimed inventions.
The research question is whether paired intervention training improves effect
preservation over the same architecture and compute trained factually.
"""
import math
import torch
from torch import nn


class SourceSoftMemory(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,slots=8,embedding_rms=.02):
        super().__init__()
        self.queries=nn.Parameter(torch.randn(slots,encoder_dim)*.02)
        self.input_norm=nn.LayerNorm(encoder_dim)
        self.project=nn.Linear(encoder_dim,decoder_dim,bias=False)
        self.output_norm=nn.LayerNorm(decoder_dim,elementwise_affine=False)
        self.log_scale=nn.Parameter(torch.tensor(math.log(embedding_rms)))

    def forward(self,features,mask):
        x=self.input_norm(features.float())
        logits=torch.einsum('sd,bld->bsl',self.queries,x)/math.sqrt(x.shape[-1])
        logits=logits.masked_fill(~mask[:,None,:],float('-inf'))
        weights=logits.softmax(-1)
        pooled=torch.einsum('bsl,bld->bsd',weights,x)
        return self.output_norm(self.project(pooled))*self.log_scale.exp()


def paired_effect_loss(teacher_logps,student_logps):
    """Columns are factual, then interventions on the same fixed answer span."""
    if teacher_logps.shape!=student_logps.shape or teacher_logps.ndim!=2 or teacher_logps.shape[1]<2:
        raise ValueError('Expected matching [batch, factual+interventions] tensors')
    teacher=teacher_logps[:,:1]-teacher_logps[:,1:]
    student=student_logps[:,:1]-student_logps[:,1:]
    return (teacher.detach()-student).square().mean()
