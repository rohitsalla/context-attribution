"""Released RECOMP sentence scores with the frontier's token-budget adapter."""

from math import isfinite

from .attribution import sentences
from .selection_mechanism import pack_fixed

MODEL_ID = "fangyuan/nq_extractive_compressor"
REVISION = "46f98c1f57fa919c72d86ced16e3ece3403f9e9e"


class RecompSelector:
    def __init__(self, path, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.encoder.float().to(device).eval().requires_grad_(False)
        self.device = device

    def rank(self, tokenizer, sources, question):
        # Use the same candidate sentences and reader-side framing as BM25.
        candidates = [
            (source.sid, index, source.title, text)
            for source in sources
            for index, text in enumerate(sentences(source.text) or [source.text])
        ]
        texts = [question] + [f"{title}: {text}" for _, _, title, text in candidates]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            hidden = self.encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].bool().unsqueeze(-1)
            pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)
            # Released RECOMP QA scoring script uses masked mean and dot product.
            scores = (pooled[1:] @ pooled[0]).cpu().tolist()
        if len(scores) != len(candidates) or not all(map(isfinite, scores)):
            raise ValueError("Invalid RECOMP sentence scores")
        order = sorted(zip(scores, candidates, strict=True),
                       key=lambda pair: (-pair[0], pair[1][0], pair[1][1]))
        return [(sid, index, tuple(tokenizer.encode(f"\n[{sid}] {title}: {text}\n", add_special_tokens=False)))
                for _, (sid, index, title, text) in order]

    @staticmethod
    def pack(ranking, sources, budget, omit=None):
        # Scores depend only on the query and individual sentence. Filtering the
        # original order is equivalent to reranking survivors with cached scores.
        return pack_fixed(ranking, [source.sid for source in sources], budget, omit)
