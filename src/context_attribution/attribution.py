"""Explicit ContextCite-style approximation, not the official implementation."""
import math
import random
import re
import numpy as np
from .backend import summary_prompt, answer_prompt


def sentences(summary):
    # Keep trailing source annotations attached to their preceding sentence.
    return re.findall(r"[^.!?]+[.!?](?:\s*\[[^\]]+\])?|[^.!?]+$", summary.strip())


def logit_from_logprob(logp):
    # log(p/(1-p)), computed without underflow for very small p.
    logp = min(float(logp), -1e-12)
    return logp - math.log(-math.expm1(logp))


def surrogate(n, score, samples=32, seed=0, alpha=.01):
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(max(samples, n + 2), n)).astype(float)
    masks[0], masks[1] = 0, 1
    y = np.array([logit_from_logprob(score(mask.astype(bool))) for mask in masks])
    x = masks - masks.mean(0)
    y -= y.mean()
    w = np.zeros(n)
    # Coordinate descent for mean squared error / 2 + alpha * L1.
    for _ in range(1000):
        old = w.copy()
        for j in range(n):
            r = y - x @ w + x[:, j] * w[j]
            rho = float(x[:, j] @ r) / len(y)
            denom = float(x[:, j] @ x[:, j]) / len(y)
            w[j] = np.sign(rho) * max(abs(rho) - alpha, 0) / denom if denom else 0
        if np.max(abs(w - old)) < 1e-7:
            break
    return w


def recursive(backend, item, summary, answer, samples=32, alpha=.01, seed=0):
    parts = sentences(summary)
    blocks, q = item["sources"], item["question"]
    if not parts:
        return {b["id"]: 0.0 for b in blocks}
    upstream = surrogate(len(parts), lambda mask: backend.call(
        "local_answer_score", answer_prompt(q, " ".join(p for p, keep in zip(parts, mask) if keep)),
        target=answer), samples, seed, alpha)
    # Fixed absolute-L1 normalization; signed products retain direction.
    upstream /= max(float(np.abs(upstream).sum()), 1e-12)
    result = np.zeros(len(blocks))
    for j, part in enumerate(parts):
        if upstream[j] == 0:
            continue
        downstream = surrogate(len(blocks), lambda mask: backend.call(
            "local_summary_score", summary_prompt(q, [b for b, keep in zip(blocks, mask) if keep]),
            target=part.strip()), samples, seed + j + 1, alpha)
        downstream /= max(float(np.abs(downstream).sum()), 1e-12)
        result += upstream[j] * downstream
    return {b["id"]: float(s) for b, s in zip(blocks, result)}


def lexical(item, answer):
    # Deliberately cheap lexical baseline; do not label this semantic similarity.
    words = set(re.findall(r"\w+", (item["question"] + " " + answer).lower()))
    return {b["id"]: len(words & set(re.findall(r"\w+", b["text"].lower())))
            for b in item["sources"]}


def tracing(item, summary):
    return {b["id"]: len(re.findall(rf"\b{re.escape(b['id'])}\b", summary))
            for b in item["sources"]}


def select(scores, budget, seed=0):
    ids = sorted(scores)
    random.Random(seed).shuffle(ids)  # Reproducible random tie-breaking.
    return sorted(ids, key=lambda x: abs(scores[x]), reverse=True)[:budget]
