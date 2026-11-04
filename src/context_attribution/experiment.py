import argparse
import hashlib
import itertools
import json
import math
import platform
import os
import random
import time
from pathlib import Path
import numpy as np
from .backend import FixtureBackend, HFBackend, MLXBackend, summary_prompt, answer_prompt
from .data import make_pilot, validate
from .attribution import recursive, lexical, tracing, select


def branch(backend, item, blocks, cap, fixed_answer=None, seed=0, temperature=0):
    start = len(backend.records)
    summary = backend.call("summary", summary_prompt(item["question"], blocks),
                           cap=cap, seed=seed, temperature=temperature)
    summary_info = dict(backend.records[-1])
    generated = backend.call("answer", answer_prompt(item["question"], summary),
                             cap=24, seed=seed, temperature=temperature)
    target = generated if fixed_answer is None else fixed_answer
    score = backend.call("answer_score", answer_prompt(item["question"], summary), target=target)
    return {"summary": summary, "generated_answer": generated, "fixed_answer": target,
            "logprob": score, "summary_info": summary_info, "cost": backend.cost(start)}


def answer_correct(item, answer):
    return answer.strip().casefold() in {str(x).casefold() for x in [item['gold']]+item.get('gold_aliases',[])}


def discovery(effects, selected, noise):
    maximum = max(abs(v) for v in effects.values())
    negligible = maximum <= noise
    return {"negligible": negligible,
            "hit": None if negligible else any(maximum - abs(effects[s]) <= noise for s in selected),
            "discovered_effect": max(abs(effects[s]) for s in selected)}


def rank_correlation(scores, effects):
    # Average ranks for ties; constant vectors have undefined correlation.
    def ranks(values):
        return np.array([sum(y < x for y in values) + (sum(y == x for y in values)-1)/2
                         for x in values])
    a = ranks([abs(scores[k]) for k in sorted(scores)])
    b = ranks([abs(effects[k]) for k in sorted(scores)])
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None


def run_item(backend, item, cap, samples=32, alpha=.01, seed=0, repeats=2, temperature=0):
    began = time.perf_counter()
    call_start = len(backend.records)
    original = branch(backend, item, item["sources"], cap, seed=seed, temperature=temperature)
    answer = original["fixed_answer"]
    raw = backend.call("raw_context_answer", answer_prompt(item["question"],
                       "\n".join(s["text"] for s in item["sources"])), cap=24, seed=seed,
                       temperature=temperature)
    unchanged = [branch(backend, item, item["sources"], cap, answer, seed+r, temperature)
                 for r in range(repeats)]
    noise = max([1e-6] + [abs(b["logprob"] - original["logprob"]) for b in unchanged])
    branches, effects = {}, {}
    # Seed pairing is preserved across factual/intervention branches for each repeat.
    for source in item["sources"]:
        kept = [s for s in item["sources"] if s["id"] != source["id"]]
        trials = [branch(backend, item, kept, cap, answer, seed+r, temperature)
                  for r in range(repeats)]
        paired = [f["logprob"] - b["logprob"] for f, b in zip(unchanged, trials)]
        effects[source["id"]] = float(np.mean(paired))
        branches[source["id"]] = {"trials": trials, "paired_effects": paired}
    groups=[]
    for group in item.get('diagnostic_groups',[]):
        trials=[branch(backend,item,[s for s in item['sources'] if s['id'] not in group],
                       cap,answer,seed+r,temperature) for r in range(repeats)]
        groups.append({'removed':group,'trials':trials,
                       'paired_effects':[f['logprob']-b['logprob'] for f,b in zip(unchanged,trials)]})
    candidates, overhead = {}, {}
    for name, scorer in [("recursive_surrogate", lambda: recursive(backend, item, original["summary"], answer,
                                                                samples, alpha, seed)),
                          ("lexical", lambda: lexical(item, answer)),
                          ("source_id", lambda: tracing(item, original["summary"]))]:
        before, tick = len(backend.records), time.perf_counter()
        candidates[name] = scorer()
        overhead[name] = {**backend.cost(before), "wall_seconds": time.perf_counter()-tick}
    comparisons = []
    for budget in [1, 2, 3]:
        # Exact random-subset expectation removes selection Monte Carlo noise (6 sources).
        combos = list(itertools.combinations(effects, budget))
        random_metrics = [discovery(effects, c, noise) for c in combos]
        comparisons.append({"method": "random_exact_expectation", "budget": budget,
                            "hit": None if random_metrics[0]["negligible"] else
                            float(np.mean([m["hit"] for m in random_metrics])),
                            "discovered_effect": float(np.mean([m["discovered_effect"] for m in random_metrics])),
                            "negligible": random_metrics[0]["negligible"]})
        for name, scores in {**candidates, "oracle": effects}.items():
            chosen = select(scores, budget, seed)
            # Evaluation looks up offline complete replay; untested sources stay null.
            verified = {s: effects[s] if s in chosen else None for s in effects}
            replay_cost = {k: sum(branches[s]["trials"][0]["cost"][k] for s in chosen)
                           for k in ["calls", "seconds", "input_tokens", "output_tokens"]}
            comparisons.append({"method": name, "budget": budget, "selected": chosen,
                                "verified_effects": verified, "untested": [s for s in effects if s not in chosen],
                                "selection_overhead": overhead.get(name), "one_replay_per_source_cost": replay_cost,
                                "support_recall": len(set(chosen) & set(item["support_ids"]))/len(item["support_ids"]),
                                **discovery(effects, chosen, noise)})
    return {"item_id": item["id"], "base_id": item["base_id"], "split": item["split"], "cap": cap,
            "backend": backend.name, "temperature": temperature, "original": original,
            "family":item.get('family'),"data_kind":item.get('data_kind'),"variant":item.get('variant','base'),
            "raw_context_answer": raw, "raw_context_correct": answer_correct(item,raw),
            "gold": item["gold"],"gold_aliases":item.get('gold_aliases',[]),"correct":answer_correct(item,answer),
            "group_diagnostics":groups,
            "unchanged": unchanged, "replay_noise": noise, "reference": branches, "effects": effects,
            "irrelevant_effect": effects[item["irrelevant_id"]], "candidate_scores": candidates,
            "rank_correlation": {n: rank_correlation(s, effects) for n, s in candidates.items()},
            "comparisons": comparisons, "wall_seconds": time.perf_counter()-began,
            "execution_cost": backend.cost(call_start)}


def summarize(rows, seed=0):
    rng, output = np.random.default_rng(seed), []
    for cap in sorted(set(r["cap"] for r in rows)):
        for split in sorted(set(r["split"] for r in rows)):
            for slice_name in ["all", "original_correct"]:
                subset = [r for r in rows if r["cap"] == cap and r["split"] == split
                          and (slice_name == "all" or r["correct"])]
                for method in ["recursive_surrogate", "lexical", "source_id", "random_exact_expectation", "oracle"]:
                    for budget in [1, 2, 3]:
                        by_base = {}
                        negligible = 0
                        for row in subset:
                            c = next(c for c in row["comparisons"] if c["method"] == method and c["budget"] == budget)
                            negligible += c["negligible"]
                            if c["hit"] is not None:
                                by_base.setdefault(row["base_id"], []).append(float(c["hit"]))
                        values = [np.mean(v) for v in by_base.values()]
                        boot = [float(np.mean(rng.choice(values, len(values)))) for _ in range(1000)] if values else []
                        output.append({"cap": cap, "split": split, "slice": slice_name, "method": method,
                                       "budget": budget, "items": len(subset), "negligible": negligible,
                                       "base_clusters": len(values), "hit_rate": float(np.mean(values)) if values else None,
                                       "bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist() if boot else None})
    return output


def resume_rows(directory, config, source_hashes, dataset_hash):
    manifest = json.loads((directory / "manifest.json").read_text())
    for key in ["backend", "model", "revision", "device", "caps", "start", "limit", "dataset",
                "samples", "alpha", "seed", "repeats", "temperature"]:
        if manifest.get(key, 0 if key == "start" else None) != config.get(key):
            raise ValueError(f"Resume configuration mismatch: {key}")
    if manifest['source_sha256'] != source_hashes or manifest['dataset_sha256'] != dataset_hash:
        raise ValueError("Resume source or dataset hash mismatch")
    trace = directory / 'traces.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    keys = [(r['item_id'], r['cap']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate conditions in resume trace")
    return rows


def write_aggregates(directory, rows, seed):
    (directory / 'summary.json').write_text(json.dumps(summarize(rows, seed), indent=2, allow_nan=False))
    cost = {k: sum(r['execution_cost'][k] for r in rows)
            for k in ['calls', 'seconds', 'input_tokens', 'output_tokens']}
    (directory / 'cost.json').write_text(json.dumps(cost, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["fixture", "hf", "mlx"], default="fixture")
    p.add_argument("--model")
    p.add_argument('--dataset',type=Path,help='Frozen JSONL dataset; defaults to legacy pilot fixtures')
    p.add_argument('--verify-likelihood',action='store_true')
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--caps", nargs="+", type=int, default=[128, 256])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--start", type=int, default=0, help="Start item index for disjoint operational shards")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--alpha", type=float, default=.01)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--output", type=Path, default=Path("results/fixture"))
    p.add_argument("--resume", action="store_true", default=bool(os.environ.get("SLURM_JOB_ID")),
                   help="Resume matching saved conditions; automatically enabled inside Slurm")
    args = p.parse_args()
    if args.repeats < 2 or args.limit < 1 or args.start < 0 or any(c < 1 for c in args.caps):
        p.error("at least two repeats, one item, and positive caps are required")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("output directory must be empty to protect earlier runs")
    args.output.mkdir(parents=True, exist_ok=True)
    items = ([json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
             if args.dataset else make_pilot())
    validate(items)
    dataset = "".join(json.dumps(i, sort_keys=True) + "\n" for i in items)
    dataset_hash = hashlib.sha256(dataset.encode()).hexdigest()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    rows = []
    if (args.output / 'manifest.json').exists():
        rows = resume_rows(args.output, {**vars(args),'dataset':str(args.dataset) if args.dataset else None}, source_hashes, dataset_hash)
    completed = {(r['item_id'], r['cap']) for r in rows}
    wanted = {(i['id'], cap) for i in items[args.start:args.start+args.limit] for cap in args.caps}
    if wanted <= completed:
        write_aggregates(args.output, rows, args.seed)
        print("All requested conditions already completed; no GPU inference needed.", flush=True)
        return
    (args.output / "dataset.jsonl").write_text(dataset)
    if args.backend != "fixture" and not args.model:
        p.error("--model is required for a real-model backend")
    backend = FixtureBackend() if args.backend == "fixture" else (
        MLXBackend(args.model) if args.backend == "mlx" else HFBackend(args.model, args.device, args.revision))
    likelihood_check=None
    if args.verify_likelihood:
        if not isinstance(backend,HFBackend): p.error('--verify-likelihood requires the hf backend')
        first=items[args.start]
        likelihood_check=backend.verify_likelihood(answer_prompt(first['question'],first['sources'][0]['text']),first['gold'])
    manifest = {**vars(args), "output": str(args.output),'dataset':str(args.dataset) if args.dataset else None,
                'likelihood_check':likelihood_check,"backend_name": backend.name,
                "source_sha256": source_hashes,
                "dataset_sha256": dataset_hash,
                'input_dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest() if args.dataset else None,
                "python": platform.python_version(), "platform": platform.platform(),
                "resolved_revision": getattr(backend, "resolved_revision", None),
                "research_evidence": args.backend != "fixture",
                "token_unit": "whitespace_fixture_units" if args.backend == "fixture" else "model_tokens",
                "local_method": "ContextCite-style L1 surrogate approximation; not official ContextCite",
                "normalization": "signed composition with absolute-L1 normalization; rank by absolute value"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.output / "traces.jsonl").open("a" if args.resume else "w") as handle:
        for item in items[args.start:args.start + args.limit]:
            for cap in args.caps:
                if (item['id'], cap) in completed:
                    continue
                row = run_item(backend, item, cap, args.samples, args.alpha, args.seed,
                               args.repeats, args.temperature)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
                print(f"{item['id']} cap={cap} correct={row['correct']} noise={row['replay_noise']:.3g}", flush=True)
    write_aggregates(args.output, rows, args.seed)


if __name__ == "__main__":
    main()
