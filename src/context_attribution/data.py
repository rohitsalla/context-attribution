"""Controlled fictional fixtures with family-disjoint pilot splits."""
import random


def make_pilot():
    items = []
    # Two examples per family; every family belongs entirely to one split.
    for family in range(10):
        for variant in range(2):
            i = 2 * family + variant
            project, lead, city = f"Aster{i}", f"Mira{i}", f"Delft{i}"
            sources = [
                {"id": "s1", "text": f"Project {project} is led by {lead}."},
                {"id": "s2", "text": f"{lead} works in {city}."},
                {"id": "s3", "text": f"Project Birch{i} is led by Niko{i}."},
                {"id": "s4", "text": f"Niko{i} works in Turin{i}."},
                {"id": "s5", "text": f"Project {project} uses a blue logo."},
                {"id": "s6", "text": f"The cafeteria serves tea on day {i}."},
            ]
            random.Random(1729 + i).shuffle(sources)
            items.append({"id": f"pilot-{i:02}", "base_id": f"base-{i:02}",
                          "family": f"family-{family}",
                          "split": "dev" if family < 4 else "test",
                          "question": f"In which city does the lead of Project {project} work?",
                          "gold": city, "sources": sources,
                          "support_ids": ["s1", "s2"], "irrelevant_id": "s6",
                          "data_kind": "controlled_fictional",
                          "audited_natural_text": False})
    return items


def validate(items):
    families, seen = {}, set()
    for item in items:
        assert item["id"] not in seen, "duplicate item id"
        seen.add(item["id"])
        ids = [s["id"] for s in item["sources"]]
        assert len(ids) == len(set(ids)) == 6, "exactly six distinct source IDs required"
        assert set(item["support_ids"]) <= set(ids)
        assert families.setdefault(item["family"], item["split"]) == item["split"], "family leakage"
