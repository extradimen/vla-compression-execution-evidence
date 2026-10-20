#!/usr/bin/env python3
"""Recompute primary final-grid estimates from the released episode outcomes."""
import csv
import json
import random
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "data/episodes.csv").open()))
assert len(rows) == 1920
ARMS = ("U_E1", "P4_E1", "U_E4", "P4_E4")
TASKS = tuple(range(2, 10))
for row in rows:
    row["task_id"] = int(row["task_id"])
    row["state_id"] = int(row["state_id"])
    row["seed"] = int(row["seed"])
    row["success"] = int(row["success"])
    row["episode_wall_seconds"] = float(row["episode_wall_seconds"])
cells = {(r["arm"], r["task_id"], r["state_id"], r["seed"]): r for r in rows}
assert len(cells) == 1920
for arm in ARMS:
    subset = [r for r in rows if r["arm"] == arm]
    assert len(subset) == 480
    print(arm, "successes", sum(r["success"] for r in subset),
          "mean_all_attempt_wall_s", stats.mean(r["episode_wall_seconds"] for r in subset))

weights = {
    "compression_at_E1": {"P4_E1": 1, "U_E1": -1},
    "compression_at_E4": {"P4_E4": 1, "U_E4": -1},
    "execution_E4_vs_E1_uncompressed": {"U_E4": 1, "U_E1": -1},
    "interaction_difference_in_differences":
        {"P4_E4": 1, "P4_E1": -1, "U_E4": -1, "U_E1": 1},
}
def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(fraction * len(ordered))))]

for name, terms in weights.items():
    for metric in ("success", "episode_wall_seconds"):
        per_task = [
            sum(weight * stats.mean(
                cells[(arm, task, state, seed)][metric]
                for state in range(20, 40) for seed in (53, 67, 79))
                for arm, weight in terms.items())
            for task in TASKS
        ]
        rng = random.Random(20261017)
        draws = [stats.mean(per_task[rng.randrange(8)] for _ in TASKS)
                 for _ in range(10000)]
        print(name, metric, json.dumps({
            "estimate": stats.mean(per_task),
            "task_cluster_percentile_95ci":
                [percentile(draws, .025), percentile(draws, .975)]}))
