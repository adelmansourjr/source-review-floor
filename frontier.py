"""Replay the exploratory source coverage frontier at the same total budget."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from replay import CONDITIONS, METHODS, TAPES, allocate, assess, average, load_prepared, score_source

FLOORS_PER_THOUSAND = (220, 230, 240, 249)


def study(data):
    _, prepared, sizes, groups = load_prepared(data)
    budget = len(groups) // 4
    conditions = {}
    for condition in CONDITIONS:
        runs = {threshold: {method: [] for method in METHODS} for threshold in FLOORS_PER_THOUSAND}
        for tape in range(TAPES):
            sources = [score_source(source, tape, condition) for source in prepared]
            scores, keys, baseline, loss = map(np.concatenate, zip(*sources, strict=True))
            for threshold, policies in runs.items():
                floors = (sizes * threshold + 999) // 1000
                for name, selected in allocate(scores, keys, groups, sizes, budget, floors).items():
                    policies[name].append(assess(selected, groups, sizes, scores, baseline, loss))
        conditions[condition] = {
            str(threshold): {name: average(metrics) for name, metrics in policies.items()}
            for threshold, policies in runs.items()
        }
    return {
        "design_sha256": sha256(Path(__file__).with_name("FRONTIER_DESIGN.md").read_bytes()).hexdigest(),
        "tapes": TAPES,
        "reviews_per_tape": budget,
        "total_cases_per_tape": len(groups),
        "conditions": conditions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = study(args.data_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    elif rendered != args.verify.read_text():
        raise SystemExit("Replay differs from the recorded frontier")
    for condition, frontier in result["conditions"].items():
        for threshold, policies in frontier.items():
            print(condition, threshold, {
                name: (round(values["minimum_source_coverage"] * 100, 4),
                       round(values["case_weighted_error"] * 100, 4))
                for name, values in policies.items()
            })
