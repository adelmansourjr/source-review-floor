"""Replay a matched-budget, source-coverage-constrained review allocation."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sys

import numpy as np

UPSTREAM_SHA = {
    "dependence.py": "0dcc4d5a76031f97557cd5f3a05cb0b22df51124acac5a84bd3b7d428ebf3ccb",
    "source_transfer.py": "548e6a86af0257cc1bbec5bb4aea7010f2c715d972ad06409f4b2b59d0aaa1d9",
    "pabulib_data.py": "7772f7765203be789db610611198511c98099f1cfe5dd19265ac986a22a031b3",
}
source_dir = Path(os.environ.get("REVIEW_UPSTREAM", Path(__file__).resolve().parents[1])).resolve()
for filename, expected in UPSTREAM_SHA.items():
    if sha256((source_dir / filename).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unexpected upstream SHA-256 for {filename}")
sys.path.insert(0, str(source_dir))
from dependence import EVIDENCE_ACCURACY, MAJORITY, REVIEW_ACCURACY, STATE
from source_transfer import CALIBRATION, TRIPLETS, WEIGHTS, fitted_prior, load_sources, tables

PROTOCOL_SHA = "ccbe1a870ce1c41c87b2781cd37baeb4601d12f5aa5385da0cd7fc8b7f0995d2"
TAPES = 8
REVIEW_FLOOR_DENOMINATOR = 5
CONDITIONS = {
    "nominal": REVIEW_ACCURACY,
    "weaker_review": .72,
    "shared_report_error": REVIEW_ACCURACY,
}
METHODS = ("global", "coverage_floor", "proportional")


def proportional_quotas(sizes, budget):
    weighted = sizes * budget
    quotas, remainders = np.divmod(weighted, sizes.sum())
    seats = budget - quotas.sum()
    quotas[np.argsort(-remainders, kind="stable")[:seats]] += 1
    return quotas


def allocate(scores, keys, groups, sizes, budget, floors):
    if floors.sum() > budget or np.any(floors > sizes):
        raise ValueError("Review floor is infeasible at this budget")
    order = np.lexsort((-keys, -scores))
    within_source = [order[groups[order] == index] for index in range(len(sizes))]
    proportional = proportional_quotas(sizes, budget)
    reserved = np.concatenate([ranks[:floor] for ranks, floor in zip(within_source, floors, strict=True)])
    remaining = np.ones(len(groups), dtype=bool)
    remaining[reserved] = False
    policies = {
        "global": order[:budget],
        "coverage_floor": np.concatenate([reserved, order[remaining[order]][:budget - len(reserved)]]),
        "proportional": np.concatenate([
            ranks[:quota] for ranks, quota in zip(within_source, proportional, strict=True)
        ]),
    }
    for selected in policies.values():
        assert len(selected) == budget and len(np.unique(selected)) == budget
    assert np.all(np.bincount(groups[policies["coverage_floor"]], minlength=len(sizes)) >= floors)
    assert scores[policies["global"]].sum() >= scores[policies["coverage_floor"]].sum() - 1e-10
    assert scores[policies["coverage_floor"]].sum() >= scores[policies["proportional"]].sum() - 1e-10
    return policies


def audit_allocator():
    rng = np.random.default_rng(51)
    sizes = np.array([4, 3, 3])
    groups = np.repeat(np.arange(len(sizes)), sizes)
    floors = np.array([1, 1, 1])
    budget = 5
    feasible = [
        selected for selected in combinations(range(len(groups)), budget)
        if np.all(np.bincount(groups[list(selected)], minlength=len(sizes)) >= floors)
    ]
    for _ in range(20):
        scores = rng.integers(-2, 7, size=len(groups))
        actual = allocate(scores, rng.random(len(groups)), groups, sizes, budget, floors)
        expected_gain = max(scores[list(selected)].sum() for selected in feasible)
        assert scores[actual["coverage_floor"]].sum() == expected_gain


def score_source(source, tape, condition):
    index, assessment, fitted = source
    selectors, probabilities, decisions = fitted
    rng = np.random.default_rng(85715411 + 11 * index + tape)
    bits = STATE[assessment]
    noise = rng.random(bits.shape) >= EVIDENCE_ACCURACY
    keys = rng.random(assessment.shape)
    shared = rng.random(assessment.shape) < .12
    if condition == "shared_report_error":
        noise ^= shared[..., None]
    report = (bits ^ noise) @ WEIGHTS
    triplet = np.arange(len(TRIPLETS))[:, None]
    targets = MAJORITY[assessment]
    baseline = (probabilities[triplet, report] >= .5) != targets
    scores = selectors["shrunk_joint"][triplet, report]
    query = scores.argmax(axis=-1)
    facts = np.take_along_axis(bits, query[..., None], axis=-1)
    decoders = decisions[triplet, report]
    chosen_decoder = np.take_along_axis(
        decoders, query[..., None, None], axis=-2
    )[..., 0, :]
    correct = np.take_along_axis(chosen_decoder, facts, axis=-1)[..., 0] != targets
    wrong = np.take_along_axis(chosen_decoder, 1 - facts, axis=-1)[..., 0] != targets
    loss_if_reviewed = correct * CONDITIONS[condition] + wrong * (1 - CONDITIONS[condition])
    return scores.max(axis=-1).ravel(), keys.ravel(), baseline.ravel(), loss_if_reviewed.ravel()


def assess(selected, groups, sizes, scores, baseline, review_loss):
    total_loss = baseline.astype(float)
    total_loss[selected] = review_loss[selected]
    coverage = np.bincount(groups[selected], minlength=len(sizes)) / sizes
    source_errors = np.bincount(groups, weights=total_loss, minlength=len(sizes)) / sizes
    return {
        "case_weighted_error": float(total_loss.mean()),
        "equal_source_error": float(source_errors.mean()),
        "minimum_source_coverage": float(coverage.min()),
        "total_predicted_gain_per_case": float(scores[selected].sum() / len(groups)),
        "source_coverage": coverage.tolist(),
        "source_error": source_errors.tolist(),
    }


def per_tape(prepared, sizes, groups, budget, floors, tape, condition):
    arrays = [score_source(source, tape, condition) for source in prepared]
    scores, keys, baseline, loss = map(np.concatenate, zip(*arrays, strict=True))
    return {
        name: assess(selected, groups, sizes, scores, baseline, loss)
        for name, selected in allocate(scores, keys, groups, sizes, budget, floors).items()
    }


def average(runs):
    return {
        key: np.asarray([run[key] for run in runs]).mean(axis=0).tolist()
        for key in runs[0]
    }


def load_prepared(data):
    sources = load_sources(data)
    assert len(sources) == 5
    donors = [fitted_prior(encoded[:, CALIBRATION:]) for _, _, encoded in sources]
    prepared = []
    for j, (index, _, encoded) in enumerate(sources):
        local = fitted_prior(encoded[:, :CALIBRATION])
        external = np.mean([prior for i, prior in enumerate(donors) if i != j], axis=0)
        prepared.append((index, encoded[:, CALIBRATION:], tables((local, external))))
    sizes = np.array([assessment.size for _, assessment, _ in prepared])
    groups = np.repeat(np.arange(len(sizes)), sizes)
    return sources, prepared, sizes, groups


def study(data):
    protocol_hash = sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    if protocol_hash != PROTOCOL_SHA:
        raise ValueError("The frozen protocol has changed")
    audit_allocator()
    sources, prepared, sizes, groups = load_prepared(data)
    budget = len(groups) // 4
    floors = (sizes + REVIEW_FLOOR_DENOMINATOR - 1) // REVIEW_FLOOR_DENOMINATOR
    conditions = {}
    for condition in CONDITIONS:
        runs = [per_tape(prepared, sizes, groups, budget, floors, tape, condition)
                for tape in range(TAPES)]
        summary = {method: average([run[method] for run in runs]) for method in METHODS}
        source_increases = [run["coverage_floor"]["minimum_source_coverage"] -
                            run["global"]["minimum_source_coverage"] for run in runs]
        error_increases = [run["coverage_floor"]["case_weighted_error"] -
                           run["global"]["case_weighted_error"] for run in runs]
        paired = {
            "per_tape_minimum_source_coverage_change": source_increases,
            "per_tape_case_weighted_error_change": error_increases,
            "mean_minimum_source_coverage_change": float(np.mean(source_increases)),
            "mean_case_weighted_error_change": float(np.mean(error_increases)),
        }
        conditions[condition] = {"mean_metrics": summary, "floor_minus_global": paired}
    nominal = conditions["nominal"]["floor_minus_global"]
    return {
        "protocol_sha256": protocol_hash,
        "tapes": TAPES,
        "reviews_per_tape": budget,
        "total_cases_per_tape": len(groups),
        "source_names": [name for _, name, _ in sources],
        "assessment_ballots_per_source": (sizes // len(TRIPLETS)).tolist(),
        "assessment_cases_per_source": sizes.tolist(),
        "review_floors": floors.tolist(),
        "primary_gate_met": nominal["mean_minimum_source_coverage_change"] >= .05
        and nominal["mean_case_weighted_error_change"] <= .005,
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
        raise SystemExit("Replay differs from the recorded result")
    print({key: result[key] for key in ("reviews_per_tape", "primary_gate_met")})
    for condition, entry in result["conditions"].items():
        change = entry["floor_minus_global"]
        print(condition, change["mean_minimum_source_coverage_change"], change["mean_case_weighted_error_change"])
