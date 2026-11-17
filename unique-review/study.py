"""Replay fixed-cost, reusable review on overlapping project triplets."""
import argparse
from fractions import Fraction
from hashlib import sha256
from heapq import heapify, heappop, heappush
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import (
    CALIBRATION, EVIDENCE_ACCURACY, MAJORITY, REVIEW_ACCURACY, STATE,
    TRIPLETS, WEIGHTS, fitted_prior, load_sources, proportional_quotas,
)
from source_transfer import EXTERNAL_WEIGHT
from dependence import CORRUPTION


PROTOCOL_SHA = "f7f02c97c0ce5a18ba1afe59b1dcefab4bd0c399b5360f69afd2875d3606e346"
TAPES = 6
CONDITIONS = {"nominal": (False, .90), "weaker_review": (False, .72),
              "shared_report_error": (True, .90)}
PROJECTS = 7
BITS = 1 << np.arange(PROJECTS)
MASKS = np.arange(1 << PROJECTS)
MASK_SIZE = np.array([int(mask).bit_count() for mask in MASKS])
MASKS_BY_COST = [MASKS[MASK_SIZE == k] for k in range(PROJECTS + 1)]
TRIPLET_IDS = np.array(TRIPLETS)
TRIPLET_MASKS = ((((MASKS[:, None] & BITS) != 0)[:, TRIPLET_IDS]) @ WEIGHTS).T
LOCAL_BITS = np.array([1, 2, 4])
EPS = 1e-12


def answer_model(mask, accuracy):
    answers = np.flatnonzero((np.arange(8) & mask) == np.arange(8))
    chosen = (mask & LOCAL_BITS) != 0
    selected_bits = STATE[:, chosen]
    same = selected_bits[:, None, :] == selected_bits[None, answers, :]
    channel = np.prod(np.where(same, accuracy, 1 - accuracy), axis=-1)
    return channel


def decision_tables(prior, actual_accuracy):
    # Each entry is indexed by (triplet, report, local review mask).
    posterior = prior[:, None, :] * CORRUPTION.T[None, :, :]
    posterior /= posterior.sum(axis=-1, keepdims=True)
    modeled_risk = np.empty((*posterior.shape[:2], 8))
    realized_loss = np.empty((*posterior.shape, 8))
    for mask in range(8):
        channel = answer_model(mask, REVIEW_ACCURACY)
        positive = (posterior * MAJORITY) @ channel
        negative = (posterior * (1 - MAJORITY)) @ channel
        decision = positive >= negative
        modeled_risk[..., mask] = np.minimum(positive, negative).sum(axis=-1)
        wrong = decision[..., None, :] != MAJORITY[:, None]
        realized_loss[..., mask] = np.sum(wrong * answer_model(mask, actual_accuracy), axis=-1)
    gains = modeled_risk[..., 0, None] - modeled_risk
    assert np.min(gains) >= -EPS
    return np.round(gains, 12), realized_loss


def source_arrays(assessment, gain, loss, index, tape, shared):
    n = assessment.shape[1]
    facts = np.empty((n, PROJECTS), dtype=np.int8)
    for project in range(PROJECTS):
        triplet = next(i for i, row in enumerate(TRIPLETS) if project in row)
        position = TRIPLETS[triplet].index(project)
        facts[:, project] = STATE[assessment[triplet], position]
    assert np.array_equal((facts[:, TRIPLET_IDS] @ WEIGHTS).T, assessment)

    rng = np.random.default_rng(37925191 + 17 * index + tape)
    flips = rng.random(facts.shape) >= EVIDENCE_ACCURACY
    joint_flip = rng.random(n) < .12
    keys = rng.random(n)
    if shared:
        flips ^= joint_flip[:, None]
    report = (facts ^ flips)[:, TRIPLET_IDS] @ WEIGHTS

    total_gain = np.zeros((n, len(MASKS)))
    total_loss = np.zeros_like(total_gain)
    legacy_gain = np.empty((n, len(TRIPLETS)))
    legacy_change = np.empty_like(legacy_gain)
    legacy_project = np.empty(legacy_gain.shape, dtype=np.int8)
    for i, projects in enumerate(TRIPLET_IDS):
        reports = report[:, i]
        observed_gain = gain[i, reports]
        observed_loss = loss[i, reports, assessment[i]]
        local_mask = TRIPLET_MASKS[i]
        total_gain += observed_gain[:, local_mask]
        total_loss += observed_loss[:, local_mask]
        choices = observed_gain[:, LOCAL_BITS].argmax(axis=1)
        legacy_gain[:, i] = observed_gain[np.arange(n), LOCAL_BITS[choices]]
        legacy_change[:, i] = observed_loss[np.arange(n), LOCAL_BITS[choices]] - observed_loss[:, 0]
        legacy_project[:, i] = projects[choices]
    return np.round(total_gain, 10), total_loss, legacy_gain, legacy_change, legacy_project, keys


def best_at_cost(gains):
    best = np.column_stack([masks[gains[:, masks].argmax(axis=1)] for masks in MASKS_BY_COST])
    return best, np.take_along_axis(gains, best, axis=1)


def majorant(values):
    hull = []
    for i, value in enumerate(values):
        while len(hull) >= 2:
            start, end = hull[-2:]
            left = (values[end] - values[start]) / (end - start)
            right = (value - values[end]) / (i - end)
            if left > right + EPS:
                break
            hull.pop()
        hull.append(i)
    increments = []
    for a, b in zip(hull, hull[1:]):
        increments.extend([(values[b] - values[a]) / (b - a)] * (b - a))
    return np.array(increments)


def reserve_and_fill(scores, keys, groups, floors, budget):
    rows, increments = np.indices(scores.shape)
    rows, increments = rows.ravel(), increments.ravel()
    order = np.lexsort((increments, -keys[rows], -scores.ravel()))
    size = scores.size
    flat_groups = groups[rows]
    reserved = np.concatenate([order[flat_groups[order] == g][:floor] for g, floor in enumerate(floors)])
    available = np.ones(size, dtype=bool)
    available[reserved] = False
    selected = np.concatenate((reserved, order[available[order]][:budget - len(reserved)]))
    assert len(selected) == budget
    assert np.all(np.bincount(flat_groups[selected], minlength=len(floors)) >= floors)
    return rows[selected], increments[selected]


def certified(best, values, hull, keys, groups, floors, budget):
    rows, steps = reserve_and_fill(hull, keys, groups, floors, budget)
    cost = np.bincount(rows, minlength=len(best))
    expected_prefix = np.arange(hull.shape[1])[None, :] < cost[:, None]
    selected_prefix = np.zeros_like(hull, dtype=bool)
    selected_prefix[rows, steps] = True
    assert np.array_equal(expected_prefix, selected_prefix)
    feasible = best[np.arange(len(best)), cost]
    upper = np.sum(hull[rows, steps]) + values[:, 0].sum()
    achieved = values[np.arange(len(best)), cost].sum()
    assert upper >= achieved - 1e-8
    return feasible, max(0., upper - achieved)


def exact_audit():
    rng = np.random.default_rng(13553)
    checked = 0
    for _ in range(40):
        n, max_cost, budget = 4, 3, 7
        key = rng.random(n)
        groups = np.array([0, 0, 1, 1])
        floors = np.array([2, 1])
        values = np.column_stack((np.zeros(n), np.cumsum(
            rng.integers(0, 8, (n, max_cost)), axis=1
        )))
        best = np.repeat(np.arange(max_cost + 1)[None, :], n, axis=0)
        hull = np.array([majorant(row) for row in values])
        chosen, gap = certified(best, values, hull, key, groups, floors, budget)
        candidates = [cost for cost in product(range(max_cost + 1), repeat=n)
                      if sum(cost) == budget and sum(cost[:2]) >= floors[0]
                      and sum(cost[2:]) >= floors[1]]
        truth = max(values[np.arange(n), cost].sum() for cost in candidates)
        achieved = values[np.arange(n), chosen].sum()
        assert achieved <= truth <= achieved + gap + 1e-9
        checked += 1
    return checked


def rational_witness():
    # Three independent approvals, each with prior probability 1/5.
    risks = []
    for mask in ((), (0,), (1,), (0, 1)):
        error = Fraction()
        for answer in product((0, 1), repeat=len(mask)):
            mass = [Fraction(), Fraction()]
            for state in product((0, 1), repeat=3):
                chance = Fraction(1, 5)**sum(state) * Fraction(4, 5)**(3 - sum(state))
                for k, observed in zip(mask, answer, strict=True):
                    chance *= Fraction(9 if state[k] == observed else 1, 10)
                mass[int(sum(state) >= 2)] += chance
            error += min(mass)
        risks.append(error)
    assert risks[0] == risks[1] == risks[2] > risks[3]
    return {"risks": [str(risk) for risk in risks], "two_review_gain": str(risks[0] - risks[3])}


def greedy(gains, keys, groups, floors, budget):
    selected = np.zeros(len(gains), dtype=np.int64)

    def next_item(i):
        old = selected[i]
        choices = gains[i, BITS | old] - gains[i, old]
        choices[(BITS & old) != 0] = -np.inf
        pick = int(choices.argmax())
        return (-round(float(choices[pick]), 10), -keys[i], int(i), pick)

    def take(heap, count):
        for _ in range(count):
            _, _, ballot, pick = heappop(heap)
            selected[ballot] |= BITS[pick]
            if selected[ballot] < MASKS[-1]:
                heappush(heap, next_item(ballot))

    leftovers = []
    for group, floor in enumerate(floors):
        heap = [next_item(i) for i in np.flatnonzero(groups == group)]
        heapify(heap)
        take(heap, int(floor))
        leftovers.extend(heap)
    heapify(leftovers)
    take(leftovers, int(budget - floors.sum()))
    assert MASK_SIZE[selected].sum() == budget
    return selected


def summarize(cost, loss, groups, sizes, gap=None, purchased=None):
    error = np.bincount(groups, weights=loss, minlength=len(sizes)) / (sizes * len(TRIPLETS))
    coverage = np.bincount(groups, weights=cost, minlength=len(sizes)) / (sizes * PROJECTS)
    metrics = {"case_weighted_error": float(loss.sum() / (len(groups) * len(TRIPLETS))),
            "equal_source_error": float(error.mean()), "minimum_source_coverage": float(coverage.min()),
            "source_coverage": coverage.tolist(), "source_error": error.tolist(),
            "unique_questions": int(cost.sum()),
            "purchased_questions": int(cost.sum() if purchased is None else purchased)}
    if gap is not None:
        metrics["predicted_optimality_gap_upper_bound"] = float(gap / (len(groups) * len(TRIPLETS)))
    return metrics


def evaluate(arrays, sizes):
    gains, loss, legacy_gains, changes, projects, keys = arrays
    groups = np.repeat(np.arange(len(sizes)), sizes)
    budget = PROJECTS * len(groups) // 4
    floor = (24 * PROJECTS * sizes + 99) // 100
    free = np.zeros_like(floor)
    proportional = proportional_quotas(PROJECTS * sizes, budget)
    best, values = best_at_cost(gains)
    hull = np.array([majorant(row) for row in values])
    results = {}
    for method, bounds in (("certificate_global", free), ("certificate_floor", floor),
                           ("certificate_proportional", proportional)):
        selected, gap = certified(best, values, hull, keys, groups, bounds, budget)
        results[method] = summarize(MASK_SIZE[selected], loss[np.arange(len(gains)), selected],
                                    groups, sizes, gap)
    for method, bounds in (("marginal_global", free), ("marginal_floor", floor)):
        selected = greedy(gains, keys, groups, bounds, budget)
        results[method] = summarize(MASK_SIZE[selected], loss[np.arange(len(gains)), selected], groups, sizes)

    rows, cols = reserve_and_fill(gains[:, BITS], keys, groups, free, budget)
    selected = np.zeros(len(gains), dtype=np.int64)
    np.bitwise_or.at(selected, rows, BITS[cols])
    results["one_shot_global"] = summarize(MASK_SIZE[selected], loss[np.arange(len(gains)), selected], groups, sizes)
    rows, cols = reserve_and_fill(legacy_gains, keys, groups, free, budget)
    selected = np.zeros(len(gains), dtype=np.int64)
    np.bitwise_or.at(selected, rows, BITS[projects[rows, cols]])
    legacy_loss = loss[:, 0].copy()
    np.add.at(legacy_loss, rows, changes[rows, cols])
    results["legacy"] = summarize(MASK_SIZE[selected], legacy_loss, groups, sizes, purchased=budget)
    assert all(method["purchased_questions"] == budget for method in results.values())
    return results


def study(data):
    digest = sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest()
    if digest != PROTOCOL_SHA:
        raise ValueError("The frozen protocol has changed")
    sources = load_sources(data)
    donors = [fitted_prior(encoded[:, CALIBRATION:]) for _, _, encoded in sources]
    prepared = []
    for j, (index, name, encoded) in enumerate(sources):
        local = fitted_prior(encoded[:, :CALIBRATION])
        external = np.mean([prior for i, prior in enumerate(donors) if i != j], axis=0)
        prior = (CALIBRATION * local + EXTERNAL_WEIGHT * external) / (CALIBRATION + EXTERNAL_WEIGHT)
        prepared.append((index, name, encoded[:, CALIBRATION:], prior))
    sizes = np.array([assessment.shape[1] for _, _, assessment, _ in prepared])
    conditions = {}
    for condition, (shared, actual_accuracy) in CONDITIONS.items():
        tables = [decision_tables(prior, actual_accuracy) for _, _, _, prior in prepared]
        runs = []
        for tape in range(TAPES):
            per_source = [source_arrays(assessment, g, l, index, tape, shared)
                          for (index, _, assessment, _), (g, l) in zip(prepared, tables, strict=True)]
            arrays = tuple(map(np.concatenate, zip(*per_source, strict=True)))
            runs.append(evaluate(arrays, sizes))
        mean = {
            method: {key: np.mean([run[method][key] for run in runs], axis=0).tolist()
                     for key in runs[0][method]}
            for method in runs[0]
        }
        conditions[condition] = {"mean": mean, "tapes": runs}
    nominal = conditions["nominal"]["mean"]
    global_review = nominal["certificate_global"]
    floor_review = nominal["certificate_floor"]
    reduction = nominal["legacy"]["case_weighted_error"] - global_review["case_weighted_error"]
    coverage_increase = floor_review["minimum_source_coverage"] - global_review["minimum_source_coverage"]
    error_increase = floor_review["case_weighted_error"] - global_review["case_weighted_error"]
    return {
        "protocol_sha256": digest, "enumeration_audits": exact_audit(),
        "rational_witness": rational_witness(), "sources": [name for _, name, _, _ in prepared],
        "source_ballots": sizes.tolist(), "conditions": conditions,
        "primary": {"reuse_error_reduction": reduction, "reuse_gate_pass": reduction >= .01,
                    "floor_coverage_increase": coverage_increase, "floor_error_increase": error_increase,
                    "floor_gate_pass": coverage_increase >= .01 and error_increase <= .0025}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = parser.parse_args()
    value = study(args.data_dir)
    formatted = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(formatted)
    else:
        assert formatted == args.verify.read_text(), "Replay differs"
    print(value["primary"])
