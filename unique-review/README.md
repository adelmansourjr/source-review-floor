# Reusing human reviews across overlapping decisions

One answer can resolve several decisions. This study asks whether a review policy can exploit that overlap at a fixed annotation budget while guaranteeing a minimum review rate for each data source. It follows Erskine, Yamagata and Santos-Rodríguez's work on selecting informative queries with noisy feedback, and studies the related problem of fact reuse in batch review. [Active Query Selection for Crowd-Based Reinforcement Learning](https://arxiv.org/abs/2508.19132).

## Model and exact result

The replay uses 7,969 historical approval ballots from five pinned Pabulib collections. A majority decision is made for each of the 35 triplets of seven projects. Every policy sees the same simulated noisy report for a given ballot. A reusable review asks about one ballot/project pair and shares the single simulated answer across all triplets containing that project. Triplet-scoped legacy ranking can purchase independent answers about the same fact for different decisions. All methods pay for exactly **13,945** reviews per tape. [Frozen protocol](PROTOCOL.md).

A rational example shows why ranking questions by immediate error reduction can miss valuable batches. Three independent approvals have probability 1/5 each and answers are 90% accurate. Either of two questions taken alone leaves Bayes error at 13/125. Asking both saves **109/12500**, or **0.872 percentage points** of expected error. The second question is more valuable after buying the first, so diminishing returns need not hold.

The allocator enumerates all 128 available question subsets per ballot and uses a concave upper bound to coordinate spending across ballots. The executable certificate brackets the best attainable predicted gain between an achieved value and a relaxation bound. Across all tapes, conditions and allocation constraints, its largest possible suboptimality was **0.000105 percentage points of modeled error**. A separate exhaustive audit checks the certificate on 40 smaller allocations. [Proof and exact witness](MATHEMATICS.md) · [Replay and checker](study.py).

## Empirical replay

Values average the six report-noise tapes per condition. Errors integrate over the simulated reviewer answers. Minimum coverage is the smallest source-level share of distinct ballot/project facts reviewed.

| Allocation | Nominal minimum source coverage | Nominal pooled error | Nominal equal-source error | Pooled error with 72% reviewer accuracy | Pooled error with shared report flips |
|---|---:|---:|---:|---:|---:|
| Triplet-scoped ranking | 10.60% | 9.710% | 20.100% | 10.360% | 14.684% |
| Reuse, one-shot gain ranking | 21.56% | **4.693%** | 12.943% | **7.701%** | 7.550% |
| Reuse, marginal greedy | 21.99% | 5.204% | 12.845% | 8.970% | 7.506% |
| Reuse, marginal greedy with floor | 24.00% | 5.313% | 14.077% | 8.925% | 7.731% |
| Reuse, certified global allocation | 21.79% | 5.230% | **12.802%** | 9.078% | **6.650%** |
| Reuse, certified 24% floor | 24.00% | 5.388% | 14.053% | 9.045% | 6.766% |
| Reuse, certified proportional quotas | 24.98% | 5.469% | 15.077% | 8.989% | 6.917% |

Both prespecified nominal gates passed:

- Global certified reuse reduced pooled error by **4.480 points** at the same budget. The 13,945 legacy purchases covered an average of 6,693.5 distinct facts. All reuse methods substantially reduced error against this comparator. One-shot reuse had the best nominal pooled error and was also strongest with noisier reviewers. Certified global allocation performed best in the shared-report-flip stress test, with a **0.855-point** improvement over updated marginal greedy.
- The certified 24% floor improved minimum coverage from **21.794% to 24.001%** for **0.158 points** of additional pooled error. The increase in equal-source average error was **1.251 points**. The floor shifted reviews toward the largest collection, whose 6,554 ballots accounted for over four fifths of the replay. Proportional quotas gave a higher minimum and increased nominal pooled and equal-source errors further.

The six tapes reuse the same historical ballots and overlapping triplets, so they measure variability over simulated corruption. The first 32 shuffled ballots per source fit the local prior, and the other four collections form the external prior. The experiment keeps the fitted posterior fixed in both stress conditions. Review cost counts binary questions. A source labels a collection. The fitted decision rules, simulated answers and historical approvals limit the inference to this replay. [Aggregate, source-level and tape-level results](results.json).

## Reproduce

Use Python 3.12 and NumPy 2.4.6. From the repository root:

```sh
python -m pip install numpy==2.4.6
git clone https://github.com/adelmansourjr/human-centered-review.git review-upstream
git -C review-upstream checkout --detach ce83d4a277efd8850e6d028c565fbfc594c504eb
python review-upstream/pabulib_data.py public-ballots
export REVIEW_UPSTREAM="$PWD/review-upstream"
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B unique-review/study.py \
  --data-dir public-ballots --verify unique-review/results.json
```

The loader verifies file checksums against the pinned Pabulib revision. The study reads only public inputs and stores aggregate results. [Pabulib archive](https://github.com/pabulib/pabulib_files/tree/beda04462f2fbee240202510b860106ee9754dd4).
