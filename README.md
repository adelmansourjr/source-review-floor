# Coverage floors for budgeted human review

[New study: reuse one review across related decisions](unique-review/README.md). The replay counts unique ballot/project questions, tests strong reuse comparators, and certifies the predicted allocation gap.

Given a cap on how many cases receive one binary review question, how much does guaranteeing minimum coverage for every data source change decision error?

This follow-up reuses five historical [Pabulib](https://pabulib.org/) approval-ballot collections and the pinned [Human-Centered Review](https://github.com/adelmansourjr/human-centered-review/tree/ce83d4a277efd8850e6d028c565fbfc594c504eb) posterior and query rule. There are 7,969 assessment ballots, each evaluated on 35 overlapping project triplets. All policies receive exactly 69,728 reviews per tape over 278,915 ballot-triplet cases. The two strong comparators are global ranking by predicted risk reduction and proportional source quotas with the same risk-reduction ranking within each source. The proposed policy reserves each source's best-ranked cases up to its minimum requirement, then allocates the remaining seats globally. This maximizes total predicted gain under the specified coverage constraints and fixed unit query cost. The evaluator independently checks this property against exhaustive enumeration on 20 small cases.

Erskine, Yamagata and Santos-Rodríguez use entropy to decide which state-action pairs merit scarce feedback and separately estimate trainer reliability. This project asks a related allocation question about the distribution of review opportunities across sources in an offline decision task. [Active Query Selection for Crowd-Based Reinforcement Learning, 2025 preprint](https://arxiv.org/abs/2508.19132). Their [IJCAI 2024 demonstration](https://ijcai.org/proceedings/2024/0999.pdf) also explores requesting richer human annotations when model behavior calls for them.

## Results

The original 20% floor never bound in the nominal condition or either stress condition. It selected the same set of cases as the global policy. The prerecording threshold of a 5-percentage-point improvement in minimum coverage thus failed. [Frozen protocol](PROTOCOL.md) · [Primary results](results.json)

The exploratory frontier tightened the constraint after the primary result. Errors below are pooled across cases and averaged over eight common report-noise tapes. Review-answer randomness is integrated exactly under each stated simulation.

| Policy | Nominal minimum source coverage | Nominal error | Error at 72% reviewer accuracy | Error with shared report flips |
|---|---:|---:|---:|---:|
| Global gain ranking | 21.6658% | 7.8864% | 11.2863% | 12.4100% |
| 22% coverage floor | 22.0001% | 7.7429% | 11.1426% | 12.4100% |
| 23% coverage floor | 23.0001% | 7.7614% | 11.1835% | 12.4167% |
| 24% coverage floor | 24.0002% | 7.8866% | 11.3431% | 12.4501% |
| 24.9% coverage floor | 24.9004% | 8.0381% | 11.5317% | 12.2619% |
| Proportional with gain ranking | 24.9962% | 8.0553% | 11.5506% | 12.2036% |

The 24% floor increased nominal minimum source coverage by 2.33 percentage points with a 0.0002-percentage-point increase in pooled nominal error. Equal-source average nominal error increased from 15.81% to 16.42%. The source receiving more reviews was the largest one (6,554 assessment ballots), so the case-weighted and equal-source error summaries answer different questions. Proportional allocation provided higher coverage and the lowest case-weighted error under shared report flips. The observed tradeoff depends on the coverage floor, how errors are weighted, and whether the ranking scores carry across sources. [Exploratory design](FRONTIER_DESIGN.md) · [Frontier results](frontier_results.json)

Historical ballots supply approvals; the incomplete reports and reviewer answers are simulated. The source is a data-collection identifier. These results establish a workload and error tradeoff in this replay and do not estimate demographic fairness, labor in minutes, performance with recruited reviewers, or generalization to new data collections. Overlapping triplets and cross-fit folds share data. The eight tapes vary the observation noise on the same ballots.

## Reproduce

Use Python 3.12 and NumPy 2.3.5. No trained model is needed. The evaluator verifies three upstream code hashes, and the input loader verifies each source data file against its pinned SHA-256. Enter the commands from the directory containing this README.

```bash
python -m pip install numpy==2.3.5
git clone https://github.com/adelmansourjr/human-centered-review.git review-upstream
git -C review-upstream checkout --detach ce83d4a277efd8850e6d028c565fbfc594c504eb
python review-upstream/pabulib_data.py public-ballots
export REVIEW_UPSTREAM="$PWD/review-upstream"
python replay.py --data-dir public-ballots --verify results.json
python frontier.py --data-dir public-ballots --verify frontier_results.json
```

Only aggregate data are stored in the two result files. The archive inputs are downloaded from the [pinned Pabulib revision](https://github.com/pabulib/pabulib_files/tree/beda04462f2fbee240202510b860106ee9754dd4).
