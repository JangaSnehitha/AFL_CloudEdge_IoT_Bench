# Project Specification: Asynchronous Federated Learning for Cloud-Edge IoT

## 1. Objective

Develop a reproducible benchmark to evaluate asynchronous federated learning
(AFL) algorithms under realistic heterogeneous Cloud-Edge IoT conditions.

This project does not propose a new AFL aggregation algorithm. It focuses on
systematic evaluation, robustness analysis, and configuration guidance for
existing AFL methods.

## 2. Research questions

RQ1: How do AFL algorithms perform as statistical heterogeneity, compute
heterogeneity, network latency, and client dropout increase?

RQ2: Does timestamp staleness predict harmful asynchronous updates as well as
model/update divergence?

RQ3: Which AFL configuration provides the best trade-off among convergence
time, accuracy, communication cost, and robustness?

RQ4: How do buffer size, local epochs, and client participation policy affect
the accuracy-time-communication trade-off?

RQ5: Do fast clients dominate asynchronous training and reduce performance for
underrepresented slow-client data?

## 3. Algorithms

- FedAvg: synchronous baseline.
- FedAsync: asynchronous aggregation with staleness weighting.
- FedBuff: buffered asynchronous aggregation.
- FedASMU: optional advanced comparison.
- FedSA: optional advanced comparison.

## 4. Datasets

- Synthetic classification dataset for initial testing.
- Fashion-MNIST or CIFAR-10 for main experiments.
- Optional IoT/sensor dataset for validation.

## 5. Heterogeneity dimensions

- IID and non-IID data distributions.
- Heterogeneous client compute speed.
- Dynamic network latency and bandwidth.
- Client dropout and rejoining behavior.
- Optional data drift.

## 6. Metrics

- Global test accuracy and loss.
- Virtual wall-clock time.
- Time to target accuracy.
- Communication volume.
- Number of accepted updates.
- Staleness distribution.
- Update/model divergence.
- Harmful-update impact.
- Worst-client accuracy.
- Client update contribution share.
- Jain's fairness index.

## 7. Experimental scenarios

| Scenario | Data | Compute | Network | Availability |
|---|---|---|---|---|
| S1 Stable edge | Near IID | Mild | Static low latency | Always online |
| S2 Statistical skew | Extreme non-IID | Mild | Static | Always online |
| S3 Resource imbalance | Moderate non-IID | Severe | Variable | Mostly online |
| S4 Unreliable IoT | Moderate non-IID | Severe | Bursty/high latency | 10-30% dropout |
| S5 Adversarially coupled | Extreme non-IID | Slow clients hold rare labels | Bursty latency | Rejoin with stale updates |

## 8. Reproducibility

- Use fixed random seeds.
- Save every experiment configuration.
- Save all metrics to CSV or JSON.
- Use identical client partitions and client profiles across algorithms for the same seed.
- Run final experiments with at least 3-5 seeds.
- Do not tune hyperparameters on the global test set.

## 9. Development order

1. Centralized model baseline.
2. Synchronous FedAvg.
3. IID and non-IID data partitioning.
4. Event-driven asynchronous simulator.
5. FedAsync.
6. FedBuff.
7. Heterogeneity profiles.
8. Divergence-versus-staleness analysis.
9. Large controlled experiments.
10. Analysis, plots, and report.
