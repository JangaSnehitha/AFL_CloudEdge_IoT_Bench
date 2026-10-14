# Phase 15 Analysis Protocol: Divergence-versus-Staleness Instrumentation and Fairness Metrics

**Document Version:** 1.0.0  
**Phase:** 15  
**Context:** Asynchronous Federated Learning (AFL) Benchmark for Cloud-Edge-IoT Systems  
**Research Question:** *Does update/model divergence predict harmful asynchronous aggregation actions better than model-version staleness alone?*

---

## 1. Research Question and Scientific Framing

In asynchronous federated learning (AFL), clients transmit parameter updates asynchronously to a central server. Because clients experience heterogeneous compute speeds and network delays, arriving updates are inevitably stale ($\tau = v_{\text{server}} - v_{\text{client\_train}} \ge 0$). Standard AFL algorithms (e.g., FedAsync) discount updates purely as a function of version staleness $\tau$.

However, in non-IID and heterogeneous Cloud–Edge IoT topologies, parameter divergence may arise from data heterogeneity rather than staleness alone. Conversely, a stale update from an IID client may align well with the global optimization trajectory, while a fresh update from an extreme outlier may perturb the server model harmfully.

### Core Scientific Questions
1. **Predictive Superiority**: Does parameter/update divergence provide a stronger signal of harmful aggregation actions than integer version staleness $\tau$?
2. **Complementarity**: Does combining staleness and divergence outperform either predictor individually?
3. **Algorithm Specificity**: How do predictive relationships differ between continuous unbuffered updates (FedAsync) and buffered quasi-synchronous updates (FedBuff)?
4. **Fairness / Representation Interactions**: How do asynchronous staleness and divergence correlate with speed tier skew, client contribution inequality, and worst-class performance?

### Strict Non-Assumption Rule
This protocol explicitly does **not** assume that divergence outperforms staleness. The benchmark is designed to measure empirical correlations and ranking performance objectively. No algorithmic decisions or hyperparameter tuning may use test split data.

---

## 2. Mathematical Formulations of Parameter Predictors

Let $\theta \in \mathbb{R}^d$ denote the flattened parameter vector of the model (including weights and biases of all trainable layers in consistent floating-point IEEE 754 precision).

For any client update $i$ arriving at the server:
- $\theta_{\text{base}, i} \in \mathbb{R}^d$: The exact global model parameters downloaded by client $i$ at the start of its local training, corresponding to server version $v_{\text{client\_train}, i}$.
- $\theta_{\text{client}, i} \in \mathbb{R}^d$: The trained parameter vector returned by client $i$.
- $\Delta_i = \theta_{\text{client}, i} - \theta_{\text{base}, i} \in \mathbb{R}^d$: The local parameter update vector (pseudo-gradient).
- $\theta_{\text{server}, t} \in \mathbb{R}^d$: The global server model immediately **before** aggregation action $t$, corresponding to current server version $v_{\text{server}, t}$.
- $\Delta_{\text{server, prev}} = \theta_{\text{server}, t} - \theta_{\text{server}, t-1} \in \mathbb{R}^d$: The server parameter change resulting from the immediately preceding server aggregation step.
- $\epsilon = 10^{-8}$: Numerical stabilization constant.

### Predictor Definitions

#### A. Model-Version Staleness ($\tau_i$)
$$\tau_i = v_{\text{server}, t} - v_{\text{client\_train}, i}$$
- **Domain**: Non-negative integer $\tau_i \in \{0, 1, 2, \dots\}$.
- **Interpretation**: Number of global model parameter revisions that elapsed while client $i$ was downloading, training, and uploading.

#### B. Update $L_2$ Norm ($\|\Delta_i\|_2$)
$$\|\Delta_i\|_2 = \sqrt{\sum_{k=1}^d (\theta_{\text{client}, i, k} - \theta_{\text{base}, i, k})^2}$$
- **Domain**: $[0, \infty)$.
- **Interpretation**: Absolute magnitude of the client's local parameter step.

#### C. Relative Update Norm ($\rho_i$)
$$\rho_i = \frac{\|\Delta_i\|_2}{\|\theta_{\text{base}, i}\|_2 + \epsilon}$$
- **Domain**: $[0, \infty)$.
- **Interpretation**: Magnitude of the local update normalized by the scale of the base model.

#### D. Normalized Current-Model Divergence ($\delta_{i, t}$)
$$\delta_{i, t} = \frac{\|\theta_{\text{client}, i} - \theta_{\text{server}, t}\|_2}{\|\theta_{\text{server}, t}\|_2 + \epsilon}$$
- **Domain**: $[0, \infty)$.
- **Interpretation**: Euclidean distance between the trained client model and the server's current state, normalized by current server parameter norm. Reflects total displacement across both local training and intervening server updates.

#### E. Cosine Disagreement with Previous Server Update ($d_{\text{cos}, i}$)
$$d_{\text{cos}, i} = 1 - \frac{\langle \Delta_i, \Delta_{\text{server, prev}} \rangle}{\max(\|\Delta_i\|_2 \|\Delta_{\text{server, prev}}\|_2, \epsilon)}$$
- **Domain**: $[0, 2]$ when defined; otherwise `null`.
- **Undefined Conventions**:
  1. If $t = 0$ (first aggregation step, where no previous server update exists), $d_{\text{cos}, i} = \text{null}$ with flag `cosine_available = False`.
  2. If $\|\Delta_i\|_2 < 10^{-12}$ or $\|\Delta_{\text{server, prev}}\|_2 < 10^{-12}$, $d_{\text{cos}, i} = \text{null}$ with flag `cosine_available = False`.
  3. No fabricated or imputed values are permitted.

#### F. Optional Predictor: Local Loss ($\mathcal{L}_{\text{local}, i}$)
- The empirical training loss reported by client $i$ on its local batch at the conclusion of training.
- **Reporting Rule**: This is logged as local termination loss, NOT as a loss change, because before/after losses are not evaluated on a fixed local test partition.

---

## 3. Harmful-Impact Ground Truth

### Target Metric
Ground truth is defined exclusively by change in loss evaluated on the fixed centralized **validation** dataset $\mathcal{D}_{\text{val}}$ (500 samples, fixed seed):
$$\text{impact}(A_t) = \mathcal{L}_{\text{val}}(\theta_{\text{server, before}}) - \mathcal{L}_{\text{val}}(\theta_{\text{server, after}})$$

### Interpretation
- $\text{impact}(A_t) > 0$: The aggregation action **improved** global validation loss (loss decreased).
- $\text{impact}(A_t) < 0$: The aggregation action **worsened** global validation loss (loss increased).
- $\text{impact}(A_t) = 0$: Neutral impact.

### Binary Harmful Labeling
With numerical tolerance $\delta_{\text{tol}} = 10^{-5}$:
$$Y_{\text{harmful}}(A_t) = \begin{cases} 1 & \text{if } \text{impact}(A_t) < -\delta_{\text{tol}} \\ 0 & \text{otherwise} \end{cases}$$

### Non-Interference Invariant
Instrumentation evaluates $\mathcal{L}_{\text{val}}(\theta)$ non-destructively:
- It does not modify model weights, optimizer states, random number generators, client queues, or event timestamps.
- It does not filter, alter, or reject any updates based on impact.

---

## 4. Observation Units: FedAsync versus FedBuff

| Attribute | FedAsync | FedBuff ($K$) |
|---|---|---|
| **Aggregation Unit** | Single arriving client update | Buffer of $K$ accumulated updates |
| **Action Definition** | Immediate parameter update upon client arrival | Joint parameter update when buffer size reaches $K$ |
| **Impact Attribution** | Attributed to the arriving client's action | Attributed **only** to the joint buffer action |
| **Individual Ground Truth** | Well-defined for each applied update | **Undefined**; joint impact must NOT be copied to individual updates |
| **Predictor Granularity** | Pre-action metrics of the single client | (1) Per-member metrics at arrival and application<br>(2) Buffer-level aggregates (mean, max) |

### Buffer-Level Predictor Aggregations (FedBuff)
For a buffer $B = \{i_1, \dots, i_K\}$ applied at step $t$:
- Mean and max staleness: $\bar{\tau}_B = \frac{1}{K} \sum_{i \in B} \tau_i$, $\tau_{B, \max} = \max_{i \in B} \tau_i$
- Mean and max divergence: $\bar{\delta}_B = \frac{1}{K} \sum_{i \in B} \delta_i$, $\delta_{B, \max} = \max_{i \in B} \delta_i$
- Joint buffer update vector: $\Delta_B = \sum_{i \in B} w_i \Delta_i$, where $w_i = \frac{n_i}{\sum_{j \in B} n_j}$
- Buffer cosine disagreement: $1 - \text{cosine}(\Delta_B, \Delta_{\text{server, prev}})$

---

## 5. Precise Timing of Measurements

1. **Dispatch ($t_{\text{dispatch}}$)**:
   - Server snapshots $\theta_{\text{server}}(t_{\text{dispatch}})$ as $\theta_{\text{base}, i}$.
   - Dispatched version: $v_{\text{client\_train}, i}$.
2. **Arrival ($t_{\text{arrival}}$)**:
   - Client upload completes.
   - Record arrival staleness: $\tau_{\text{arr}, i} = v_{\text{server}}(t_{\text{arrival}}) - v_{\text{client\_train}, i}$.
   - In FedAsync: $t_{\text{apply}} = t_{\text{arrival}}$.
   - In FedBuff: Update is placed in buffer; $t_{\text{apply}} \ge t_{\text{arrival}}$.
3. **Application ($t_{\text{apply}}$)**:
   - Record application staleness: $\tau_{\text{apply}, i} = v_{\text{server}}(t_{\text{apply}}) - v_{\text{client\_train}, i}$.
   - Evaluate $\theta_{\text{server, before}} = \theta_{\text{server}}(t_{\text{apply}})$.
   - Compute predictors $B, C, D, E$ against $\theta_{\text{server, before}}$.
   - Apply aggregation rule $\to \theta_{\text{server, after}}$.
   - Evaluate $\theta_{\text{server, after}}$.
   - Compute $\text{impact} = \mathcal{L}_{\text{val}}(\theta_{\text{before}}) - \mathcal{L}_{\text{val}}(\theta_{\text{after}})$.

---

## 6. Fairness, Exposure, and Representation Metrics

### A. Jain's Fairness Index on Applied Updates
For all $N$ configured clients:
$$J(u) = \frac{\left(\sum_{i=0}^{N-1} u_i\right)^2}{N \sum_{i=0}^{N-1} u_i^2}$$
- **Range**: $[\frac{1}{N}, 1.0]$.
- **Population**: Must include all $N$ clients, including clients with $u_i = 0$.
- **Edge Case**: If $\sum_{i=0}^{N-1} u_i = 0$, $J(u) = \text{null}$ with explanation `"zero_total_contributions"`.

### B. Accounting Disaggregation
The system explicitly distinguishes:
1. `arrival_count`: Updates received at server.
2. `applied_count`: Updates incorporated into model revisions.
3. `sample_exposure`: $\sum_{j \in \text{applied}} n_j$ samples contributed.
4. `aggregation_influence`: Effective accumulated weighting in model updates.

### C. Speed-Tier Contribution Shares
$$\text{share}_{\text{tier}} = \frac{\sum_{i \in \text{tier}} u_i}{\sum_{j=0}^{N-1} u_j}$$

### D. Label Exposure
Fraction of total uploaded training samples associated with each class label:
$$\text{exposure}(c) = \frac{\sum_{i=0}^{N-1} u_i \cdot n_{i, c}}{\sum_{c'} \sum_{i=0}^{N-1} u_i \cdot n_{i, c'}}$$

### E. Held-Out Evaluation Metrics
Evaluated on centralized validation and test splits:
- Per-class accuracy: $\text{Acc}_c = \frac{\text{TP}_c}{N_c}$ along with support counts $N_c$.
- Macro-averaged accuracy: $\frac{1}{C} \sum_{c=1}^C \text{Acc}_c$.
- Worst-class accuracy: $\min_c \text{Acc}_c$.
- If a class has zero evaluation samples, its accuracy is reported as `null` (not 0.0).

---

## 7. Statistical Evaluation of Predictors

### Hypothesized Orientation
All predictors are oriented such that **higher values are hypothesized to indicate higher probability of harmful updates** (i.e., lower or more negative impact):
- Higher staleness $\tau \to$ higher harm.
- Higher update norm $\|\Delta\| \to$ higher harm.
- Higher divergence $\delta \to$ higher harm.
- Higher cosine disagreement $d_{\text{cos}} \to$ higher harm.

### Metrics Computed
1. **Spearman Rank Correlation ($r_s$)**:
   $$\text{corr}(\text{predictor}, \text{impact})$$
   Hypothesized sign: negative (larger predictor $\to$ smaller/negative impact).
2. **AUROC**:
   Area under the ROC curve treating $Y_{\text{harmful}} = 1$ as positive class.
   Hypothesized value: $> 0.50$.
3. **Average Precision (PR-AUC)**:
   Area under Precision-Recall curve for $Y_{\text{harmful}} = 1$.
   Baseline: harmful update prevalence $\pi = \frac{N_{\text{harmful}}}{N_{\text{total}}}$.

### Degenerate Data Handling
- If all outcomes are negative or all positive ($N_{\text{harmful}} = 0$ or $N_{\text{harmful}} = N$): AUROC is `null` (`"single_class_outcome"`).
- If predictor is constant: $r_s$ and AUROC are `null` (`"constant_predictor"`).
- If sample size $N < 3$: metrics are `null` (`"insufficient_samples"`).

### Cross-Validation Framing
Single-seed bounded runs (e.g., 30 updates) serve as **instrumentation smoke tests**. Predictive claims and multi-variable model rankings must be evaluated across multi-seed benchmarks using Leave-One-Seed-Out (LOSO) cross-validation to prevent trajectory autocorrelation leakage.

---

## 8. Limitations of Server-Held Validation Data

1. **Proxy Nature**: Centralized validation loss evaluates global model performance on an aggregated distribution, which may not capture localized regression on minority client domains.
2. **Sample Size**: Validation set comprises 500 fixed samples to bound computational overhead during fine-grained per-action evaluation.
3. **Step Variance**: Individual mini-batch gradient updates introduce variance; single-action impacts may exhibit small-scale stochasticity.
