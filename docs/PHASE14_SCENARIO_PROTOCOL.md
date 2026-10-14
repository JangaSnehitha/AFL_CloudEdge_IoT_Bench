# Phase 14: Dynamic Cloud–Edge IoT Scenario Engine Protocol

## 1. Scope and Scientific Purpose

This protocol specifies the design, lifecycle, semantics, and accounting rules for the dynamic Cloud–Edge IoT scenario engine in the Asynchronous Federated Learning (AFL) benchmark.

**Project Framing**:
The core contribution of this work is a reproducible, interaction-focused Cloud–Edge IoT benchmark and a future divergence-versus-staleness study. The synthetic scenarios defined here model realistic interaction dynamics (heterogeneous compute, bandwidth constraints, latency spikes, outages, dropouts, and rejoins) without claiming direct physical-device or trace-driven realism. All scenarios operate over controlled synthetic system parameters.

---

## 2. Analysis of Existing Event Scheduling (Phases 9 & 13)

Before introducing dynamic system profiles, the baseline event lifecycle in `src/fedasync.py` and `src/fedbuff.py` operated as follows:

1. **Compute Time and Network Delay**:
   - `compute_duration = compute_time * local_epochs` (default: $1.0 \times 1 = 1.0\text{ s}$).
   - `comm_delay = delay_simulator.sample_delay(cid, server_version)` (sampled uniformly from $[0, \text{max\_delay}]$).
   - Arrival occurred at $t_{\text{arrival}} = t_{\text{dispatch}} + \text{compute\_duration} + \text{comm\_delay}$. Network transmission was not decomposed into download, upload, bandwidth, model bytes, or latency.
2. **Model Snapshot Timing**:
   - The client obtained its model parameters synchronously at the moment of dispatch: $\theta_{\text{client}} \leftarrow \theta_{\text{server}}(t_{\text{dispatch}})$. Download transmission delay was zero.
3. **Redispatch Timing**:
   - Clients were redispatched immediately upon `UPDATE_ARRIVAL` at the server if the target update budget was not yet exhausted. In FedBuff, redispatch occurred immediately after buffer aggregation if the arrival triggered aggregation.
4. **Outstanding Jobs per Client**:
   - At most one active job per client was in flight at any given moment.
5. **Equal-Time Event Ordering and Tie-Breaking**:
   - The priority queue ordered events by tuple `(timestamp, priority, event_counter, event_dict)`.
   - Priorities: `DISPATCH` (1), `TRAIN_COMPLETE` (2), `UPDATE_ARRIVAL` (3).
   - Strict FIFO tie-breaking was guaranteed by a monotonically increasing `event_counter`.

---

## 3. Physical Units and Dimensional Specifications

All virtual-time physical dimensions in Phase 14 are strictly positive and use standardized units:

| Dimension | Unit | Symbol | Notes / Conversion |
|---|---|---|---|
| **Virtual Time** | Seconds | $\text{s}$ | Floating-point simulated time, strictly non-decreasing |
| **Model Size** | Bytes | $\text{B}$ | Exact byte count from tensor parameters: $\sum_l \text{numel}(W_l) \times \text{element\_size}(W_l)$ (14,224 B for the benchmark MLP in float32) |
| **Bandwidth** | Bytes per second | $\text{B/s}$ | Network transfer throughput ($1\text{ KB/s} = 1,024\text{ B/s}$, $1\text{ MB/s} = 1,048,576\text{ B/s}$) |
| **Network Latency** | Seconds | $\text{s}$ | Base propagation and queuing delay ($1\text{ ms} = 0.001\text{ s}$) |
| **Compute Factor** | Unitless factor | $c_{\text{factor}}$ | Multiplier on base execution time ($c_{\text{factor}} \in [0.2, 5.0]$) |

---

## 4. Persistent Client Profiles

Clients belong to persistent speed tiers that reflect realistic device classes in Cloud–Edge IoT topologies:

1. **Fast Tier (Cloud / High-End Edge)**:
   - High compute capability: $c_{\text{factor}} = 0.5$
   - High bandwidth: Download $2\text{ MB/s}$ ($2,097,152\text{ B/s}$), Upload $1\text{ MB/s}$ ($1,048,576\text{ B/s}$)
   - Low latency: $\text{latency} = 0.010\text{ s}$ ($10\text{ ms}$)
2. **Medium Tier (Standard Edge Gateway / Mobile)**:
   - Standard compute capability: $c_{\text{factor}} = 1.0$
   - Standard bandwidth: Download $512\text{ KB/s}$ ($524,288\text{ B/s}$), Upload $256\text{ KB/s}$ ($262,144\text{ B/s}$)
   - Moderate latency: $\text{latency} = 0.050\text{ s}$ ($50\text{ ms}$)
3. **Slow Tier (Constrained IoT Sensor / Microcontroller)**:
   - Constrained compute capability: $c_{\text{factor}} = 2.5$
   - Constrained bandwidth: Download $64\text{ KB/s}$ ($65,536\text{ B/s}$), Upload $32\text{ KB/s}$ ($32,768\text{ B/s}$)
   - High latency: $\text{latency} = 0.200\text{ s}$ ($200\text{ ms}$)

### Explicit Timing Formulas

1. **Download Duration**:
   $$T_{\text{download}}(t_{\text{start}}) = \text{base\_latency}(t_{\text{start}}) + \frac{\text{model\_bytes}}{\text{download\_bandwidth}(t_{\text{start}})}$$
2. **Training Computation Duration**:
   $$T_{\text{compute}} = E \times \left\lceil \frac{n_k}{B} \right\rceil \times c_{\text{factor}} \times t_{\text{step}}$$
   where $E$ is local epochs, $n_k$ is client sample count, $B$ is batch size, and $t_{\text{step}} = 0.02\text{ s}$ is the calibrated base step time.
3. **Upload Duration**:
   $$T_{\text{upload}}(t_{\text{start}}) = \text{base\_latency}(t_{\text{start}}) + \frac{\text{model\_bytes}}{\text{upload\_bandwidth}(t_{\text{start}})}$$

---

## 5. Dynamic Network Conditions and Sampling Rule

Dynamic network conditions allow time-varying disturbances:
- **Stable Network**: Base bandwidth and latency maintained.
- **Latency Spike**: Additive latency disturbance $\Delta L$ over a time window $[t_{\text{start}}, t_{\text{end}}]$.
- **Bandwidth Degradation**: Multiplicative throttle $\gamma \in (0, 1)$ on download/upload bandwidth over $[t_{\text{start}}, t_{\text{end}}]$.
- **Recovery**: Automatic return to baseline after $t_{\text{end}}$.
- **Client Network Improvement**: Step change permanently upgrading a constrained client's connectivity.

**Deterministic Sampling Rule**:
Network conditions (bandwidth and latency) are **sampled at the start of each transmission stage** ($t_{\text{trans\_start}}$). Integrating varying bandwidth across in-flight transmissions is non-deterministic under discrete event simulation; sampling at transmission start yields an exact, closed-form transmission duration $T_{\text{trans}}$ recorded deterministically in the event queue.

---

## 6. Dropout, Outages, and Rejoin Policies

### Failure Stages
A job $j$ can fail at three explicit lifecycle stages:
1. `BEFORE_DOWNLOAD`: Client disconnected when dispatch attempted. Dispatch rejected or fails immediately.
2. `DURING_TRAINING`: Client experiences device crash, thermal shutdown, or battery depletion during local training.
3. `BEFORE_UPLOAD`: Network drops out while client attempts to push updated parameters to server.

### Outage and Rejoin Mechanics
- **Scheduled Outage**: Specified by virtual-time interval $[t_{\text{outage\_start}}, t_{\text{outage\_end}}]$. Any transmission or training active during this window experiences failure.
- **Independent Dropouts**: Stochastic failure probability per job $p_{\text{fail}}$ evaluated using dedicated RNG stream `dropout_rng`.

### Rejoin Policies
When a client regains connectivity at $t_{\text{rejoin}} \ge t_{\text{outage\_end}}$:
- **Policy A (`fresh_restart`)**: The client discards any aborted or pending local work. It downloads the *current* global model version $\theta_{\text{server}}(t_{\text{rejoin}})$ and restarts training fresh.
- **Policy B (`retained_update`)**: If the client *successfully finished local training* prior to the upload disruption, it buffers the completed parameter update $\theta_{\text{client}}$. Upon rejoining at $t_{\text{rejoin}}$, it uploads this retained update carrying its *original* training version $\tau_{\text{client}}$. The server computes actual model-version staleness at arrival:
  $$\tau = v_{\text{server}}(t_{\text{arrival}}) - v_{\text{client, training}}$$

**Integrity Constraints**:
- Completed updates are **never fabricated** if training was disrupted.
- Retained updates **never overwrite** their original training version; model versioning is immutable.
- At most **one active or retained job** exists per client at any time.
- Every job has a unique immutable `job_id` (`job_<cid>_<seq>`) ensuring no update is aggregated twice.

---

## 7. Client/Data Coupling

In realistic IoT deployments, physical capabilities are often coupled with data generation (e.g. specialized edge sensors collect distinct minority classes but possess weaker radios).

- **Coupling Mechanism**: Phase 14 maps existing data partitions to persistent client profiles based on partition properties (e.g., assigning the client with the highest Dirichlet class skew to the `slow` tier).
- **Data Integrity**: Coupling **never** alters the underlying synthetic dataset, validation split, or test split. The evaluation sets remain strictly bitwise identical to prior phases.
- **Auditability**: All profile-to-partition assignments and label distributions are exported to the result JSON payload.

---

## 8. Random-Seed Architecture

To ensure perfect cross-method replay and isolation, Phase 14 employs four independent, explicitly seeded RNG streams:
1. `data_seed`: Seeds synthetic dataset generation and Dirichlet/IID partitioning.
2. `profile_seed`: Seeds client-to-tier assignments when stochastic profile generation is used.
3. `network_seed`: Seeds stochastic latency jitter or bandwidth fluctuations.
4. `dropout_seed`: Seeds stochastic dropouts and job failures.

Exogenous scenario timelines can also be defined via fully deterministic static JSON schedules.

---

## 9. Stopping Budgets and Accounting Rules

To prevent infinite simulation execution when clients suffer prolonged outages or high delays, simulations are bounded by three hard stopping thresholds:
1. `target_applied_updates`: Primary stopping target (e.g., 15 or 300 applied client updates).
2. `max_attempted_jobs`: Maximum job dispatches allowed (default: $5 \times \text{target\_applied\_updates}$).
3. `max_virtual_time`: Maximum virtual time allowed (default: $10,000.0\text{ s}$).

### Accounting Distinctions
The engine strictly differentiates between:
- `jobs_dispatched`: Number of jobs initialized.
- `training_starts`: Jobs that successfully completed download and began training.
- `training_completions`: Jobs that finished local training.
- `failed_jobs`: Disrupted jobs (categorized by failure stage).
- `retained_updates`: Completed jobs preserved across outages.
- `arrivals`: Updates successfully received at the server.
- `applied_updates`: Updates actually aggregated into the global model.
- `server_aggregations`: Number of global parameter update steps (1 per arrival in FedAsync, 1 per $K$ arrivals in FedBuff).
- `pending_jobs`: Incomplete jobs remaining at simulation termination.

---

## 10. Named Benchmark Scenarios (S1 – S5)

| Scenario | Title | Data Partition | System Profiles | Network Conditions | Outages & Dropouts | Rejoin Policy |
|---|---|---|---|---|---|---|
| **S1** | Stable Edge | Near-IID ($\alpha=10.0$ or IID) | Mild compute variation (Fast/Med) | Stable, high bandwidth | None | `fresh_restart` |
| **S2** | Statistical Skew | Severe Dirichlet ($\alpha=0.1$) | Uniform medium profiles | Stable network | None | `fresh_restart` |
| **S3** | Resource Imbalance | Moderate Dirichlet ($\alpha=0.5$) | 20% Fast, 50% Med, 30% Slow | Time-varying latency jitter | None | `fresh_restart` |
| **S4** | Unreliable IoT | Moderate Dirichlet ($\alpha=0.5$) | Heterogeneous (Fast/Med/Slow) | Periodic latency spike & throttle | Scheduled outage & $p_{\text{fail}}=0.1$ | `retained_update` |
| **S5** | Coupled Stress | Strong Dirichlet ($\alpha=0.2$) | Skewed clients coupled to Slow tier | Degraded bandwidth on slow tier | Disruption on coupled clients | `retained_update` |

---

## 11. Integration Scope and Deferred Items

- **FedAsync Integration**: Updates aggregate immediately upon arrival with staleness-discounted weighting.
- **FedBuff Integration**: Updates enter server buffer; sample-count-weighted aggregation occurs when buffer reaches $K$.
- **Synchronous FedAvg Note**: Synchronous FedAvg cannot be directly compared against virtual time without a dedicated synchronous barrier adapter (where round duration equals $\max_{k} T_{\text{round}, k}$). In Phase 14, synchronous FedAvg is not executed.
- **Deferred Proposal Item (Data Drift)**: Dynamic concept drift (temporal shifts in client class distributions $P_t(Y|X)$ or $P_t(X)$) is documented as a deferred future item and is **not** implemented in this phase.
