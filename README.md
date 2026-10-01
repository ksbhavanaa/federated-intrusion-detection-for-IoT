# Real-Time Federated Intrusion Detection System for IoT Networks

An academic research platform and Security Operations Center (SOC) dashboard implementing multi-class intrusion detection across distributed IoT device networks using Federated Learning (FedAvg), Explainable AI (SHAP), dynamic risk-based mitigation, and robust research benchmarking.

---

## 1. Project Objective
To detect, categorize, explain, and mitigate network intrusions across heterogeneous IoT edge devices without centralizing private network traffic telemetry. The platform couples centralized baselines with sample-weighted Federated Averaging, zero-sum secure aggregation simulation, client dropout resilience, non-stationary concept drift adaptation, and sub-millisecond inference line rates.

---

## 2. System Architecture
```
                                ┌────────────────────────────────────────┐
                                │     Central Aggregation Server         │
                                │   - Global Model: FederatedMLP         │
                                │   - Parameter Aggregation: FedAvg      │
                                │   - Secure Aggregation Simulation      │
                                └──────────────────┬─────────────────────┘
                                                   │
                ┌──────────────────┬───────────────┴──────────────┬──────────────────┐
                ▼                  ▼                              ▼                  ▼
        ┌──────────────┐   ┌──────────────┐               ┌──────────────┐   ┌──────────────┐
        │ Smart Camera │   │ WiFi Router  │               │ Sensor Node  │   │  Door Lock   │
        │ (DDoS-Heavy) │   │ (Scan-Heavy) │               │ (Bot-Heavy)  │   │(Brute-Heavy) │
        └──────────────┘   └──────────────┘               └──────────────┘   └──────────────┘
                │                  │                              │                  │
                └──────────────────┴───────────────┬──────────────┴──────────────────┘
                                                   │
                                                   ▼
                                  ┌────────────────────────────────┐
                                  │      SOC Operations Engine     │
                                  │  - PyTorch Real-Time Detector  │
                                  │  - Multi-Class Inference (6)   │
                                  │  - Dynamic Risk Scoring Engine │
                                  │  - Software Mitigation Policy  │
                                  │  - Gradient * Input (SHAP XAI) │
                                  │  - SQLite Research DB (13 tbl) │
                                  │  - Modern SOC Web Dashboard    │
                                  └────────────────────────────────┘
```

---

## 3. Dataset & Data Provenance
- **Benchmark Source:** Canadian Institute for Cybersecurity (CICIDS2017).
- **Files Processed:** `Monday.csv`, `Tuesday.csv`, `Wednesday.csv`, `Thursday1.csv`, `Thursday2.csv`, `Friday.csv`, `Friday1.csv` ($2,639,710$ raw records).
- **Stratified Working Sample:** $50,032$ records across $78$ continuous network flow features.
- **Strict Leakage Prevention:** `StandardScaler` fitted strictly on the training partition ($N=35,862$). Held-out test set ($N=10,247$, $20\%$) evaluated with zero sample contamination ($100\%$ passed automated leakage verification).
- **Academic Data Disclosure:** 5 attack classes are extracted directly from genuine CICIDS2017 PCAP files. Because the Ares botnet capture (`Friday-WorkingHours-Morning.pcap_ISCX.csv`) was omitted in the source files, Botnet flows ($N=1,200$) were augmented using the published IRC-C2 flow profile before partitioning.

---

## 4. Six Canonical Attack Classes
The system enforces an authoritative 6-class mapping:
1. `Normal` (Legitimate baseline network traffic)
2. `DDoS` (Volumetric distributed denial of service / HTTP flooding)
3. `Botnet` (Command-and-Control zombie communication profile)
4. `PortScan` (Host and service network reconnaissance sweeps)
5. `BruteForce` (Credential stuffing against SSH and FTP interfaces)
6. `WebAttack` (HTTP exploits, SQL injection, and cross-site scripting)

Neural network output dimension is strictly $6$ (`Linear(64, 6)`) across all models and endpoints.

---

## 5. Centralized ML Benchmarking
Evaluated on the exact same $10,247$ held-out test split:
- **Random Forest ($n=100$):** Accuracy: **99.80%**, Macro F1: **0.9881**, Weighted F1: **0.9980**
- **XGBoost ($n=100, d=6$):** Accuracy: **99.90%**, Macro F1: **0.9889**, Weighted F1: **0.9990**
- **Centralized Neural Network (MLP):** Accuracy: **98.91%**, Macro F1: **0.8029**, Weighted F1: **0.9882**

---

## 6. Federated Model Architecture
Because decision trees cannot be aggregated via continuous gradient averaging, the federated network utilizes a dedicated PyTorch neural network (**FederatedMLP**):
$$\text{Linear}(78 \to 256) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(256 \to 128) \to \text{BatchNorm} \to \text{ReLU} \to \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Linear}(64 \to 6)$$

---

## 7. Mathematical FedAvg Parameter Aggregation
Client model weights $W_k$ are aggregated using sample-weighted Federated Averaging:
$$W_{\text{global}} = \sum_{k=1}^K \left( \frac{n_k}{\sum_{j=1}^K n_j} \right) W_k$$
where $n_k$ is the number of local training samples at client $k$. Verified via automated unit tests ($0.25 \times 2.0 + 0.75 \times 6.0 = 5.000000 \pm 10^{-7}$).

---

## 8. IID Experiment
Stratified uniform distribution across 5 clients ($~10,246$ flows each).
- **Rounds:** 3
- **Test Accuracy:** **96.09%**
- **Macro F1:** **0.6210**
- **Communication:** **2,550.29 KB**

---

## 9. Non-IID Experiment (Heterogeneous IoT Fleet)
Realistic device-specific attack skew reflecting IoT role vulnerabilities:
- **Smart Camera ($13,176$ flows):** DDoS-heavy ($4,795$ DDoS flows)
- **WiFi Router ($9,978$ flows):** PortScan-heavy ($1,351$ scan flows)
- **Temperature Sensor ($10,758$ flows):** Botnet-heavy ($840$ botnet flows)
- **Smart Door Lock ($8,843$ flows):** BruteForce-heavy ($135$ credential attacks)
- **Smart Light Controller ($8,764$ flows):** WebAttack-heavy ($56$ web exploit flows)
- **Results (5 Rounds):** Test Accuracy: **96.73%**, Macro F1: **0.6277**, Weighted F1: **0.9653**.

---

## 10. Client Dropout Resilience
Simulates unstable IoT wireless links with an active client schedule of $[5, 4, 5, 3, 5]$:
- Dropped nodes are strictly excluded from round weight aggregation.
- **5 Rounds:** Final Accuracy **96.79%**, Macro F1 **0.6293**, Total Comm **3,740.42 KB** (reduced from $4,250$ KB due to missing client uploads).

---

## 11. Secure Aggregation Simulation
Simulates zero-sum pairwise masking:
$$M_{ij} = -M_{ji} \implies \sum_{i=1}^K M_i = \mathbf{0}$$
The server aggregates masked client updates $\sum (W_i + M_i) = \sum W_i$.
- **Observed Aggregation Difference:** $\max |W_{\text{masked}} - W_{\text{unmasked}}| < 10^{-7}$.
- **Academic Classification:** Research simulation demonstrating algebraic mask cancellation; not a production cryptographic SMPC protocol.

---

## 12. Non-Stationary Concept Drift
Evaluated across four sequential attack emergence stages:
- **Stage 1 (Normal + DDoS):** $97.17\% \to 98.52\%$ (rapid adaptation)
- **Stage 2 (DDoS + PortScan):** $84.45\% \to 81.44\%$ (moderate drift)
- **Stage 3 (BruteForce + WebAttack):** $0.00\% \to 0.00\%$ (rare class shock; empirical finding on extreme imbalance)
- **Stage 4 (Botnet + DDoS):** $78.16\% \to 74.71\%$ (dominant recovery)

---

## 13. Scalability Benchmarking
- **3 Clients:** 10.59s training time, 4,590.53 KB comm, 96.47% accuracy
- **5 Clients:** 16.35s training time, 7,650.88 KB comm, 96.09% accuracy
- **8 Clients:** 22.18s training time, 12,241.41 KB comm, 96.17% accuracy
Demonstrates linear communication scaling ($O(K)$).

---

## 14. Real-Time Line Rate & Latency
High-precision evaluation using `time.perf_counter()` on 500 single-flow inferences:
- **Mean Inference Latency:** **0.224 ms**
- **Median Latency:** **0.237 ms**
- **95th Percentile (P95):** **0.263 ms**
- **Throughput:** **4,467.4 flows/sec**

---

## 15. Explainable AI (XAI): Gradient × Input Attribution
Computes local feature attributions using input $\times$ gradient first-order Taylor approximation ($A_i = x_i \cdot \frac{\partial y_c}{\partial x_i}$):
- **Attribution Methodology:** Analytical input-gradient attribution for neural network decision explanations (not sampling-based KernelSHAP or tree-based TreeSHAP).
- **Fidelity Score:** **93.3%** (verifies that zeroing the top-5 attributed features drops target-class confidence across held-out flows).
- **Stability Score:** **100.0%** (Jaccard consistency of top-5 features under Gaussian input perturbation $\sigma=0.02$).

---

## 16. Dynamic Risk Scoring Engine
Risk scores ($0–100$) are calculated deterministically:
$$\text{Risk} = \min(100, (\text{Severity} \times \text{Confidence} \times \text{DeviceCriticality} \times 80) + \text{ThreatFrequencyBonus})$$
- Attack Severities: Normal (0.05), WebAttack (0.65), PortScan (0.70), Botnet (0.85), BruteForce (0.85), DDoS (0.95).
- Device Criticalities: Gateway Router (1.25), Door Lock (1.20), Camera (1.00), Sensor (0.90), Lighting (0.85).

---

## 17. Software Mitigation Policy (Honest Disclosure)
- **Risk < 30:** `MONITOR` (Normal telemetry logging)
- **Risk 30–59:** `ALERT` (SOC operator notification)
- **Risk 60–84:** `SIMULATED QUARANTINE` (Rate-limiting decision simulation)
- **Risk 85–100:** `SIMULATED ISOLATION` (Device isolation decision simulation)

*Academic Disclosure:* All mitigations operate as software-level decision simulations without modifying production firewalls or routing tables.

---

## 18. Security Operations Center (SOC) Dashboard
Interactive web interface built with Vanilla CSS and responsive JavaScript:
- **Overview:** High-level KPIs, Fleet Status, and Dynamic Model Topology (`Linear(64, 6)`).
- **Live Threat IDS:** Interactive attack injection passing flows through the unified detection pipeline.
- **IoT Device Fleet:** Device-level risk indicators and mitigation state history.
- **Federated Learning:** Convergence charts with experiment-scoped round numbering.
- **Explainable AI:** Feature waterfall charts displaying top positive and negative contributors.
- **Security Alerts:** Persistent alert registry with direct SHAP explainability linkage.
- **Analytics & Benchmarks:** 6x6 interactive Confusion Matrix and Centralized vs. Federated comparison.
- **Research Experiments:** SQLite catalog tracking 18 completed empirical experiments.

---

## 19. SQLite Database Schema
Database file: `results/ids_research.db` (13 normalized relational tables):
1. `users`: SOC analysts and operator credentials.
2. `devices`: 5 IoT device state and risk telemetry.
3. `predictions`: Real-time inference log with microsecond latency.
4. `alerts`: Unique persistent threat notifications with feature JSON.
5. `federated_rounds`: Round-by-round convergence and communication metrics.
6. `client_metrics`: Local client training loss and validation accuracy.
7. `fl_experiments`: Federated experiment catalog.
8. `detection_latency`: High-resolution latency profiling records.
9. `xai_results`: Stored feature importance, fidelity, and stability scores.
10. `mitigation_history`: Automated risk-mitigation decision audit trail.
11. `concept_drift_experiments`: Multi-stage drift evaluation records.
12. `scalability_experiments`: Multi-client training and communication metrics.
13. `research_experiments`: Unified research experiment registry.

---

## 20. Reproducibility & Quickstart Guide

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Scikit-learn, Pandas, NumPy, XGBoost, ReportLab, Flask

### Run Full Pipeline
```powershell
# Step 1: Preprocess dataset and verify zero data leakage
python -m src.data.preprocessor

# Step 2: Generate Non-IID IoT device shards
python src/data/client_splitter.py --mode non_iid

# Step 3: Train centralized baselines
python src/models/s2_random_forest.py
python src/models/s2_xgboost.py
python src/models/pytorch_nn.py

# Step 4: Train federated global model (10 rounds)
python src/federated/fed_server.py

# Step 5: Execute full research evaluation suite
python src/federated/experiments.py

# Step 6: Run automated test suite
python -m pytest tests/ -v

# Step 7: Launch dashboard application
python run.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 21. Research Limitations
1. **Calibrated Botnet Flow Augmentation:** Friday morning PCAP was absent from source folder; 1,200 Botnet flows were synthesized using published Ares IRC-C2 parameters.
2. **Rare Class Performance under FedAvg:** `BruteForce` (39 test samples) and `WebAttack` (16 test samples) exhibit zero recall in the global federated neural network due to severe majority class imbalance ($8,196$ Normal samples).
3. **Algebraic Secure Aggregation:** Pairwise masking validates zero-sum cancellation ($\sum M_i = \mathbf{0}$) rather than a full SMPC cryptographic scheme.
4. **Hardware Environment:** Latency profiling was conducted on workstation CPU; low-power IoT microcontrollers will experience scaled execution times.
