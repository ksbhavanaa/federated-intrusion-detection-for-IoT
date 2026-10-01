# Comprehensive Academic Audit Report
# Project: Real-Time Federated Intrusion Detection System for IoT Networks
**Evaluation Target:** Full Repository Codebase, ML Pipeline, Data Partitions, Database, REST API, Web SOC Dashboard, and Documentation  
**Audit Date:** 2026-09-04  
**Auditor Role:** Senior ML Engineer, Federated Learning Researcher, IoT Cybersecurity Engineer & Academic Project Auditor  

---

## 1. Current Architecture
The repository is structured as a two-semester B.Tech Major Project platform comprising:
* **Preprocessing & Data Pipeline (`src/data/`):** Cleans CICIDS2017 raw CSV files, handles duplicate and NaN removal, applies label mapping, performs stratified sampling, scales continuous features via `StandardScaler`, and shards data across 5 IoT devices.
* **Centralized ML Benchmarks (`src/models/`):** Evaluates Random Forest (`s2_random_forest.py`), XGBoost (`s2_xgboost.py`), and a Centralized PyTorch Multi-Layer Perceptron (`pytorch_nn.py`).
* **Federated Learning Framework (`src/federated/`):** Implements a PyTorch `FederatedMLP` architecture, client-side training (`fed_client.py`), and sample-weighted Federated Averaging (`fed_server.py`, `experiments.py`).
* **Real-Time Edge Detection & Traffic Simulation (`src/realtime/`):** Generates statistical background and attack traffic streams (`traffic_simulator.py`) and evaluates flows using `IDSDetector` with a 4-tier risk engine (`detector.py`).
* **Explainable AI Engine (`src/explainability/`):** Computes input $\times$ gradient attribution approximating Shapley values for deep networks (`shap_explainer.py`).
* **Relational Database (`src/database/`):** SQLite database (`results/ids_research.db`) with WAL journaling enabled, containing 13 tables for devices, alerts, predictions, rounds, and experiment telemetry.
* **Web SOC Dashboard (`src/dashboard/`):** Flask application serving a dark-themed, responsive SOC user interface with REST API endpoints and ReportLab PDF report generation.

---

## 2. Current Dataset
* **Raw Source:** 7 CSV files from the Canadian Institute for Cybersecurity (UNB) CICIDS2017 dataset: `Monday.csv`, `Tuesday.csv`, `Wednesday.csv`, `Thursday1.csv`, `Thursday2.csv`, `Friday.csv`, `Friday1.csv`.
* **Raw Records:** ~2.64 million raw flow records.
* **Continuous Features:** 78 numerical network flow features (e.g., Flow Duration, Total Fwd/Bwd Packets, Flow IAT, Flags, Header Lengths).
* **Sampled Partitioning:** 50,032 samples sampled stratified (70% train = 35,022, 10% val = 5,003, 20% held-out test = 10,007).
* **Ground-Truth Availability Disclosure:**
  * Real classes in uploaded raw CSVs: `Normal` (BENIGN), `DDoS` (DoS Hulk, GoldenEye, slowloris, Slowhttptest, DDoS), `PortScan`, `BruteForce` (FTP-Patator, SSH-Patator), and `WebAttack` (Brute Force, XSS, SQL Injection).
  * `Botnet` (Ares Botnet): The raw `Friday-WorkingHours-Morning.pcap_ISCX.csv` was omitted from the local dataset archive. 

---

## 3. Current Labels & Classes
* **Intended Canonical Classes (6 Classes):**
  1. `Normal`
  2. `DDoS`
  3. `Botnet`
  4. `PortScan`
  5. `BruteForce`
  6. `WebAttack`
* **Current State in Files:**
  * `config.py` lists 6 classes in `CLASS_NAMES`.
  * However, `preprocessor.py` fit `LabelEncoder` strictly on classes present in the raw CSVs, resulting in only **5 classes** in `train_labels.npy` and `test_labels.npy` (`[0, 1, 2, 3, 4]`).
  * `FederatedMLP` currently has an output layer of `Linear(64, 5)` (Softmax 5) and centralized confusion matrices are 5x5.

---

## 4. Current Models
1. **Random Forest (Centralized):** `scikit-learn` `RandomForestClassifier(n_estimators=100)`.
2. **XGBoost (Centralized):** `xgboost` `XGBClassifier(n_estimators=100, max_depth=6)`.
3. **Centralized MLP:** PyTorch feedforward neural network (`78 -> 256 -> 128 -> 64 -> n_classes`).
4. **Federated MLP:** Identical PyTorch architecture trained on edge client shards and aggregated on the server.

---

## 5. Current Federated Learning Algorithm
* **Algorithm:** True sample-weighted Federated Averaging (FedAvg):
  $$W_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N} W_k$$
* **Weight Aggregation:** Mathematical parameter tensor averaging over PyTorch `state_dict` objects.
* **Communication Serialization:** Real byte measurement using `torch.save(state_dict, buf)`.

---

## 6. Current Experiment Types
1. **Centralized Benchmarks:** RF, XGBoost, and MLP on held-out test set.
2. **FL-IID:** 3-round stratified uniform partition across 5 clients.
3. **FL-Non-IID:** 5-round device-skewed partition across 5 IoT devices.
4. **Client Dropout:** 5-round simulation with dynamic client dropout and dynamic reweighting.
5. **Secure Aggregation Simulation:** Pairwise zero-sum noise masking ($M_{ij} = -M_{ji}$).
6. **Concept Drift:** 4-stage empirical drift benchmark (DDoS -> PortScan -> BruteForce -> Botnet).
7. **Scalability:** 3, 5, and 8 logical client scaling.
8. **Inference Latency:** High-resolution timing (mean, median, P95, min, max, throughput).
9. **XAI Fidelity & Stability:** Feature masking and Gaussian noise consistency.

---

## 7. Current Database Tables (SQLite: `results/ids_research.db`)
1. `users`: System operators and roles.
2. `devices`: 5 IoT device telemetry, risk scores, and mitigation states.
3. `predictions`: Inference logs, confidence, latency, and features.
4. `alerts`: Triggered threat alerts with severity, risk score, and mitigation state.
5. `federated_rounds`: Communication round telemetry (accuracy, macro F1, communication KB, loss).
6. `client_metrics`: Per-client sample counts, local loss, and aggregation weights.
7. `fl_experiments`: High-level federated run records.
8. `detection_latency`: Benchmarked latency percentiles and throughput.
9. `xai_results`: Stored SHAP explanations, top positive/negative features, fidelity, stability.
10. `mitigation_history`: Audit trail of software mitigation transitions.
11. `concept_drift_experiments`: Telemetry across 4 drift stages.
12. `scalability_experiments`: Telemetry across 3, 5, 8 clients.
13. `research_experiments`: Generic experiment registry.

---

## 8. Current API Routes (`src/dashboard/app.py`)
* `GET /`: Serves SOC dashboard.
* `GET /api/status`: System KPIs and fleet telemetry.
* `GET /api/dataset/status`: Dataset integrity and split metadata.
* `GET /api/devices`: Monitored IoT device fleet cards.
* `GET /api/alerts`: Recent alerts.
* `GET /api/predictions`: Recent inference logs.
* `POST /api/predict`: Live inference.
* `POST /api/simulate-attack-custom`: Attack Simulation Lab injection.
* `POST /api/simulation/start`: Start background traffic stream.
* `POST /api/simulation/stop`: Stop background traffic stream.
* `GET /api/explain/<alert_code>`: Local SHAP attribution for alert.
* `GET /api/xai/global`: Fleet-wide feature importance, fidelity, stability.
* `GET /api/federated/rounds`: Federated round telemetry.
* `POST /api/federated/run`: Trigger live FL run.
* `GET /api/experiments/comparison`: Centralized vs Federated comparison.
* `GET /api/experiments/concept-drift`: Concept drift telemetry.
* `POST /api/experiments/run-concept-drift`: Run concept drift benchmark.
* `GET /api/experiments/scalability`: Scalability telemetry.
* `POST /api/experiments/run-scalability`: Run scalability benchmark.
* `GET /api/experiments/history`: Research experiment registry list.
* `GET /api/analytics/latency`: Latency percentiles.
* `POST /api/experiments/run-full-evaluation`: Run full research evaluation suite.
* `GET /api/reports/pdf`: Generate and download ReportLab PDF report.
* `GET /api/reports/csv`: Export SQLite tables as zipped CSVs.

---

## 9. Current Dashboard Pages / Tabs
1. **Overview:** Fleet health, KPIs, traffic chart, attack breakdown.
2. **Live Threat IDS:** Interactive Attack Simulation Lab and real-time detection monitor.
3. **IoT Device Fleet:** Device cards with risk bars and mitigation states.
4. **Federated Learning:** FedAvg convergence curves and round table.
5. **Explainable AI (SHAP):** Local attribution breakdown and global importance chart.
6. **Security Alerts:** Incident response table with explain triggers.
7. **Analytics & Benchmarks:** Comparison table and performance charts.
8. **Research Experiments:** Experiment registry, concept drift, and scalability.

---

## 10. Technical Inconsistencies Identified
1. **5-Class vs 6-Class Pipeline Mismatch:**
   * While `config.py` and UI headings claim 6 classes (`Normal`, `DDoS`, `Botnet`, `PortScan`, `BruteForce`, `WebAttack`), the preprocessor generated 5 classes (`0..4`) because `Botnet` was missing in raw CSVs.
   * Consequently, the PyTorch model output dimension was `Linear(64, 5)`, confusion matrices were 5x5, and `Botnet` was absent from test metrics.
2. **Repeated Round Identifiers on Convergence Charts:**
   * `/api/federated/rounds` queried `ORDER BY id DESC LIMIT 10`, mixing rounds across separate runs (e.g. `R1, R2, R3, R1, R2, R3`).
3. **Static Telemetry in `/api/dataset/status`:**
   * Several sample counts and flags were hard-coded in the route rather than dynamically queried from preprocessed splits.
4. **Missing Direct 6x6 Confusion Matrix Endpoint:**
   * The dashboard lacked a dedicated endpoint returning the complete 6x6 confusion matrix with normalized percentages and per-class metrics.

---

## 11. Data Leakage Risks Identified
1. **Risk:** Scaler fitting on entire dataset prior to splitting.
   * *Status:* Already addressed in `preprocessor.py` (split before fit), but lacks a formal programmatic validator that verifies:
     * Zero train/test sample overlap.
     * `StandardScaler.mean_` is mathematically identical to $X_{\text{train}}.\text{mean}(0)$ and does not include $X_{\text{test}}$.
     * Zero target attribute leakage in feature space.

---

## 12. Metric Consistency Problems
1. **Macro F1 Calculation in Multiclass Imbalance:**
   * Macro F1 must be calculated strictly across all 6 canonical classes using `f1_score(y_true, y_pred, average='macro', zero_division=0)`.
2. **Centralized vs. Federated Baseline Comparison:**
   * All models must be evaluated on the exact same held-out test split (`test_features.npy`, `test_labels.npy`).
3. **Overview KPIs vs. Database Records:**
   * Overview metrics (Federated Accuracy, Macro F1, Latency, Active Threats) must strictly mirror the latest records in `ids_research.db`.

---

## 13. Missing Validation Items
1. Formal test asserting `model.n_classes == len(CLASS_NAMES) == 6`.
2. Formal mathematical unit test asserting weighted FedAvg math on synthetic dummy weights.
3. Standalone test for pairwise zero-sum masking cancellation ($\|W_{\text{masked}} - W_{\text{unmasked}}\|_\infty < 10^{-6}$).
4. Formal data leakage validation module.

---

## 14. Incorrect / Overextended Claims to Eliminate
1. **Claim:** "Dataset: CICIDS2017" without qualification.
   * *Correction:* Truthfully label as **"CICIDS2017-derived benchmark with calibrated synthetic augmentation (for Botnet class due to omitted raw Friday morning capture)"**.
2. **Claim:** "Physical device isolation / firewall blocking".
   * *Correction:* Truthfully label as **"SIMULATED QUARANTINE"** and **"SIMULATED ISOLATION"** (software-level mitigation simulation; no physical host firewall or VLAN modification).
3. **Claim:** "XGBoost Federated Averaging".
   * *Correction:* Explicitly document that Random Forest and XGBoost are centralized benchmarks, whereas PyTorch MLP is used for federated learning because its parameters can be mathematically aggregated via FedAvg.
4. **Claim:** "Cryptographic Secure Aggregation".
   * *Correction:* Truthfully label as **"Secure Aggregation Simulation using Pairwise Zero-Sum Masks"**.

---

## 15. Required Fixes
1. **Enforce Canonical 6-Class Pipeline:**
   * Integrate calibrated Botnet flow dynamics (empirical IRC-C2 profile) into the preprocessed dataset before train/val/test splitting.
   * Fit `LabelEncoder` strictly on `CLASS_NAMES = ["Normal", "DDoS", "Botnet", "PortScan", "BruteForce", "WebAttack"]`.
   * Set neural network output dimension to `Linear(64, 6)`.
   * Retrain Centralized RF, XGBoost, Centralized MLP, and Federated MLP on 6 classes.
2. **Data Leakage Validator:**
   * Create `src/data/leakage_validator.py` to programmatically verify zero sample overlap, train-only scaler fitting, and label integrity.
3. **Dynamic Dataset Status API:**
   * Make `/api/dataset/status` compute and return real counts dynamically from loaded numpy splits.
4. **Federated Round History Scoping:**
   * Update `/api/federated/rounds` to scope queries by distinct `experiment_id`, ensuring clean sequential `Round 1 .. Round 5` charts.
5. **Interactive 6x6 Confusion Matrix:**
   * Add `/api/analytics/confusion-matrix` and an interactive visualizer on the Analytics tab.
6. **Synchronize All Dashboard Tabs & Reports:**
   * Ensure Overview, Live IDS, Fleet, FL, XAI, Alerts, Analytics, and PDF/CSV reports strictly reflect real SQLite data.

---

## 16. Files That Will Be Modified / Added
* **Modified:**
  * `config.py`: Canonical 6-class declarations, dataset metadata.
  * `src/data/preprocessor.py`: Botnet calibration, 6-class encoding, leakage assertions.
  * `src/data/client_splitter.py`: 6-class non-IID and IID device partitioning.
  * `src/federated/fed_model.py`: 6-output MLP architecture (`Linear(64, 6)`).
  * `src/models/s2_random_forest.py`: 6-class Random Forest training & evaluation.
  * `src/models/s2_xgboost.py`: 6-class XGBoost training & evaluation.
  * `src/models/pytorch_nn.py`: 6-class Centralized MLP training & evaluation.
  * `src/federated/fed_server.py`: 6-class global model aggregation and persistence.
  * `src/federated/experiments.py`: Unified experiment engine with unique run IDs and 6-class evaluation.
  * `src/realtime/detector.py`: 6-class inference, updated risk engine, honest mitigation labels.
  * `src/realtime/traffic_simulator.py`: Centroid computation for all 6 classes.
  * `src/explainability/shap_explainer.py`: 6-class gradient attribution and XAI metrics.
  * `src/dashboard/app.py`: Dynamic APIs, confusion matrix endpoint, scoped FL rounds.
  * `src/dashboard/templates/index.html`: 6x6 confusion matrix UI, integrity badges, clean round labels.
  * `src/reports/report_generator.py`: Dynamic 6-class PDF research report generator.
  * `data/README.md`: Honest documentation of dataset provenance and synthetic Botnet augmentation.
  * `README.md`: Full academic documentation update.
* **Added:**
  * `AUDIT_REPORT.md`: This comprehensive audit document.
  * `src/data/leakage_validator.py`: Dedicated leakage validation module.
  * `tests/test_six_classes.py`: Comprehensive test suite verifying all 17 Phase 30 criteria.
  * `FINAL_VALIDATION_REPORT.md`: Final validation matrix with PASS/FAIL evidence.
  * `FINAL_RESULTS.md`: Actual experimental metrics table.
  * `PAPER_ALIGNMENT.md`: Formal alignment of academic claims vs. implementation reality.

---

## 17. Experiments That Must Be Rerun
1. **Data Preprocessing & Sharding:** Generate 6-class train, val, and held-out test splits.
2. **Centralized Benchmark 1:** Random Forest (6 classes).
3. **Centralized Benchmark 2:** XGBoost (6 classes).
4. **Centralized Benchmark 3:** Centralized PyTorch MLP (6 classes).
5. **Federated Experiment 1 (FL-IID):** 3 rounds across 5 clients.
6. **Federated Experiment 2 (FL-Non-IID):** 5 rounds across 5 IoT devices.
7. **Federated Experiment 3 (Client Dropout):** 5 rounds with simulated client disconnects.
8. **Federated Experiment 4 (Secure Aggregation):** Pairwise zero-sum mask simulation.
9. **Research Benchmark 1 (Concept Drift):** 4-stage empirical drift adaptation.
10. **Research Benchmark 2 (Scalability):** 3, 5, and 8 logical client benchmarks.
11. **Telemetry Benchmark (Inference Latency):** 500-sample high-resolution latency profiling.
12. **XAI Benchmark:** Real-sample Fidelity and Stability scoring.
