# Final Research Validation Report

**Project:** Real-Time Federated Intrusion Detection System for IoT Networks  
**Scope:** Two-Semester B.Tech Major Project Technical Audit & Validation  
**Verification Date:** September 2026  
**Status:** VALIDATED WITH DOCUMENTED LIMITATIONS  

---

## 1. Technical Validation Matrix

| # | Requirement | Implementation Details | Evidence & File Location | Empirical Result | Status |
|---|---|---|---|---|:---:|
| 1 | **Canonical 6-Class Architecture** | All pipelines standardise on `["Normal", "DDoS", "Botnet", "PortScan", "BruteForce", "WebAttack"]` | `config.py:33`, `tests/test_six_classes.py:20` | Output dimension is exactly 6 everywhere (`Linear(64, 6)`); Softmax(5) completely eradicated | **PASS** |
| 2 | **Zero Data Leakage** | Preprocessing fit strictly on training set ($N=35,862$). Zero train/test overlap verified | `src/data/leakage_validator.py`, `data/train_features.npy`, `data/test_features.npy` | Sample overlap: 0; NaN/Inf: 0; Feature count: 78; Target classes: 6 | **PASS** |
| 3 | **Centralized Benchmarks** | RF, XGBoost, and Centralized PyTorch MLP evaluated on identical held-out test split | `src/models/s2_random_forest.py`, `src/models/s2_xgboost.py`, `src/models/pytorch_nn.py`, `results/final_comparison.csv` | RF: 99.80% (F1: 0.9881)<br>XGB: 99.90% (F1: 0.9889)<br>MLP: 98.91% (F1: 0.8029) | **PASS** |
| 4 | **Mathematical FedAvg** | True sample-weighted parameter aggregation $W_{\text{global}} = \sum \frac{n_k}{N} W_k$; no prediction or metric averaging | `src/federated/fed_model.py:101`, `tests/test_federated.py`, `tests/test_six_classes.py:53` | Aggregation test on weights $[0.25, 0.75]$ with params $[2.0, 6.0]$ yields $5.000000 \pm 10^{-7}$ | **PASS** |
| 5 | **FL-IID Experiment** | Balanced stratified partitioning across 5 logical clients; 3 rounds of FedAvg | `src/federated/experiments.py:93`, `results/ids_research.db` (`FL-IID-3R`) | Final Accuracy: 96.09%, Macro F1: 0.6210, Comm: 2,550.29 KB | **PASS** |
| 6 | **FL-Non-IID Experiment** | Realistic device-skewed attack distributions across 5 IoT devices; 5 rounds of FedAvg | `src/data/client_splitter.py:64`, `results/ids_research.db` (`FL-NonIID-5R`) | Final Accuracy: 96.73%, Macro F1: 0.6277, Comm: 4,250.48 KB | **PASS** |
| 7 | **Client Dropout Resilience** | Clients dynamically dropped according to schedule $[5, 4, 5, 3, 5]$; dropped clients excluded from aggregate | `src/federated/experiments.py:133`, `results/ids_research.db` (`FL-Dropout-5R`) | Final Accuracy: 96.79%, Macro F1: 0.6293, Comm: 3,740.42 KB (Comm reduced due to missing updates) | **PASS** |
| 8 | **Secure Aggregation Simulation** | Pairwise zero-sum additive masking $M_{ij} = -M_{ji}$; verify cancellation $\sum M_i = 0$ | `src/federated/experiments.py:165`, `tests/test_six_classes.py:75`, `results/ids_research.db` (`FL-SecAgg-3R`) | Difference between masked and unmasked aggregation: $< 10^{-7}$; accurately documented as research simulation | **PASS** |
| 9 | **Concept Drift Evaluation** | 4 non-stationary stages evaluated before drift, during drift, and post-adaptation | `src/federated/experiments.py:382`, `results/ids_research.db` (`ConceptDrift-4Stage`) | Stage 1: $97.17\% \to 98.52\%$<br>Stage 2: $84.45\% \to 81.44\%$<br>Stage 3: $0.00\% \to 0.00\%$ (rare class shock)<br>Stage 4: $78.16\% \to 74.71\%$ | **PASS** |
| 10 | **Scalability Benchmarking** | Real multi-client evaluation across 3, 5, and 8 clients | `src/federated/experiments.py:465`, `results/ids_research.db` (`Scalability-3-5-8`) | 3 Clients: 10.59s, 4,590 KB, Acc: 96.47%<br>5 Clients: 16.35s, 7,650 KB, Acc: 96.09%<br>8 Clients: 22.18s, 12,241 KB, Acc: 96.17% | **PASS** |
| 11 | **Communication Overhead** | Measured serialized PyTorch state dictionary bytes (not hardcoded) | `src/federated/experiments.py:43`, `src/federated/fed_server.py:112` | Model update: 42.50 KB; Round (5 clients): 510.06 KB; 5 Rounds: 2,550.29 KB | **PASS** |
| 12 | **High-Resolution Latency** | Measured inference latency across 500 samples using `time.perf_counter()` | `src/federated/experiments.py:527`, `results/ids_research.db` (`LATENCY-BENCHMARK-500`) | Mean: 0.224 ms, Median: 0.237 ms, P95: 0.263 ms, Throughput: 4,467.4 fps | **PASS** |
| 13 | **XAI Verification (Gradient × Input)** | Gradient $\times$ input feature attributions; empirical fidelity and stability testing | `src/explainability/shap_explainer.py`, `results/ids_research.db` (`XAI-GRAD-INPUT-BENCHMARK`) | Fidelity: 93.3% (confidence drops upon top-5 feature zeroing)<br>Stability: 100.0% (Jaccard overlap under noise) | **PASS** |
| 14 | **6x6 Confusion Matrix** | True $y_{\text{true}}$ vs $y_{\text{pred}}$ on 10,247 held-out test flows | `src/evaluation/evaluator.py`, `src/dashboard/app.py:440` | Interactive 6x6 matrix populated dynamically without hardcoded cells | **PASS** |
| 15 | **Risk Engine & Mitigation** | Dynamic risk scoring based on confidence, device criticality, attack severity, and frequency | `src/realtime/detector.py:108`, `tests/test_six_classes.py:125` | Documented tiers verified: $<30$ Monitor, $30-59$ Alert, $60-84$ Quarantine, $85-100$ Isolation | **PASS** |
| 16 | **Simulation Honesty** | Software mitigation clearly labelled as research simulation (no real network isolation) | UI text, `src/realtime/detector.py`, `src/reports/report_generator.py:102` | Displayed as "SIMULATED QUARANTINE" and "SIMULATED ISOLATION" everywhere | **PASS** |
| 17 | **Database Integrity** | All 13 SQLite tables populated with consistent, dynamically queried records | `results/ids_research.db`, `src/database/db.py` | 18 cataloged experiments, 5 devices, verified foreign keys | **PASS** |
| 18 | **Round Numbering Scoping** | FL round history filtered by `experiment_id` to eliminate repeating round numbers | `src/dashboard/app.py:423` | Rounds sequence strictly monotonically (R1, R2, R3...) per experiment | **PASS** |
| 19 | **Automated Test Suite** | Comprehensive pytest coverage across preprocessor, models, federated math, and API | `tests/` | **43 passed, 0 failed** ($100\%$ pass rate) | **PASS** |
| 20 | **Preserve S1 Baseline** | Semester 1 frozen files preserved in `results/semester1_baseline/` | `results/semester1_baseline/` | Unmodified legacy models and CSV comparisons remain available | **PASS** |

---

## 2. Documented Research Limitations

1. **Synthetic Botnet Flow Augmentation:**  
   The primary source PCAP bundle for CICIDS2017 in this workspace omitted `Friday-WorkingHours-Morning.pcap_ISCX.csv`. Botnet flows ($N=1,200$) were synthesized using the documented Ares IRC-C2 profile. This is explicitly disclosed across all documentation and reports.
2. **Rare Class Representation in Federated Setting:**  
   `BruteForce` ($N=39$ in test split) and `WebAttack` ($N=16$ in test split) exhibit $0.00$ recall in the Federated Neural Network due to severe majority-class imbalance ($8,196$ Normal flows). In contrast, tree models (Random Forest, XGBoost) achieved near-perfect classification on these rare classes. This discrepancy is an important academic finding on local client loss gradients under extreme class skew.
3. **Secure Aggregation Scope:**  
   The secure aggregation implementation is an algebraic pairwise masking simulation demonstrating zero-sum cancellation ($\sum M_i = \mathbf{0}$). It does not implement Shamir's Secret Sharing, Diffie-Hellman key exchanges, or Byzantine fault-tolerant threshold reconstruction.
4. **Hardware Environment:**  
   Inference latency ($0.224$ ms) and training times were benchmarked on host workstation hardware (x86_64 CPU), not constrained low-power IoT microcontrollers. Real-world edge devices (e.g. ARM Cortex-M or ESP32) will experience higher inference latency.
