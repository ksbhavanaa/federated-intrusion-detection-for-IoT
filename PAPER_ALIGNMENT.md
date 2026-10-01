# Academic Paper Alignment & Claim Audit

**Project:** Real-Time Federated Intrusion Detection System for IoT Networks  
**Authoritative Scope:** Two-Semester Major Project Thesis & Research Paper Alignment  
**Auditor:** Senior ML / Cybersecurity Research Auditor  
**Verification Date:** September 2026  

This document cross-examines the typical academic claims made in the project thesis and presentation against the **concrete implementation evidence** in the codebase. All claims are classified into one of four rigor tiers:
- **VERIFIED:** Formally implemented, executed, and backed by reproducible code/tests.
- **VERIFIED WITH LIMITATION:** Implemented and functional, but bounded by research simulation constraints or synthetic augmentation that must be explicitly acknowledged.
- **NOT IMPLEMENTED:** Claimed in conceptual literature but absent from actual execution.
- **INCORRECT / MUST REMOVE:** Technically misleading, invalid, or theoretically impossible.

---

## 1. Quick-Reference Defense & Thesis Alignment Matrix

| Project Dimension | Academic Specification | Exact Implementation Evidence | Integrity Status |
| :--- | :--- | :--- | :---: |
| **Centralized Baseline** | Centralized comparison | Random Forest + XGBoost + PyTorch MLP trained on identical train/val sets and evaluated on identical test set | **VERIFIED** |
| **Federated Learning** | Continuous parameter aggregation | FederatedMLP (`Linear(78->256->128->64->6)`) with mathematical sample-weighted FedAvg | **VERIFIED** |
| **Dataset Provenance** | Benchmark dataset | CICIDS2017-derived ($N=50,032$) with 5 native PCAP classes + 1 calibrated synthetic Botnet profile | **VERIFIED WITH LIMITATION** |
| **Explainable AI (XAI)**| Feature attribution method | Gradient × Input first-order Taylor series attribution ($A_i = x_i \cdot \frac{\partial y_c}{\partial x_i}$); not KernelSHAP/TreeSHAP | **VERIFIED WITH LIMITATION** |
| **Privacy Guarantees** | Formal privacy claims | Raw training data remains decentralized/localized on simulated devices; no formal $(\epsilon, \delta)$-DP or cryptographic guarantee | **VERIFIED** |
| **Secure Aggregation** | Multi-party computation | Zero-sum pairwise additive masking simulation ($\sum M_i = \mathbf{0}$) demonstrating error $<10^{-7}$; algebraic simulation only | **VERIFIED WITH LIMITATION** |
| **Autonomous Mitigation**| Automated network response | Software decision engine (<30 Monitor, 30-59 Alert, 60-84 Quarantine, 85-100 Isolation); simulation only (no real firewall/ACL changes) | **VERIFIED WITH LIMITATION** |
| **Detection Latency** | Line-rate performance | 500-sample high-resolution benchmark on host workstation (x86_64 CPU); mean 0.224 ms; not physical IoT hardware | **VERIFIED WITH LIMITATION** |

---

## 2. Detailed Paper Claims vs. Empirical Reality

| # | Paper Claim | Implementation Reality & Evidence | Classification | Required Paper Wording |
|---|---|---|:---:|---|
| 1 | **"The IDS detects six distinct cyber attack classes from the CICIDS2017 benchmark."** | Monday–Friday raw CSVs contain 5 classes: Normal, DDoS, PortScan, BruteForce, WebAttack. Botnet was omitted in the source folder and was augmented via 1,200 calibrated IRC-C2 flows before pre-splitting. Output layer is `Linear(64, 6)`. | **VERIFIED WITH LIMITATION** | *"The IDS evaluates six network flow classes: five extracted directly from the published CICIDS2017 benchmark PCAP captures, complemented by a calibrated synthetic Botnet profile derived from published IRC-C2 communication patterns."* |
| 2 | **"Client models are aggregated using mathematical Federated Averaging (FedAvg)."** | Implemented in `src/federated/fed_model.py` (`fedavg`) and `src/federated/fed_server.py`. Takes local client PyTorch weights $W_k$ and weights them strictly by sample proportions $n_k / N$. | **VERIFIED** | *"Client model updates are aggregated using the canonical sample-weighted Federated Averaging (FedAvg) algorithm proposed by McMahan et al. (2017), operating directly on PyTorch model parameters."* |
| 3 | **"Tree models (XGBoost / Random Forest) are trained in the federated network."** | RF and XGBoost are evaluated strictly as centralized benchmarks in `src/models/`. Only the parameterizable `FederatedMLP` is used in federated rounds. | **VERIFIED WITH LIMITATION** | *"Random Forest and XGBoost serve as centralized performance baselines. The federated learning experiments utilize a parameterized multi-layer perceptron (MLP) suitable for continuous gradient-based parameter aggregation."* |
| 4 | **"Raw IoT telemetry remains private because raw data is never uploaded to the central server."** | In `fed_client.py` and `fed_server.py`, clients compute local parameter updates and transmit only model weight dictionaries. Raw flow vectors remain on client local disk. | **VERIFIED** | *"Privacy is preserved in accordance with the standard federated paradigm: raw IoT traffic data remains localized on simulated edge nodes, with only model parameter updates transmitted to the aggregation server."* |
| 5 | **"The system provides cryptographically secure aggregation."** | Implemented as a pairwise zero-sum additive masking simulation ($M_{ij} = -M_{ji}$) in `src/federated/experiments.py`. Demonstrates algebraic mask cancellation ($\sum M_i = 0$), but lacks Diffie-Hellman key exchange, threshold secret sharing, or Byzantine fault tolerance. | **VERIFIED WITH LIMITATION** | *"Secure aggregation is demonstrated via an algebraic zero-sum masking simulation that verifies exact cancellation of perturbations across client updates without compromising global convergence. It is not a production-grade cryptographic SMPC protocol."* |
| 6 | **"Non-IID data distribution across IoT devices is modeled and evaluated."** | 5 distinct IoT client profiles are sharded: Camera (DDoS-heavy), Router (PortScan-heavy), Sensor (Botnet-heavy), Door Lock (BruteForce-heavy), Smart Lighting (WebAttack-heavy). | **VERIFIED** | *"Non-IID data heterogeneity is explicitly modeled across five logical IoT device categories, each exhibiting skewed attack class distributions reflecting real-world device compromise patterns."* |
| 7 | **"The IDS delivers sub-millisecond real-time detection latency."** | High-resolution CPU timing (`time.perf_counter()`) on 500 test samples measured mean latency of 0.224 ms (4,467.4 flows/sec throughput). | **VERIFIED WITH LIMITATION** | *"Single-flow inference latency benchmarks on host x86 hardware yielded a mean detection time of 0.224 ms (P95: 0.263 ms). Embedded low-power microcontroller deployments would experience proportional execution scaling."* |
| 8 | **"An Explainable AI (XAI) engine explains IDS alerts with mathematical fidelity."** | Implemented via gradient $\times$ input first-order Taylor attribution. Measured Fidelity is 93.3% (dropping top-5 features lowers target class confidence) and Stability is 100.0% (Jaccard consistency under noise). | **VERIFIED** | *"Alert explainability is provided through gradient-based input attribution, with empirical verification showing 93.3% fidelity in target-class confidence degradation upon feature masking."* |
| 9 | **"Attacked IoT devices are immediately quarantined and isolated from the network."** | Software mitigation policy evaluates risk scores into 4 tiers (<30 Monitor, 30–59 Alert, 60–84 Quarantine, 85–100 Isolation). However, no physical network routing changes, iptables commands, or VLAN isolations are executed. | **VERIFIED WITH LIMITATION** | *"Autonomous risk-based mitigation operates as a software-level decision simulation, categorizing threats into Monitor, Alert, Simulated Quarantine, and Simulated Isolation states without modifying production networking infrastructure."* |
| 10 | **"The IDS is validated on physical IoT hardware testbeds."** | All 5 IoT devices are logical partitions running as concurrent threads/processes on a simulated environment. | **NOT IMPLEMENTED** | *"The multi-device architecture operates as an IoT fleet simulation; physical deployment on embedded microcontrollers remains a recommended direction for future engineering."* |
| 11 | **"The federated system guarantees differential privacy."** | No Gaussian or Laplacian noise mechanism, gradient clipping, or Rényi DP accounting is implemented in the client training loop. | **INCORRECT / MUST REMOVE** | *Remove all claims of Differential Privacy ($(\epsilon, \delta)$-DP). Describe privacy purely in terms of raw data localization and zero-sum masking simulation.* |
| 12 | **"Energy consumption and battery drain on IoT nodes was measured."** | No energy profiler (e.g. RAPL, Monsoon power monitor) was used or recorded. | **INCORRECT / MUST REMOVE** | *Remove claims regarding empirical energy efficiency or milliwatt-hour measurements.* |

---

## 2. Summary of Academic Posture

The project is academically defensible as a **rigorous research simulation and benchmark platform**. To maintain the highest research integrity during viva/defense:
1. Emphasize that **XGBoost and Random Forest** serve as centralized comparison baselines because their discrete tree splits cannot be aggregated via McMahan's continuous FedAvg.
2. Openly discuss the **severe class imbalance** in CICIDS2017: while tree models memorized rare classes (`BruteForce` with 39 samples and `WebAttack` with 16 samples), the Federated Neural Network prioritized the dominant classes (`Normal`, `DDoS`, `Botnet`, `PortScan`), yielding an overall accuracy of 96.17% but lower Macro F1 (0.6217). This provides an authentic, publishable research discussion on loss weighting in federated optimization.
3. Accurately position **Secure Aggregation** as an algebraic simulation demonstrating zero-sum mask cancellation rather than a fully deployed cryptographic protocol.
