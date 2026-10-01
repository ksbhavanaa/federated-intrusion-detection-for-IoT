# PROJECT_AUDIT.md
# Federated Learning-Based Intrusion Detection System for IoT Networks
# Audit Date: 2026-09-03

---

## 1. Repository Overview

```
federated_ids/
├── data/              15 files (~830 MB raw CSVs + ~90 MB processed)
├── models/            4 files (semester 1 XGBoost .pkl, ~3.6 MB each)
├── notebooks/         EMPTY
├── results/           11 files (CSVs + PNGs)
├── src/
│   ├── models/        3 files (random_forest.py, xgboost_model.py, neural_network.py)
│   ├── nodes/         4 files (camera, sensor, router, smartlock)
│   ├── combine_datasets.py
│   ├── compare_models.py
│   ├── data_preprocessing.py
│   ├── federated_server.py
│   ├── final_comparison.py
│   ├── split_data.py
│   └── visualize_results.py
├── venv/
├── README.md          (outdated — only describes Phases 1 & 2)
└── .gitignore
```

---

## 2. Actual CICIDS2017 Label Values (Inspected 2026-09-03)

All raw CSVs were scanned. The following labels were found across the complete dataset:

| Raw Label                      | Count      | Mapped To    | Status      |
|-------------------------------|------------|--------------|-------------|
| BENIGN                        | 1,554,112  | Normal       | ✅ Included |
| DoS Hulk                      | 231,073    | DDoS         | ✅ Included |
| PortScan                      | 158,930    | PortScan     | ✅ Included |
| DDoS                          | 128,027    | DDoS         | ✅ Included |
| DoS GoldenEye                 | 10,293     | DDoS         | ✅ Included |
| FTP-Patator                   | 7,938      | BruteForce   | ✅ Included |
| SSH-Patator                   | 5,897      | BruteForce   | ✅ Included |
| DoS slowloris                 | 5,796      | DDoS         | ✅ Included |
| DoS Slowhttptest              | 5,499      | DDoS         | ✅ Included |
| Web Attack – Brute Force      | 1,507      | WebAttack    | ✅ Included |
| Web Attack – XSS              | 652        | WebAttack    | ✅ Included |
| Infiltration                  | 36         | Excluded     | ⚠️ Too few  |
| Web Attack – Sql Injection    | 21         | WebAttack    | ✅ Included |
| Heartbleed                    | 11         | Excluded     | ⚠️ Too few  |

### Label Mapping Decision

**Included classes (≥ 50 samples after sampling):**
- `Normal` ← BENIGN
- `DDoS` ← DoS Hulk, DDoS, DoS GoldenEye, DoS slowloris, DoS Slowhttptest
- `PortScan` ← PortScan
- `BruteForce` ← FTP-Patator, SSH-Patator
- `WebAttack` ← Web Attack – Brute Force, Web Attack – XSS, Web Attack – Sql Injection

**Excluded classes (< MIN_CLASS_SAMPLES=50):**
- `Infiltration` — only 36 samples in the full dataset, not representable after sampling
- `Heartbleed` — only 11 samples, excluded for the same reason

**Note:** `Bot` class does NOT exist in this CICIDS2017 dataset. It is NOT included.
The original plan listed Bot, but it is absent from all 7 raw CSVs.

**Final classes (5 classes):**
```
0 = Normal
1 = DDoS
2 = PortScan
3 = BruteForce
4 = WebAttack
```

---

## 3. Current Architecture (Semester 1)

```
Raw CSVs → data_preprocessing.py → clean_features.csv / clean_labels.csv
         → combine_datasets.py   → combined_features.csv / combined_labels.csv
         → split_data.py         → camera_data.csv / sensor_data.csv /
                                    router_data.csv / smartlock_data.csv

combined_features.csv → src/models/random_forest.py    → random_forest_results.csv
                      → src/models/xgboost_model.py    → xgboost_results.csv
                      → src/models/neural_network.py   → neural_network_results.csv

camera_data.csv    → src/nodes/camera_node.py    → camera_result.csv + camera_model.pkl
sensor_data.csv    → src/nodes/sensor_node.py    → sensor_result.csv + sensor_model.pkl
router_data.csv    → src/nodes/router_node.py    → router_result.csv + router_model.pkl
smartlock_data.csv → src/nodes/smartlock_node.py → smartlock_result.csv + smartlock_model.pkl

camera_result.csv + sensor_result.csv + router_result.csv + smartlock_result.csv
    → src/federated_server.py → federated_result.csv   [FAKE FEDAVG]
    → src/final_comparison.py → final_comparison.csv   [HARDCODED CENTRAL ACC]

final_comparison.csv → src/visualize_results.py → final_comparison_graph.png
```

---

## 4. Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| data_preprocessing.py | ⚠️ BROKEN | Hardcoded absolute path for different user |
| combine_datasets.py | ✅ Works | Relative paths, binary labels only |
| split_data.py | ✅ Works | IID split only |
| random_forest.py | ✅ Works | Relative paths, binary labels |
| xgboost_model.py | ✅ Works | Relative paths, binary labels |
| neural_network.py | ✅ Works | sklearn MLP, binary labels |
| camera_node.py | ✅ Works | XGBoost local training |
| sensor_node.py | ✅ Works | XGBoost local training |
| router_node.py | ✅ Works | XGBoost local training |
| smartlock_node.py | ✅ Works | XGBoost local training |
| federated_server.py | ❌ WRONG | Averages accuracy numbers, not model weights |
| final_comparison.py | ❌ WRONG | Hardcodes centralized accuracy = 0.99725 |
| visualize_results.py | ⚠️ Minimal | Only 29 lines, one basic bar chart |
| notebooks/ | ❌ EMPTY | No notebooks created yet |

---

## 5. Critical Issues Found

### CRITICAL (must fix)

1. **Hardcoded user path** — `data_preprocessing.py` line 5 contains:
   `r"C:\Users\admin\OneDrive\Desktop\federated_ids\data\Friday.csv"` 
   Will crash on any machine other than the original author's.

2. **Fake FedAvg** — `federated_server.py` computes:
   `global_accuracy = (cam + sen + rout + smart) / 4`
   This is NOT federated learning. It is metric averaging.
   True FedAvg must aggregate model parameters (weights), not scalar metrics.

3. **Hardcoded result** — `final_comparison.py` hardcodes:
   `centralized_accuracy = 0.99725`
   This is a fabricated result. Accuracy must be computed from model evaluation.

### HIGH (should fix)

4. **Binary-only labels** — `combine_datasets.py` uses `LabelEncoder.fit_transform` on all labels,
   which produces integer codes that obscure attack type information.
   Multi-class attack detection requires preserving raw label text before encoding.

5. **IID data split** — `split_data.py` splits combined data randomly and equally among 4 nodes.
   In reality, IoT devices see different traffic types. A camera is more likely to face DDoS;
   a smart lock is more likely to face BruteForce. Non-IID splits are more realistic.

6. **Per-node LabelEncoder** — `router_node.py` and `smartlock_node.py` each fit a new
   `LabelEncoder()` independently. In a multi-class scenario this would cause label class
   misalignment across nodes (class 0 on one node ≠ class 0 on another).

7. **No validation set** — all models use only train/test splits, no held-out validation set
   for hyperparameter selection or early stopping.

### MEDIUM (improve)

8. **No reproducibility control** — scaler is fit fresh each time, models retrain from scratch.
9. **No logging** — no structured output logs.
10. **compare_models.py and final_comparison.py overlap** — two separate files both generate
    comparison CSVs with different data/purposes; confusing.
11. **Notebooks empty** — no reproducible experiment record.

---

## 6. Semester 1 Results (Preserved at `results/semester1_baseline/`)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Random Forest | from CSV | from CSV | from CSV | from CSV |
| XGBoost | from CSV | from CSV | from CSV | from CSV |
| Neural Network (sklearn MLP) | from CSV | from CSV | from CSV | from CSV |
| Federated "Global" (FAKE) | ~99.43% | — | — | — |

> Note: The "Federated" result is not a true federated learning result.
> It is the average of 4 local XGBoost accuracies on their respective test sets.

---

## 7. Recommended Implementation Order (Semester 2)

1. `config.py` — central configuration
2. Inspect labels (done above)
3. `src/data/preprocessor.py` — fix preprocessing, multi-class, leakage-free
4. `src/data/client_splitter.py` — non-IID IoT splitting
5. Upgrade `src/models/` — multi-class, save reports
6. `src/federated/fed_model.py` — PyTorch MLP
7. `src/federated/fed_client.py` — real local training, returns state_dict
8. `src/federated/fed_server.py` — real FedAvg parameter aggregation
9. `src/evaluation/evaluator.py` — confusion matrices, reports
10. `src/visualize_results.py` — upgrade
11. `notebooks/` — 6 notebooks
12. `src/realtime/` — safe simulation
13. `src/dashboard/` — Flask + dark UI
14. `src/explainability/` — SHAP
15. `tests/` — unit tests
16. `run_pipeline.py` — end-to-end
17. `README.md` — full rewrite
18. `requirements.txt`
19. `FINAL_IMPLEMENTATION_REPORT.md`
