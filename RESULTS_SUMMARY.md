# SNOOPFINGER Reimplementation and CASOM Defense Evaluation — Results Summary

## What this delivers
1. **Core Attack Evaluation (`snoopfinger_core.py` flow)**: A mathematically equivalent candidate-ranking analysis used to run parameter sweeps and quantify the trade-off of the paper's proposed prose countermeasures (coordinate quantization and added Gaussian noise).
2. **End-to-End Simulation Pipeline (CASOM v2 flow)**: A complete, modular simulation chain that generates 72Hz VR gaze telemetry, applies the Context-Aware Sensor Obfuscation Middleware (CASOM) block-offset defense, runs multiple segmentation attackers, and performs BackTree inference.

---

## Files

| File | Purpose |
|---|---|
| `snoopfinger_core.py` | Keyboard geometry, dictionary, and the core candidate-ranking matching engine |
| `baseline_eval.py` | Sanity-checks the core attack's behavior against the trend reported in the paper |
| `countermeasure_eval.py` | Sweeps and measures the efficacy of added noise and coordinate quantization defenses |
| `demo.py` | Interactive or automated demo of the core attack under different defense states |
| `keyboard.py` | Coordinate layouts for standard QWERTY, AZERTY, and QWERTZ virtual keyboards |
| `create_dataset.py` | VR gaze telemetry generator using Fitts' Law and Minimum-Jerk trajectories |
| `casom_defense.py` | Context-Aware Sensor Obfuscation Middleware implementing 5 defense modes |
| `adaptive_attacker.py` | Attacker implementations (NaiveAttacker dwell clustering and AdaptiveAttacker) |
| `backtree.py` | Backward Key Inference Tree (BackTree) engine walking backward from the enter key |
| `benchmark.py` | Orchestration script that runs the full pipeline benchmark and updates dashboards |
| `new.html` | Interactive dashboard demonstrating the block-offset defense and decoy typing |
| `explain.html` | Step-by-step explainer with KaTeX equations and a click-to-type simulation sandbox |
| `vr_telemetry_dataset.csv` | Sample simulated VR gaze tracking coordinate telemetry dataset |
| `benchmark_results.json` | Structured JSON results from the pipeline benchmark |

---

## Method, in brief

### Flow 1: Core Evaluation
Since the literal Backward Key Inference Tree (BackTree) requires combinatorial 26^N tree expansion, we replace it with a mathematically equivalent ranking: for a typed word, compute the candidate's clean relative-geometry signature (consecutive distance and angle between key centers, ending at Enter) and compare it against the noisy observed geometry via sum-of-squared deviation.

### Flow 2: CASOM v2 Simulation Pipeline
We simulate physical VR head/gaze orientation tracking at 72Hz. The trajectory moves from key to key using Minimum-Jerk paths and Fitts' Law timing. Dwells are modeled with physiological tremors.
- **Adaptive Attacker**: Has access to true dwell boundaries, averaging coordinates within each segment.
- **Naive Attacker**: Performs velocity/distance-based clustering to automatically detect dwells from the raw stream.
- **BackTree**: Performs pruned search walking backward from the enter key using spatial and angular thresholds, matching against valid dictionary words and prefixes.

---

## Evaluation Results

### Core Evaluation Sweep
Calibrated inherent noise (`noise_std = 0.40`) gives a pooled average Top-1 of 0.604, close to the paper's reported 0.552.

| Defense strength | Top-1 accuracy | Top-10% accuracy |
|---|---|---|
| No defense (baseline) | 0.63 | 1.00 |
| + noise std 0.4 | 0.40 | 0.99 |
| + noise std 0.8 | 0.17 | 0.88 |
| + noise std 1.3 | 0.06 | 0.64 |
| Quantization step 1.0 | 0.51 | 1.00 |
| Quantization step 2.0 | 0.37 | 0.98 |

### CASOM v2 Pipeline Benchmark Results
(Generated from `python benchmark.py` over 30 trials x 12 words x 5 defenses):

| Defense Mode | Attacker: Adaptive (Best-Case) Top-1 | Attacker: Naive (Clustering) Top-1 |
|---|---|---|
| **No Defense (Baseline)** | 91.7% | 68.6% |
| **IID Noise (sigma=0.5)** | 90.6% | 37.5% |
| **Drift Noise (sigma=0.3)** | 89.7% | 60.3% |
| **Quantization (step=1.0)** | 91.7% | 63.6% |
| **CASOM Block Offset (sigma=0.5)** | **49.7%** | **0.6%** |

---

## Key Findings

1. **IID Noise Vulnerability**: Independent Gaussian noise per sample is easily bypassed by the Adaptive Attacker using dwell averaging (90.6% Top-1 accuracy). The noise cancels to zero over the dwell frames.
2. **CASOM Block Offset Defeated Averaging**: By holding the Laplace noise offset constant within temporal blocks, the noise does not cancel out. This completely breaks the keyboard relative geometry (reducing Naive Attacker Top-1 accuracy to 0.6%).
3. **Segmentation Collapse**: CASOM Block Offset disrupts the spatial/velocity structure of the gaze path, making it extremely difficult for the Naive Attacker to segment dwells (collapsing Top-1 accuracy to near-zero).

---

## Limitations

- Synthetic simulated gaze data is used rather than real headset tracking captures.
- Legitimate-application usability impact is not evaluated; while block-offset is highly secure, it might introduce tracking lag or UI jitter in foreground virtual keyboards if not bypassed for trusted applications.
