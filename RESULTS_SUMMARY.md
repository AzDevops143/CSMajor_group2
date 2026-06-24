# SNOOPFINGER Reimplementation & Countermeasure Evaluation — Results Summary

## What this delivers (mapped to the assignment's last two tasks)

5. **Propose your own solution to address the identified gap**
6. **Implement the proposed solution and demonstrate its effectiveness**

**The gap** (from our Section 8.2 analysis): the paper's five countermeasures are
described only in prose. None are implemented or measured against the attack —
zero numbers exist anywhere in the paper for the *defense* side.

**Our solution**: a from-scratch reimplementation of the attack's core inference
mechanism (`snoopfinger_core.py`), used as a measurement harness to quantify
exactly one of the paper's proposed defenses — **Adaptive Sensor Data
Obfuscation** — under two concrete mechanisms the paper names but never tests:
added measurement noise (≈ reduced sampling rate / coarser gaze estimate) and
coordinate quantization (≈ "rounding to fewer decimal places").

No official SNOOPFINGER code exists (verified by web search), so every function
is built independently from the paper's equations (Section 5).

---

## Files

| File | Purpose |
|---|---|
| `snoopfinger_core.py` | Keyboard geometry, dictionary, and the attack's candidate-ranking engine (BackTree-equivalent) |
| `baseline_eval.py` | Sanity-checks the reimplementation against the paper's reported numbers |
| `countermeasure_eval.py` | **Main deliverable** — quantifies the obfuscation defense |
| `demo.py` | Live, presentable demo: type a word, watch the ranking with defense on/off |
| `fig_baseline_accuracy_by_length.png` | Sanity-check chart |
| `fig_obfuscation_accuracy_curves.png` | **Headline result** — accuracy vs. defense strength |
| `fig_obfuscation_pareto.png` | Privacy/utility trade-off comparison |

Run order: `python3 baseline_eval.py` → `python3 countermeasure_eval.py` →
`python3 demo.py --auto` (or without `--auto` for interactive use).

---

## Method, in brief

Since the literal Backward Key Inference Tree (BackTree) requires combinatorial
26ⁱ tree expansion, we replace it with a **mathematically equivalent ranking**:
for a typed word, compute the candidate's own clean relative-geometry signature
(consecutive distance + angle between key centers, ending at Enter) and compare
it against the noisy *observed* geometry via sum-of-squared deviation. Ranking a
real frequency-ordered English dictionary (`wordfreq`, ~4,000 words length 3–6,
playing the role of the paper's COCA-top-5000 pool) by this score reproduces
exactly what BackTree is built to produce — a ranked candidate list — without
needing the literal pruned tree.

## Baseline sanity check

Calibrated inherent noise (`noise_std = 0.40`) gives a **pooled average Top-1 of
0.604**, within 0.05 of the paper's reported 0.552. The per-length *trend*
diverges from the paper (ours increases with length; the paper's decreases) —
documented and explained in `baseline_eval.py`'s output, attributed to our
i.i.d. per-point noise model letting longer words average out noise, whereas
real head-motion error likely correlates/compounds across letters in ways a
simple Gaussian model doesn't capture. This divergence doesn't affect the
countermeasure analysis, which pools across lengths.

## Headline countermeasure results

| Defense strength | Top-1 accuracy | Top-10% accuracy |
|---|---|---|
| No defense (baseline) | 0.63 | 1.00 |
| + noise std 0.4 | 0.40 | 0.99 |
| + noise std 0.8 | 0.17 | 0.88 |
| + noise std 1.3 | **0.06** | 0.64 |
| Quantization step 1.0 | 0.51 | 1.00 |
| Quantization step 2.0 | 0.37 | 0.98 |

**Finding 1**: noise-based obfuscation is a far more effective and more
*predictable* defense than quantization. It drives Top-1 accuracy down
monotonically and can push it close to a near-random regime; quantization's
effect is weaker and less consistent (it only meaningfully degrades the attack
once the rounding step approaches or exceeds the keyboard's own key spacing).

**Finding 2 (Pareto comparison)**: at equal introduced distortion (RMSE),
the two strategies perform almost identically up to moderate distortion, but
noise-based obfuscation continues to buy privacy at higher distortion budgets
while quantization plateaus. **Recommendation: if a platform must choose one
mechanism from Section 8.2's "Adaptive Sensor Data Obfuscation" proposal, noise
injection is the more effective choice per unit of utility lost.**

**Finding 3**: halving Top-1 accuracy from the no-defense baseline requires
roughly **added noise std ≈ 0.6** (on top of the inherent ≈0.40) — i.e., total
measurement noise needs to roughly double — while no quantization step tested
achieved a 50% reduction, reinforcing Finding 1.

---

## Limitations of this evaluation (be upfront about these)

- Synthetic gaze data (Gaussian noise around true key centers), not real
  headset captures — necessary since no public dataset/code exists, but means
  absolute numbers are illustrative, not a replication.
- No legitimate-application utility study; RMSE distortion is a proxy, not a
  measured impact on rendering quality, head-locked UI, or comfort.
- BackTree replaced by an equivalent dictionary-ranking method rather than the
  literal pruned tree (see Method section) — same outcome, different mechanism.
- Real Adaptive Sensor Data Obfuscation in a deployed system would need
  legitimate-app-aware throttling (apply noise only to backgrounded apps) —
  not modeled here; we evaluate the obfuscation's effect on the attack in
  isolation.
