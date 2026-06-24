"""
countermeasure_eval.py
=======================

THIS IS THE MAIN DELIVERABLE for the assignment's gap-analysis -> proposed
solution -> implementation -> demonstration chain.

------------------------------------------------------------------------
THE GAP (identified from Section 8.2 of the paper)
------------------------------------------------------------------------
Section 8.2 ("Countermeasures") proposes five defenses in prose, including
"Adaptive Sensor Data Obfuscation": reduce the sampling frequency or
numerical precision of orientation data delivered to background apps. The
paper gives NO threshold, NO measurement, and NO experiment showing how much
obfuscation is actually needed to meaningfully degrade the attack, nor what
that costs in data fidelity. Every quantitative result in the paper
(Sections 7.1-7.5) is about the ATTACK; none are about the DEFENSE.

------------------------------------------------------------------------
OUR PROPOSED SOLUTION
------------------------------------------------------------------------
An empirical evaluation harness that:
  1. Applies the two obfuscation mechanisms the paper itself names --
     (a) reduced precision (coordinate quantization / rounding), and
     (b) reduced fidelity (added Gaussian noise, standing in for a lower
         effective sampling rate / coarser gaze estimate) --
     to the simulated victim gaze stream, at a sweep of strengths.
  2. Re-runs the SNOOPFINGER reimplementation (snoopfinger_core.py) under
     each obfuscation strength and measures the resulting Top-1 / Top-k%
     accuracy.
  3. Simultaneously measures a UTILITY-COST proxy at each strength -- the
     RMSE distortion the obfuscation introduces into the legitimate signal
     -- so the privacy/utility trade-off the paper only mentions narratively
     ("may decrease accuracy ... without compromising legitimate application
     functionality") can be read directly off a curve, with numbers.

This directly fills the gap: instead of "obfuscation might help," we get
"obfuscation of strength X reduces Top-1 accuracy from A to B, at a cost of
C units of introduced positional error" -- the exact missing quantification.
------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import snoopfinger_core as sf

random.seed(11)
np.random.seed(11)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
N_TRIALS = 40            # noisy draws per word per setting
N_WORDS_PER_LEN = 8       # test words sampled per length
LENGTHS = (3, 4, 5, 6)
KS = (1, 10)              # focus on Top-1 (paper's headline metric) and Top-10%

# "Defense OFF" operating point: the inherent measurement noise that exists
# even with no deliberate obfuscation (sensor + projection + normalization
# error chain, Sections 5.3-5.4). Calibrated in baseline_eval.py so pooled
# Top-1 lands close to the paper's reported 0.552 average.
INHERENT_NOISE = 0.40

# Sweep 1: Adaptive Sensor Data Obfuscation via ADDED NOISE
#          (models reduced sampling rate / coarser gaze estimation -- the
#          defense intentionally injects additional measurement uncertainty
#          on top of what already exists)
ADDED_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3]

# Sweep 2: Adaptive Sensor Data Obfuscation via COORDINATE QUANTIZATION
#          (models "rounding to fewer decimal places", i.e. reduced
#          numerical precision, at the paper's own suggestion)
QUANT_STEPS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_test_words(dictionary, n_per_len, lengths=LENGTHS):
    out = []
    for L in lengths:
        out += random.sample(dictionary[L], min(n_per_len, len(dictionary[L])))
    return out


def evaluate_pooled(words, cache, noise_std, quant_step=None, n_trials=N_TRIALS, ks=KS):
    """Pooled (length-independent) accuracy across all `words`, plus the
    mean RMSE distortion the obfuscation introduces into the legitimate
    (true) signal -- our utility-cost proxy."""
    acc = {k: [] for k in ks}
    distortions = []
    for w in words:
        n_cand = len(cache[len(w)]["words"])
        ranks = []
        for _ in range(n_trials):
            rank, _, true_pts, obs_pts, _ = sf.rank_candidates(w, cache, noise_std, quant_step)
            ranks.append(rank)
            distortions.append(float(np.sqrt(np.mean((true_pts - obs_pts) ** 2))))
        ranks = np.array(ranks)
        for k in ks:
            if k == 1:
                acc[k].append(float(np.mean(ranks == 0)))
            else:
                thresh = sf.topk_threshold(n_cand, k)
                acc[k].append(float(np.mean(ranks < thresh)))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["distortion_rmse"] = float(np.mean(distortions))
    return out


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    print("Building dictionary and candidate-geometry cache ...")
    dictionary = sf.build_dictionary(n=6000)
    cache = sf.precompute_candidate_cache(dictionary)
    test_words = sample_test_words(dictionary, N_WORDS_PER_LEN)
    print(f"Pooled test set ({len(test_words)} words): {test_words}\n")

    # ---------------- Sweep 1: added-noise obfuscation ----------------
    print("=== Sweep 1: Adaptive obfuscation via added measurement noise ===")
    noise_results = []
    for added in ADDED_NOISE_LEVELS:
        total_noise = float(np.sqrt(INHERENT_NOISE ** 2 + added ** 2))
        res = evaluate_pooled(test_words, cache, total_noise, quant_step=None)
        res["added_noise"] = added
        res["total_noise"] = total_noise
        noise_results.append(res)
        print(f"  added_noise={added:.2f} (total={total_noise:.2f}): "
              f"Top-1={res[1]:.3f}  Top-10%={res[10]:.3f}  "
              f"distortion_rmse={res['distortion_rmse']:.3f}")

    # ---------------- Sweep 2: quantization obfuscation ----------------
    print("\n=== Sweep 2: Adaptive obfuscation via coordinate quantization ===")
    quant_results = []
    for q in QUANT_STEPS:
        qstep = q if q > 0 else None
        res = evaluate_pooled(test_words, cache, INHERENT_NOISE, quant_step=qstep)
        res["quant_step"] = q
        quant_results.append(res)
        print(f"  quant_step={q:.2f}: "
              f"Top-1={res[1]:.3f}  Top-10%={res[10]:.3f}  "
              f"distortion_rmse={res['distortion_rmse']:.3f}")

    # ================= PLOT 1: accuracy vs obfuscation strength =================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ADDED_NOISE_LEVELS, [r[1] for r in noise_results], "o-",
            color="#C77F12", label="Top-1")
    ax.plot(ADDED_NOISE_LEVELS, [r[10] for r in noise_results], "s--",
            color="#2DA8A0", label="Top-10%")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Added obfuscation noise (std, key-width units)")
    ax.set_ylabel("Attack accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Defense 1: Noise-based obfuscation\n(simulates reduced sampling rate / coarser gaze estimate)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(QUANT_STEPS, [r[1] for r in quant_results], "o-",
            color="#C77F12", label="Top-1")
    ax.plot(QUANT_STEPS, [r[10] for r in quant_results], "s--",
            color="#2DA8A0", label="Top-10%")
    ax.set_xlabel("Quantization step (key-width units)")
    ax.set_ylabel("Attack accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Defense 2: Precision-reduction obfuscation\n(\"rounding to fewer decimal places\", paper's own suggestion)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Quantifying Section 8.2's \"Adaptive Sensor Data Obfuscation\" -- "
                  "the missing experiment", fontsize=12)
    fig.tight_layout()
    fig.savefig("./fig_obfuscation_accuracy_curves.png", dpi=150)
    print("\nSaved fig_obfuscation_accuracy_curves.png")

    # ================= PLOT 2: privacy/utility Pareto comparison =================
    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
    ax2.plot([r["distortion_rmse"] for r in noise_results], [r[1] for r in noise_results],
              "o-", color="#C77F12", label="Noise-based obfuscation (Top-1)")
    ax2.plot([r["distortion_rmse"] for r in quant_results], [r[1] for r in quant_results],
              "s--", color="#21295C", label="Quantization obfuscation (Top-1)")
    ax2.set_xlabel("Utility-cost proxy: RMSE distortion introduced into legitimate signal")
    ax2.set_ylabel("Attack Top-1 accuracy")
    ax2.set_title("Privacy/utility trade-off:\nwhich obfuscation strategy buys more privacy per unit distortion?")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("./fig_obfuscation_pareto.png", dpi=150)
    print("Saved fig_obfuscation_pareto.png")

    # ================= Headline numbers for the report =================
    def first_below(results, key, threshold, xs, xkey):
        for r, x in zip(results, xs):
            if r[key] <= threshold:
                return r[xkey]
        return None

    noise_for_half = first_below(noise_results, 1, 0.5 * noise_results[0][1], ADDED_NOISE_LEVELS, "added_noise")
    quant_for_half = first_below(quant_results, 1, 0.5 * quant_results[0][1], QUANT_STEPS, "quant_step")

    print("\n=== HEADLINE RESULTS ===")
    print(f"No-defense Top-1 (pooled):                  {noise_results[0][1]:.3f}")
    print(f"Added noise needed to halve Top-1 accuracy:  {noise_for_half}")
    print(f"Quantization step needed to halve Top-1:     {quant_for_half}")
    print(f"Top-1 at strongest noise tested ({ADDED_NOISE_LEVELS[-1]}):       {noise_results[-1][1]:.3f}")
    print(f"Top-1 at strongest quantization tested ({QUANT_STEPS[-1]}):  {quant_results[-1][1]:.3f}")


if __name__ == "__main__":
    main()
