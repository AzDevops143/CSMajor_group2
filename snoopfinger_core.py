"""
snoopfinger_core.py
====================

A from-scratch reimplementation of the core SNOOPFINGER inference pipeline,
built directly from the equations and methodology described in:

    Lee, S. & Choi, W. "Eyes on Your Typing: Snooping Finger Motions on
    Virtual Keyboards." IEEE Symposium on Security and Privacy (S&P), 2025.

No official code release accompanies the paper (verified via web search),
so every function below is derived independently from the paper's text,
Section 5 (Attack Design) in particular. Where the paper's exact algorithm
involves combinatorial tree expansion that would be impractical to fully
replicate (the literal Backward Key Inference Tree, BackTree, expands a
branching candidate tree of depth = word length), we implement a
*geometrically equivalent* candidate-ranking method instead:

    Paper's BackTree (conceptually):
        - Root = Enter key (known anchor)
        - At each depth, score candidate keys by how well their relative
          distance/angle to the previous (already-resolved) point matches
          the distance/angle actually observed between consecutive gaze
          points.
        - Expand the tree, weight candidates by recurrence, rank resulting
          words by total weight.

    Our reimplementation (mathematically equivalent ranking, tractable):
        - For a candidate word w of the observed length, compute w's own
          "clean" geometry (the sequence of consecutive distances/angles
          between its letters' true key positions on the adversary's
          reconstructed keyboard, ending at Enter).
        - Compare that clean geometry against the *observed* (noisy) gaze
          geometry extracted from the victim's actual keypress sequence.
        - Score = sum of squared deviations in distance and (circularly
          wrapped) angle across all consecutive pairs.
        - Rank the full candidate dictionary by this score (ascending =
          best match first).

    This produces the same outcome BackTree is built to produce — a ranked
    list of candidate words from relative geometric similarity, with no
    training and no victim-specific data — without requiring the literal
    26^N branching tree. We rank against a real frequency-ordered English
    dictionary (via `wordfreq`) rather than the paper's COCA-top-5000 list,
    which is unavailable to us, but is conceptually the same kind of
    candidate pool the paper itself uses for Top-k% accuracy.

This module is the SHARED BASELINE used by:
    1. baseline_eval.py   -- sanity-checks the reimplementation against the
                             paper's qualitative trend (accuracy decreasing
                             with word length).
    2. countermeasure_eval.py -- quantifies the effect of Section 8.2's
                             "Adaptive Sensor Data Obfuscation" countermeasure,
                             which the original paper proposes but never
                             measures.
"""

import math
import random
from collections import defaultdict

import numpy as np
from wordfreq import top_n_list

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# 1. Keyboard layout
# ---------------------------------------------------------------------------
def build_qwerty_layout():
    """
    Build 2D key-center coordinates for a standard QWERTY virtual keyboard,
    consistent with the paper's stated assumption (Section 8.1, "Virtual
    Keyboard Layout"): standard QWERTY, Enter on the right, Spacebar at the
    bottom. Units are arbitrary "key-width" units; only relative geometry
    matters, since that is what the attack (and our reimplementation)
    actually exploits.
    """
    layout = {}
    row1 = "qwertyuiop"
    row2 = "asdfghjkl"
    row3 = "zxcvbnm"
    for i, c in enumerate(row1):
        layout[c] = (float(i), 0.0)
    for i, c in enumerate(row2):
        layout[c] = (float(i) + 0.5, 1.0)
    for i, c in enumerate(row3):
        layout[c] = (float(i) + 1.0, 2.0)
    layout["enter"] = (10.5, 0.5)   # right-hand side, per paper's description
    layout["space"] = (4.5, 3.0)    # bottom-center, per paper's description
    return layout


LAYOUT = build_qwerty_layout()


# ---------------------------------------------------------------------------
# 2. Candidate dictionary (adversary's / attacker's known word pool)
# ---------------------------------------------------------------------------
def build_dictionary(n=6000, lengths=(3, 4, 5, 6)):
    """
    Build a frequency-ranked English word pool, analogous in spirit to the
    paper's "top 5,000 most frequently used words in the Corpus of
    Contemporary American English (COCA)" (Section 6, User Recruitment and
    Typed Data). COCA itself isn't freely redistributable, so we use
    `wordfreq`'s frequency ranking over a general English corpus instead --
    same idea (a realistic, frequency-weighted candidate pool), different
    underlying corpus.
    """
    words = top_n_list("en", n)
    words = [w for w in words if w.isalpha() and len(w) in lengths]
    by_len = defaultdict(list)
    for w in words:
        by_len[len(w)].append(w)
    return dict(by_len)


# ---------------------------------------------------------------------------
# 3. Geometry helpers
# ---------------------------------------------------------------------------
def word_points(word):
    """Sequence of 2D key-center coordinates for `word`, ending at Enter --
    mirrors the paper's word-typing protocol (Section 6): type the word,
    then press Enter."""
    return np.array([LAYOUT[c] for c in word] + [LAYOUT["enter"]])


def consecutive_geometry(points):
    """
    Given a sequence of N+1 points, return the N consecutive (distance,
    angle) pairs -- the same relative quantities the paper computes between
    successive 2D gaze points (Section 5.6, "d" and "theta").
    """
    diffs = points[1:] - points[:-1]
    dist = np.linalg.norm(diffs, axis=1)
    ang = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    return dist, ang


def circular_diff_deg(a, b):
    """Smallest signed angular difference between a and b, in degrees,
    correctly wrapped (so e.g. 179 vs -179 reads as a 2-degree difference,
    not 358)."""
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 4. Precompute candidate geometry per word length (vectorized, cached)
# ---------------------------------------------------------------------------
def precompute_candidate_cache(dictionary):
    """
    For every word length present in `dictionary`, precompute the clean
    (noise-free) consecutive-distance and consecutive-angle arrays for every
    candidate word, vectorized with numpy. This is the adversary's
    reference geometry -- equivalent to the paper's "adversary's
    reconstructed virtual keyboard layout" step (Section 5.5), computed once
    and reused for every inference.
    """
    cache = {}
    for L, words in dictionary.items():
        n = len(words)
        pts = np.zeros((n, L + 1, 2))
        for i, w in enumerate(words):
            pts[i] = word_points(w)
        diffs = pts[:, 1:, :] - pts[:, :-1, :]
        dist = np.linalg.norm(diffs, axis=2)                       # (n, L)
        ang = np.degrees(np.arctan2(diffs[:, :, 1], diffs[:, :, 0]))  # (n, L)
        index = {w: i for i, w in enumerate(words)}
        cache[L] = {"words": words, "dist": dist, "ang": ang, "index": index}
    return cache


# ---------------------------------------------------------------------------
# 5. Victim-side simulation: turning a typed word into noisy "observed" gaze
# ---------------------------------------------------------------------------
def simulate_observed_points(word, noise_std=0.05, quant_step=None):
    """
    Simulate the *victim's* noisy, attacker-observed 2D gaze points for a
    typed word, mirroring the real measurement chain the paper describes:
    head orientation -> 3D gaze direction -> equirectangular projection ->
    normalized 2D gaze point (Sections 5.3-5.4). Every stage of that chain
    introduces some imprecision (sensor noise, projection error, normalization
    error), which we collapse into a single per-point Gaussian noise term
    (`noise_std`) for tractability -- this is the *baseline* measurement
    noise that exists even with no countermeasure deployed.

    `quant_step`, if set, additionally rounds each coordinate to the nearest
    multiple of `quant_step` -- a direct implementation of the paper's own
    suggested countermeasure (Section 8.2, "Adaptive Sensor Data
    Obfuscation"): "limiting detailed data precision (e.g., rounding to
    fewer decimal places)".
    """
    true_pts = word_points(word)
    noisy = true_pts + np.random.normal(0.0, noise_std, true_pts.shape)
    if quant_step:
        noisy = np.round(noisy / quant_step) * quant_step
    return true_pts, noisy


# ---------------------------------------------------------------------------
# 6. Attacker-side inference (BackTree-equivalent ranking)
# ---------------------------------------------------------------------------
def rank_candidates(word, cache, noise_std=0.05, quant_step=None):
    """
    Run the attack on a single typed `word`: simulate the victim's noisy
    gaze sequence, then rank every same-length dictionary candidate by how
    well its clean relative geometry matches the observed geometry.

    Returns:
        rank        -- 0-based rank of the TRUE word in the sorted output
                       (0 = perfect Top-1 inference)
        n_candidates -- size of the candidate pool at this word length
        true_pts, noisy_pts -- for optional visualization
    """
    L = len(word)
    sub = cache[L]
    true_pts, obs_pts = simulate_observed_points(word, noise_std, quant_step)
    obs_dist, obs_ang = consecutive_geometry(obs_pts)

    cand_dist = sub["dist"]      # (n, L)
    cand_ang = sub["ang"]        # (n, L)

    dist_err = (cand_dist - obs_dist[None, :]) ** 2
    ang_err = (circular_diff_deg(cand_ang, obs_ang[None, :]) / 180.0) ** 2
    score = dist_err.sum(axis=1) + ang_err.sum(axis=1)

    order = np.argsort(score)
    true_idx = sub["index"][word]
    rank = int(np.where(order == true_idx)[0][0])
    ranked_words = [sub["words"][i] for i in order]
    return rank, len(sub["words"]), true_pts, obs_pts, ranked_words


def topk_threshold(n_candidates, k_percent):
    """Number of top candidates corresponding to the paper's Top-k% metric
    (Section 6.1): 'the probability that the actual input word is found
    within the top k% of the candidate words.'"""
    return max(1, math.ceil(n_candidates * k_percent / 100.0))
