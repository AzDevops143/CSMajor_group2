"""
demo.py
========

Interactive, presentation-ready demonstration of:
  1. The SNOOPFINGER reimplementation attacking a chosen word with NO
     defense active.
  2. The same attack, with the "Adaptive Sensor Data Obfuscation"
     countermeasure switched on, showing the ranking degrade in real time.

Run interactively:
    python3 demo.py

Or non-interactively with a preset script of words (used for the slide
deck's "live demonstration" walkthrough):
    python3 demo.py --auto
"""

import sys
import argparse
import numpy as np

import snoopfinger_core as sf

np.random.seed()  # let demo runs vary naturally, unlike the eval scripts


def print_ranking(word, cache, noise_std, quant_step, top=8, label=""):
    rank, n_cand, true_pts, obs_pts, ranked = sf.rank_candidates(
        word, cache, noise_std, quant_step
    )
    print(f"\n{'-'*60}")
    print(f"{label}")
    print(f"  Typed word (ground truth, hidden from the 'adversary'): {word!r}")
    print(f"  Candidate pool size at this length:                     {n_cand}")
    print(f"  True word's rank in attacker's output (0 = Top-1):      {rank}")
    print(f"  Attacker's Top-{top} guesses:")
    for i, w in enumerate(ranked[:top]):
        marker = "  <-- TRUE WORD" if w == word else ""
        print(f"    {i+1}. {w}{marker}")
    if rank >= top:
        print(f"    ...")
        print(f"    {rank+1}. {word}  <-- TRUE WORD (outside Top-{top})")


def run_demo(word, cache):
    print(f"\n{'='*60}\nDEMO: typing \"{word}\"\n{'='*60}")

    print_ranking(
        word, cache, noise_std=0.40, quant_step=None, top=8,
        label="STEP 1 -- No defense (inherent measurement noise only)"
    )

    print_ranking(
        word, cache, noise_std=0.85, quant_step=None, top=8,
        label="STEP 2 -- Defense ON: noise-based obfuscation (added std=0.75)"
    )

    print_ranking(
        word, cache, noise_std=0.40, quant_step=1.5, top=8,
        label="STEP 3 -- Defense ON: precision-reduction obfuscation (quant_step=1.5)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                         help="run a preset sequence of words non-interactively")
    args = parser.parse_args()

    print("Building dictionary and candidate-geometry cache ...")
    dictionary = sf.build_dictionary(n=6000)
    cache = sf.precompute_candidate_cache(dictionary)
    print("Ready.\n")

    if args.auto:
        for w in ["fox", "code", "share", "police"]:
            run_demo(w, cache)
        return

    print("Type a 3-6 letter English word to attack (or 'quit' to exit).")
    while True:
        word = input("\n> word to type: ").strip().lower()
        if word in ("quit", "exit", "q"):
            break
        if not word.isalpha() or not (3 <= len(word) <= 6):
            print("Please enter a 3-6 letter alphabetic word.")
            continue
        if len(word) not in cache or word not in cache[len(word)]["index"]:
            print(f"'{word}' is not in the candidate dictionary for length "
                  f"{len(word)} -- try a more common word.")
            continue
        run_demo(word, cache)


if __name__ == "__main__":
    main()
