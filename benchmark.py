"""
benchmark.py
=============
Automated trial orchestrator that runs the simulated pipeline:
- Generates 72Hz VR head-tracking telemetry using Fitts' Law and Minimum-Jerk.
- Applies 5 CASOM defense modes.
- Evaluates against NaiveAttacker and AdaptiveAttacker.
- Performs word reconstruction using BackTree candidate ranking.
- Reports Top-1, Top-3, and Top-5 accuracy.
"""

import random
from keyboard import KEYBOARD_LAYOUT, set_active_layout
from create_dataset import generate_telemetry_stream
from casom_defense import CASOM
from adaptive_attacker import NaiveAttacker, AdaptiveAttacker
from backtree import infer_word_candidates
from wordfreq import top_n_list

# Configuration
WORDS = ["fox", "the", "quick", "brown", "jumps", "over", "lazy", "dog", "code", "work", "game", "text"]
TRIALS = 30
SAMPLE_RATE = 72.0
SEED = 42

def build_dictionary(n=6000, lengths=(3, 4, 5, 6)):
    words = top_n_list("en", n)
    words = [w for w in words if w.isalpha() and len(w) in lengths]
    return words

def main():
    random.seed(SEED)
    rng = random.Random(SEED)
    
    print("Building dictionary and prefix cache...")
    dictionary = build_dictionary(n=6000)
    prefixes = set()
    for w in dictionary:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    print(f"Dictionary loaded with {len(dictionary)} words.")
    
    # Define defenses to test
    defenses_config = {
        "No Defense (Baseline)": {"mode": "none"},
        "IID Noise (sigma=0.5)": {"mode": "iid", "noise_scale": 0.5},
        "Drift Noise (sigma=0.3)": {"mode": "drift", "noise_scale": 0.3},
        "Quantization (step=1.0)": {"mode": "quant", "quant_step": 1.0},
        "CASOM Block Offset (sigma=0.5)": {"mode": "block", "noise_scale": 0.5, "block_samples": 15}
    }
    
    # Initialize attackers
    naive_attacker = NaiveAttacker(threshold=0.6, min_dwell_len=5)
    adaptive_attacker = AdaptiveAttacker()
    
    print(f"\nRunning benchmark: {TRIALS} trials x {len(WORDS)} words x {len(defenses_config)} defenses...")
    
    results = {}
    
    for defense_name, config in defenses_config.items():
        print(f"\nEvaluating: {defense_name}...")
        
        casom = CASOM(
            mode=config.get("mode"),
            noise_scale=config.get("noise_scale", 0.5),
            quant_step=config.get("quant_step", 1.0),
            block_samples=config.get("block_samples", 15),
            seed=SEED
        )
        
        naive_top1 = 0
        naive_top3 = 0
        naive_top5 = 0
        
        adaptive_top1 = 0
        adaptive_top3 = 0
        adaptive_top5 = 0
        
        total_trials = 0
        
        for word in WORDS:
            for trial in range(TRIALS):
                # Generate clean telemetry stream and dwell segments
                pts_dicts, segments = generate_telemetry_stream(word, sample_rate=SAMPLE_RATE, rng=rng)
                
                # Extract clean coordinates
                clean_pts = [(pt['x'], pt['y']) for pt in pts_dicts]
                
                # Apply defense obfuscation
                obfuscated_pts = casom.obfuscate(clean_pts)
                
                # 1. Evaluate NaiveAttacker
                naive_centroids = naive_attacker.get_centroids(obfuscated_pts, len(word))
                naive_candidates = infer_word_candidates(naive_centroids, dictionary, prefixes)
                
                if naive_candidates:
                    if naive_candidates[0] == word:
                        naive_top1 += 1
                    if word in naive_candidates[:3]:
                        naive_top3 += 1
                    if word in naive_candidates[:5]:
                        naive_top5 += 1
                        
                # 2. Evaluate AdaptiveAttacker (uses true segments)
                adaptive_centroids = adaptive_attacker.get_centroids(obfuscated_pts, segments)
                adaptive_candidates = infer_word_candidates(adaptive_centroids, dictionary, prefixes)
                
                if adaptive_candidates:
                    if adaptive_candidates[0] == word:
                        adaptive_top1 += 1
                    if word in adaptive_candidates[:3]:
                        adaptive_top3 += 1
                    if word in adaptive_candidates[:5]:
                        adaptive_top5 += 1
                        
                total_trials += 1
                
        results[defense_name] = {
            "naive": {
                "top1": naive_top1 / total_trials,
                "top3": naive_top3 / total_trials,
                "top5": naive_top5 / total_trials
            },
            "adaptive": {
                "top1": adaptive_top1 / total_trials,
                "top3": adaptive_top3 / total_trials,
                "top5": adaptive_top5 / total_trials
            }
        }
        
    # Print results
    print("\n" + "="*80)
    print("                      SNOOPFINGER PIPELINE BENCHMARK RESULTS")
    print("="*80)
    
    print("\n--- ADAPTIVE ATTACKER (Best-Case / Dwell-Aware) ---")
    print(f"{'Defense Mode':<35} | {'Top-1':<8} | {'Top-3':<8} | {'Top-5':<8}")
    print("-"*70)
    for def_name, res in results.items():
        print(f"{def_name:<35} | {res['adaptive']['top1']:<8.3f} | {res['adaptive']['top3']:<8.3f} | {res['adaptive']['top5']:<8.3f}")
        
    print("\n--- NAIVE ATTACKER (Distance-Threshold Clustering) ---")
    print(f"{'Defense Mode':<35} | {'Top-1':<8} | {'Top-3':<8} | {'Top-5':<8}")
    print("-"*70)
    for def_name, res in results.items():
        print(f"{def_name:<35} | {res['naive']['top1']:<8.3f} | {res['naive']['top3']:<8.3f} | {res['naive']['top5']:<8.3f}")
        
    print("="*80)

if __name__ == "__main__":
    main()
