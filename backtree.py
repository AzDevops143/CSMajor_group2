"""
backtree.py
============
Backward Key Inference Tree (BackTree) attack engine.
Walks backward from the 'enter' key to reconstruct typed word candidates
using relative coordinate distance and angle.
"""

import math
from keyboard import KEYBOARD_LAYOUT

def _d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _ang(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def _ang_diff(a, b):
    d = a - b
    return abs((d + 180.0) % 360.0 - 180.0)

def infer_word_candidates(gaze_points, dictionary, prefixes=None,
                          max_per_depth=12, top_words=50):
    """
    Finds and ranks candidate words matching the observed gaze sequence.
    Walks backward from 'enter' and ranks using geometric fit error.
    """
    n_letters = len(gaze_points) - 1  # Exclude 'enter'
    if n_letters <= 0:
        return []

    # Simple suffix pruning during backward search:
    # A reverse path is [char_at_N-1, char_at_N-2, ..., char_at_0]
    paths = [[]]

    for depth in range(n_letters):
        gaze_to = gaze_points[len(gaze_points) - 1 - depth]
        gaze_from = gaze_points[len(gaze_points) - 2 - depth]
        measured_d = _d(gaze_to, gaze_from)
        measured_θ = _ang(gaze_to, gaze_from)

        new_paths = []
        for path in paths:
            if not path:
                ref_pos = KEYBOARD_LAYOUT["enter"]
            else:
                ref_pos = KEYBOARD_LAYOUT[path[-1]]

            cands = []
            for k, pos in KEYBOARD_LAYOUT.items():
                if k in ('enter', 'space'):
                    continue
                d = _d(ref_pos, pos)
                ang = _ang(ref_pos, pos)
                # Standard tolerance threshold
                if abs(d - measured_d) <= 1.8 and _ang_diff(ang, measured_θ) <= 50.0:
                    cands.append(k)

            for k in cands:
                new_path = path + [k]
                
                # Suffix/prefix pruning check if prefixes set is provided
                if prefixes is not None:
                    partial_word = "".join(reversed(new_path))
                    # Check if this could be a suffix of any dictionary word
                    # (which means reversed partial_word is a prefix of reversed dictionary words)
                    # For simplicity, we can do a quick check if partial_word matches the end of any dictionary word
                    possible = False
                    for w in dictionary:
                        if len(w) == n_letters and w.endswith(partial_word):
                            possible = True
                            break
                    if not possible:
                        continue
                
                new_paths.append(new_path)

        # Prune to keep only the best matches geometrically
        def path_error(p):
            err = 0.0
            full_p = ["enter"] + p
            for idx in range(len(full_p) - 1):
                kp_to = KEYBOARD_LAYOUT[full_p[idx]]
                kp_from = KEYBOARD_LAYOUT[full_p[idx+1]]
                g_to = gaze_points[len(gaze_points) - 1 - idx]
                g_from = gaze_points[len(gaze_points) - 2 - idx]
                err += abs(_d(kp_to, kp_from) - _d(g_to, g_from))
                err += 0.05 * _ang_diff(_ang(kp_to, kp_from), _ang(g_to, g_from))
            return err

        new_paths = sorted(new_paths, key=path_error)[:max_per_depth]
        paths = new_paths

    # Reconstruct candidate words
    candidate_words = []
    for path in paths:
        word = "".join(reversed(path))
        if word in dictionary:
            candidate_words.append(word)

    # Fallback to dictionary words of the same length if none found
    if not candidate_words:
        candidate_words = [w for w in dictionary if len(w) == n_letters]

    # Geometric fit ranking helper
    def fit_error(word):
        keys = list(word) + ["enter"]
        err = 0.0
        for i in range(len(keys) - 1):
            kp_a = KEYBOARD_LAYOUT[keys[i]]
            kp_b = KEYBOARD_LAYOUT[keys[i + 1]]
            g_a, g_b = gaze_points[i], gaze_points[i + 1]
            err += abs(_d(kp_a, kp_b) - _d(g_a, g_b))
            err += 0.05 * _ang_diff(_ang(kp_a, kp_b), _ang(g_a, g_b))
        return err

    ranked = sorted(list(set(candidate_words)), key=fit_error)
    return ranked[:top_words]
