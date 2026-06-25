"""
create_dataset.py
==================
Generates simulated VR head/gaze tracking telemetry using Fitts' Law
and Minimum-Jerk trajectories.
"""

import math
import csv
from keyboard import KEYBOARD_LAYOUT

def minimum_jerk_trajectory(t, duration, p_start, p_end):
    """Standard minimum-jerk formula for natural human movement"""
    tau = min(t / duration, 1.0)
    scale = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)
    x = p_start[0] + (p_end[0] - p_start[0]) * scale
    y = p_start[1] + (p_end[1] - p_start[1]) * scale
    return x, y

def generate_fitts_movement_time(dist, width=1.0):
    """Fitts' law: MT = a + b * log2(2*D/W)"""
    a, b = 0.2, 0.15
    return a + b * math.log2(2 * max(dist, 0.1) / width)

def generate_telemetry_stream(word, sample_rate=72.0, rng=None):
    """
    Simulates a 72Hz gaze trajectory typing the word.
    Returns a list of dicts: [{'timestamp': t, 'x': x, 'y': y, 'is_dwell': bool, 'key': key}]
    and a list of dwell segments: [(start_idx, end_idx)]
    """
    if rng is None:
        import random
        rng = random.Random(42)
        
    dt = 1.0 / sample_rate
    t = 0.0
    pts = []
    segments = []
    
    # Start at keyboard center (spacebar or center key)
    cur_pos = KEYBOARD_LAYOUT.get('space', (4.5, -1.0))
    
    # Sequence of keys to press
    keys = list(word.lower()) + ['enter']
    
    for key in keys:
        if key not in KEYBOARD_LAYOUT:
            continue
        dest_pos = KEYBOARD_LAYOUT[key]
        dist = math.hypot(dest_pos[0] - cur_pos[0], dest_pos[1] - cur_pos[1])
        
        # Transition segment
        duration = generate_fitts_movement_time(dist)
        n_steps = max(int(duration / dt), 1)
        for i in range(n_steps):
            tx = (i / n_steps) * duration
            x, y = minimum_jerk_trajectory(tx, duration, cur_pos, dest_pos)
            # Add physiological tremor
            x += rng.gauss(0, 0.03)
            y += rng.gauss(0, 0.03)
            pts.append({
                'timestamp': t,
                'x': x,
                'y': y,
                'is_dwell': False,
                'key': key
            })
            t += dt
            
        # Dwell segment (lingers on target key)
        dwell_duration = rng.uniform(0.15, 0.30)
        n_dwell_steps = max(int(dwell_duration / dt), 3)
        start_idx = len(pts)
        for _ in range(n_dwell_steps):
            # Natural tremor around the exact key position
            x = dest_pos[0] + rng.gauss(0, 0.03)
            y = dest_pos[1] + rng.gauss(0, 0.03)
            pts.append({
                'timestamp': t,
                'x': x,
                'y': y,
                'is_dwell': True,
                'key': key
            })
            t += dt
        end_idx = len(pts)
        segments.append((start_idx, end_idx))
        
        cur_pos = dest_pos
        
    return pts, segments

def save_to_csv(pts, filename="vr_telemetry_dataset.csv"):
    """Saves the telemetry stream to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'x', 'y', 'is_dwell', 'key'])
        writer.writeheader()
        for pt in pts:
            writer.writerow(pt)
