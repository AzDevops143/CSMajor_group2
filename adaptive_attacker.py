"""
adaptive_attacker.py
=====================
Attacker models targeting Virtual Keyboard gaze telemetry.
Implements NaiveAttacker (refined dwell-clustering segmentation)
and AdaptiveAttacker (uses ground-truth dwell boundaries).
"""

import math
from keyboard import KEYBOARD_LAYOUT

def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _rank_keys(x, y):
    """Return key labels sorted by distance — nearest first."""
    return [k for k, _ in sorted(KEYBOARD_LAYOUT.items(),
                                 key=lambda kv: _dist((x, y), kv[1]))]

class AdaptiveAttacker:
    """Uses dwell segment boundaries to average within each keypress.
    Passing true boundaries = upper-bound (best-case) adversary."""

    def get_centroids(self, pts, segments):
        out = []
        for (s, e) in segments:
            seg = pts[s:e]
            if not seg:
                continue
            cx = sum(p[0] for p in seg) / len(seg)  # Centroid X
            cy = sum(p[1] for p in seg) / len(seg)  # Centroid Y
            out.append((cx, cy))
        return out

    def ranked_per_key(self, pts, segments):
        out = []
        for (s, e) in segments:
            seg = pts[s:e]
            if not seg:
                continue
            cx = sum(p[0] for p in seg) / len(seg)  # Centroid X
            cy = sum(p[1] for p in seg) / len(seg)  # Centroid Y
            out.append(_rank_keys(cx, cy))
        return out

class NaiveAttacker:
    """Segmenter that doesn't know the ground-truth dwell boundaries.
    It groups points using a velocity/distance-based clustering of consecutive coordinates."""
    
    def __init__(self, threshold=0.15, min_dwell_len=5):
        self.threshold = threshold
        self.min_dwell_len = min_dwell_len

    def get_centroids(self, pts, L):
        if not pts:
            return []
            
        # Identify candidate dwell frames (low step distance)
        is_dwell_frame = [False] * len(pts)
        for i in range(len(pts) - 1):
            d = math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
            if d < self.threshold:
                is_dwell_frame[i] = True
        if len(pts) > 1:
            is_dwell_frame[-1] = is_dwell_frame[-2]
            
        # Group consecutive dwell frames
        dwell_clusters = []
        cur_cluster = []
        for i, pt in enumerate(pts):
            if is_dwell_frame[i]:
                cur_cluster.append((pt, i))
            else:
                if cur_cluster:
                    dwell_clusters.append(cur_cluster)
                    cur_cluster = []
        if cur_cluster:
            dwell_clusters.append(cur_cluster)
            
        # Filter clusters that are too short
        dwell_clusters = [c for c in dwell_clusters if len(c) >= 3]
        
        # If we have at least L+1 clusters, take the L+1 largest ones and sort by time
        if len(dwell_clusters) >= L + 1:
            dwell_clusters = sorted(dwell_clusters, key=len, reverse=True)[:L+1]
            dwell_clusters = sorted(dwell_clusters, key=lambda c: c[0][1])
            
        centroids = []
        for cluster in dwell_clusters:
            xs = [p[0][0] for p in cluster]
            ys = [p[0][1] for p in cluster]
            centroids.append((sum(xs)/len(xs), sum(ys)/len(ys)))
            
        # Fallback to temporal chunking if clustering fails to find exactly L+1 dwells
        if len(centroids) != L + 1:
            centroids = []
            chunk_size = len(pts) // (L + 1)
            for i in range(L + 1):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < L else len(pts)
                seg = pts[start:end]
                dwell_part = seg[len(seg)//2 :]
                if not dwell_part:
                    dwell_part = seg
                cx = sum(p[0] for p in dwell_part) / len(dwell_part)
                cy = sum(p[1] for p in dwell_part) / len(dwell_part)
                centroids.append((cx, cy))
                
        return centroids

    def ranked_per_key(self, pts, L):
        centroids = self.get_centroids(pts, L)
        out = []
        for (cx, cy) in centroids:
            out.append(_rank_keys(cx, cy))
        return out
