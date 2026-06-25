"""
casom_defense.py
=================
Context-Aware Sensor Obfuscation Middleware (CASOM).
Implements 5 sensor data obfuscation modes to protect VR/AR gaze tracking.
"""

import math
import random

class CASOM:
    """Context-Aware Sensor Obfuscation Middleware."""
    
    def __init__(self, mode="block", noise_scale=0.5, block_samples=15, quant_step=1.0, seed=42):
        self.mode = mode
        self.noise_scale = noise_scale
        self.block_samples = block_samples
        self.quant_step = quant_step
        self._rng = random.Random(seed)
        
    def _laplace(self, b):
        # Inverse-CDF sampling of Laplace(0, b)
        u = self._rng.random() - 0.5
        return -b * math.copysign(1, u) * math.log(1 - 2 * abs(u))
        
    def obfuscate(self, pts):
        if self.mode == "none":
            return list(pts)
        elif self.mode == "iid":
            return self._iid(pts)
        elif self.mode == "drift":
            return self._drift(pts)
        elif self.mode == "quant":
            return self._quant(pts)
        elif self.mode == "block":
            return self._block(pts)
        else:
            raise ValueError(f"Unknown defense mode: {self.mode}")
            
    def _iid(self, pts):
        # Independent noise per sample — DEFEATED by averaging!
        return [(x + self._laplace(self.noise_scale),
                 y + self._laplace(self.noise_scale)) for (x, y) in pts]
                 
    def _drift(self, pts):
        # Drift noise (random walk)
        out = []
        ox = oy = 0.0
        step_scale = self.noise_scale * 0.1
        for (x, y) in pts:
            ox += self._laplace(step_scale)
            oy += self._laplace(step_scale)
            out.append((x + ox, y + oy))
        return out
        
    def _quant(self, pts):
        # Coordinate quantization
        step = self.quant_step
        if step == 0.0:
            return list(pts)
        return [(round(x / step) * step, round(y / step) * step) for (x, y) in pts]
        
    def _block(self, pts):   # RECOMMENDED MODE
        out = []
        ox = oy = 0.0
        for i, (x, y) in enumerate(pts):
            if i % self.block_samples == 0:
                ox = self._laplace(self.noise_scale)  # Fresh offset!
                oy = self._laplace(self.noise_scale)
            out.append((x + ox, y + oy))  # Constant within block
        return out
