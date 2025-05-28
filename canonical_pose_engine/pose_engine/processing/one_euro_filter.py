# canonical_pose_engine/pose_engine/processing/one_euro_filter.py
import numpy as np

class OneEuroFilter:
    """
    A vectorized One-Euro filter for smoothing signal data (like pose landmarks).
    Optimized for real-time performance.
    """
    def __init__(self, min_cutoff=0.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, te, cutoff):
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        te = t - self.t_prev

        # If time difference is too small, return previous value to avoid division by zero
        if te < 1e-6: # Add a small epsilon check
            return self.x_prev
        
        # Filter for the derivative
        alpha_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Filter for the value
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._smoothing_factor(te, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat