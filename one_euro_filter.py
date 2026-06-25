import time
import math

class OneEuroFilter:
    """
    First-order low-pass filter with an adaptive cutoff frequency.
    Maintains low latency during fast movements and high smoothing during slow movements/rest.
    """
    def __init__(self, min_cutoff=1.0, beta=0.005, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def _exponential_smoothing(self, alpha, x, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, x, t=None):
        """
        Filters the input value 'x' at timestamp 't' (seconds).
        If 't' is not provided, time.time() is used.
        """
        if t is None:
            t = time.time()

        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = 0.0
            return x

        t_e = t - self.t_prev
        if t_e <= 0.0:
            # Prevent division by zero if multiple samples arrive at the exact same timestamp
            return self.x_prev

        # 1. Filter the derivative (velocity)
        dx = (x - self.x_prev) / t_e
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._exponential_smoothing(alpha_d, dx, self.dx_prev)

        # 2. Compute the adaptive cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. Filter the signal value
        alpha = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(alpha, x, self.x_prev)

        # 4. Save state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
