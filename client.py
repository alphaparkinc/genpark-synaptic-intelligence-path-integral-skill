"""
Synaptic Intelligence Path Integral Skill Client
Pure Python Standard Library implementation of Synaptic Intelligence (Zenke, Poole & Ganguli).
Accumulates running parameter trajectories and gradients online via path integrals,
estimating synaptic importance omega_k to penalize catastrophic disruption without caching old datasets.
"""

from typing import List, Dict, Any, Tuple, Optional


class SynapticIntelligenceTracker:
    def __init__(self, c_reg: float = 1.0, xi: float = 0.1):
        self.c_reg = c_reg  # Regularization strength
        self.xi = xi        # Damping parameter
        self.omega: Dict[str, float] = {}             # Consolidated parameter importances
        self.running_w: Dict[str, float] = {}         # Running path integral sum
        self.reference_theta: Dict[str, float] = {}    # Initial parameters for current task
        self.previous_theta: Dict[str, float] = {}     # Immediately preceding step parameters

    def start_task(self, initial_weights: Dict[str, float]):
        self.reference_theta = dict(initial_weights)
        self.previous_theta = dict(initial_weights)
        self.running_w = {k: 0.0 for k in initial_weights}

    def update_step(self, current_weights: Dict[str, float], step_gradients: Dict[str, float]):
        """Accumulate small parameter changes multiplied by gradient: Delta w_k -= g_k * Delta theta_k."""
        for k, w_curr in current_weights.items():
            delta_theta = w_curr - self.previous_theta.get(k, w_curr)
            grad = step_gradients.get(k, 0.0)
            # Path integral contribution
            self.running_w[k] = self.running_w.get(k, 0.0) - grad * delta_theta
            self.previous_theta[k] = w_curr

    def finalize_task(self, final_weights: Dict[str, float]):
        """Compute Omega_k = sum_tasks w_k / (Delta_k^2 + xi)."""
        for k, w_fin in final_weights.items():
            delta_total = w_fin - self.reference_theta.get(k, w_fin)
            denom = (delta_total ** 2) + self.xi
            omega_k = max(0.0, self.running_w.get(k, 0.0)) / denom
            self.omega[k] = self.omega.get(k, 0.0) + omega_k

    def compute_regularization(self, current_weights: Dict[str, float], task_optima: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute SI quadratic surrogate loss: c * sum_k Omega_k * (theta_k - theta_k^*)^2."""
        loss = 0.0
        grads = {}
        for k, w in current_weights.items():
            opt_w = task_optima.get(k, w)
            om = self.omega.get(k, 0.0)
            diff = w - opt_w
            loss += self.c_reg * om * (diff ** 2)
            grads[k] = 2.0 * self.c_reg * om * diff
        return loss, grads
