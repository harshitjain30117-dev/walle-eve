"""Belief state: a joint distribution over EVE's status and grid cell.

Every observation is turned into a likelihood array of shape (S, H, W) or (1, H, W) and
multiplied into the belief (Bayes rule). The signal likelihood is a mixture:
    (1 - eps) * lognormal(strength) * von-Mises(bearing)  +  eps * uniform-outlier
so spoofed readings (decoys) are down-weighted instead of dragging the estimate away.
"""
import math

import numpy as np

STATUSES = ("active", "dormant")
S = len(STATUSES)


class SensorModel:
    """What the agent assumes about its sensors (taken from the mission config)."""

    def __init__(self, cfg):
        s, c = cfg["sensors"], cfg["clues"]
        self.A, self.d0 = s["amplitude"], s["d0"]
        self.sig_s, self.sig_b, self.share = s["sigma_logstrength"], s["sigma_bearing"], s["bias_share"]
        self.log_gain = np.log(np.array([s["status_gain"][k] for k in STATUSES]))
        self.eps = cfg["agent"]["assumed_outlier_prob"]
        self.passive_mult, self.initial_mult = s["passive_noise_mult"], s["initial_noise_mult"]
        self.sig_track, self.sig_debris, self.log_acc = c["tracks_sigma"], c["debris_sigma"], c["log_accuracy"]
        self.p_detect, self.det_acc = s["p_detect"], s["detect_status_accuracy"]

    def eff(self, mult):
        """Effective (strength, bearing) sigma for a reading with noise multiplier `mult`."""
        f = math.sqrt((1 - self.share) * mult ** 2 + self.share)
        return self.sig_s * f, self.sig_b * f


def _vm(delta, kappa):
    """von Mises density."""
    return np.exp(kappa * np.cos(delta)) / (2 * np.pi * np.i0(kappa))


class Belief:
    def __init__(self, walls, status_prior, model):
        self.walls, self.model = walls, model
        self.H, self.W = walls.shape
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self.XX, self.YY = xx.astype(float), yy.astype(float)
        self.free = ~walls
        p = np.zeros((S, self.H, self.W))
        for i, s in enumerate(STATUSES):
            p[i][self.free] = status_prior[s]
        self.p = p / p.sum()
        self.rejected = 0  # updates skipped because they contradicted everything

    # ---- summaries
    def marginal(self):
        return self.p.sum(0)

    def entropy(self, p=None):
        p = self.p if p is None else p
        return float(-(p * np.log(p + 1e-30)).sum())

    def map_cell(self):
        m = self.marginal()
        y, x = np.unravel_index(np.argmax(m), m.shape)
        return int(x), int(y)

    def confidence(self, radius=2.0):
        """Posterior mass within `radius` of the most likely cell."""
        cx, cy = self.map_cell()
        mask = (self.XX - cx) ** 2 + (self.YY - cy) ** 2 <= radius ** 2
        return float(self.marginal()[mask].sum())

    def status_probs(self):
        s = self.p.sum((1, 2))
        return {k: float(v) for k, v in zip(STATUSES, s)}

    def top_modes(self, k=3, nms=4.0):
        """Up to k separated peaks of the position marginal: [(x, y, mass_within_nms)]."""
        m = self.marginal().copy()
        out = []
        for _ in range(k):
            y, x = np.unravel_index(np.argmax(m), m.shape)
            if m[y, x] <= 0:
                break
            near = (self.XX - x) ** 2 + (self.YY - y) ** 2 <= nms ** 2
            out.append((int(x), int(y), float(m[near].sum())))
            m[near] = 0
        return out

    # ---- likelihoods
    def prep_signal(self, cx, cy):
        """Candidate-position terms of the signal model (reusable across hypothetical readings)."""
        dx, dy = self.XX - cx, self.YY - cy
        base_ls = math.log(self.model.A) - np.log1p((dx * dx + dy * dy) / self.model.d0 ** 2)
        return base_ls, np.arctan2(dy, dx)

    def lik_signal(self, prep, bearing, strength, mult):
        """Likelihood of a reading given the robot cell described by `prep`."""
        m = self.model
        sig_s, sig_b = m.eff(mult)
        base_ls, ang = prep
        z = (math.log(strength) - base_ls[None] - m.log_gain[:, None, None]) / sig_s
        ls_pdf = np.exp(-0.5 * z * z) / (sig_s * math.sqrt(2 * math.pi))
        b_pdf = _vm(bearing - ang, 1.0 / sig_b ** 2)
        outlier = (1 / (2 * math.pi)) * (1 / (1.5 * math.sqrt(2 * math.pi)))
        return (1 - m.eps) * ls_pdf * b_pdf[None] + m.eps * outlier

    def lik_clue(self, clue):
        m, (cx, cy), pl = self.model, clue["pos"], clue["payload"]
        if clue["type"] == "tracks":
            ang = np.arctan2(self.YY - cy, self.XX - cx)
            L = 0.95 * _vm(math.radians(pl["bearing_deg"]) - ang, 1.0 / m.sig_track ** 2) + 0.05 / (2 * math.pi)
            return L[None]
        if clue["type"] == "debris":
            d = np.hypot(self.XX - cx, self.YY - cy)
            L = 0.95 * np.exp(-0.5 * ((pl["distance"] - d) / m.sig_debris) ** 2) / (
                m.sig_debris * math.sqrt(2 * math.pi)) + 0.05 / (self.W + self.H)
            return L[None]
        L = np.full((S, 1, 1), (1 - m.log_acc) / (S - 1))   # "log": status report
        L[STATUSES.index(pl["status"])] = m.log_acc
        return L

    def lik_detect_neg(self, cx, cy, r):
        inside = (self.XX - cx) ** 2 + (self.YY - cy) ** 2 <= r * r
        return np.where(inside, 1 - self.model.p_detect, 1.0)[None]

    def lik_detect_pos(self, x, y, status):
        L = np.zeros((S, self.H, self.W))
        acc = self.model.det_acc
        for i, s in enumerate(STATUSES):
            L[i, y, x] = acc if s == status else 1 - acc
        return L

    # ---- Bayes update
    def update(self, lik):
        q = self.p * lik
        z = q.sum()
        if not np.isfinite(z) or z <= 1e-300:
            self.rejected += 1
            return False
        q /= z
        q[:, self.free] = np.maximum(q[:, self.free], 1e-12)  # never let a cell die completely
        q[:, ~self.free] = 0
        self.p = q / q.sum()
        return True
