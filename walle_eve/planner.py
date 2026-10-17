"""Information-gain planner.

Candidates are (target cell, scan radius). For each feasible candidate we estimate, by
Monte-Carlo with common random numbers, how much a scan there is expected to reduce the
entropy of the belief:

    gain = H(belief) - E_[hypothesis ~ belief, noise][ H(posterior) ]

then rank by  gain / (time_cost/T + battery_cost/B)  where cost includes travel. A candidate
is feasible only if, after travelling and scanning, WALL-E can still reach a charging
station with `reserve` battery to spare (the "turn back" rule) and time remains.
"""
import math

import numpy as np

from .grid import INF


class InfoPlanner:
    def __init__(self, belief, cfg, info, dist_station):
        self.b, self.cfg = belief, cfg
        self.a = cfg["agent"]
        self.res = cfg["resources"]
        self.dist_station = dist_station
        self.W, self.H = info["W"], info["H"]
        m = self.b.model
        acc = m.det_acc  # entropy left after a confirmed detection (status still uncertain)
        self.h_detect = -(acc * math.log(acc) + (1 - acc) * math.log(1 - acc))

    def rank(self, pos, battery, time_left, dist_robot, scanned_mask, scanned_positions, rng):
        a, c, b = self.a, self.res["costs"], self.b
        mb, mt = c["move"]["battery"], c["move"]["time"]
        reserve = self.res["reserve"]
        T, B = self.res["time_budget"], self.res["battery_capacity"]
        stride = int(a["candidate_stride"])

        cells = {(x, y) for y in range(0, self.H, stride) for x in range(0, self.W, stride)}
        cells |= {tuple(pos), b.map_cell()}
        cells |= {(x, y) for x, y, _ in b.top_modes(3)}

        # hypotheses and noise are drawn once and shared by every candidate (common random numbers)
        K = int(a["mc_samples"])
        idx = rng.choice(b.p.size, size=K, p=b.p.ravel())
        hs, hy, hx = np.unravel_index(idx, b.p.shape)
        z1, z2 = rng.normal(size=K), rng.normal(size=K)
        u_out, u_det = rng.random(K), rng.random(K)
        u_ang = rng.uniform(-math.pi, math.pi, K)
        H0 = b.entropy()
        m = b.model
        marg = b.marginal()

        out = []
        for (x, y) in cells:
            if b.walls[y, x] or dist_robot[y, x] >= INF:
                continue
            travel = int(dist_robot[y, x])
            prep = None
            for r in a["scan_radii"]:
                bc = travel * mb + c["scan"]["battery"] + c["scan"]["battery_per_radius"] * r
                tc = travel * mt + c["scan"]["time"]
                if bc > battery or tc > time_left:
                    continue
                if battery - bc < self.dist_station[y, x] * mb + reserve:
                    continue  # could not get home afterwards
                if prep is None:
                    prep = b.prep_signal(x, y)
                gain = self._gain(prep, x, y, r, hs, hy, hx, z1, z2, u_out, u_det, u_ang, H0, m)
                revisit = any(math.dist((x, y), p) <= 1.5 for p in scanned_positions)
                if revisit:
                    gain *= a["revisit_penalty"]
                # exploring unscanned ground may reveal clues, which the entropy gain cannot see
                area = (b.XX - x) ** 2 + (b.YY - y) ** 2 <= r * r
                novel = float((area & ~scanned_mask & b.free).sum()) / max(1, float((area & b.free).sum()))
                gain += a["clue_bonus"] * novel
                # the point of the mission: chance that this scan actually finds EVE
                gain += a["find_bonus"] * m.p_detect * float(marg[area].sum())
                cost = max(1e-3, tc / T + bc / B)
                out.append({"target": (x, y), "radius": r, "gain": float(gain), "cost": float(cost),
                            "score": float(gain / cost), "travel": travel,
                            "battery_cost": float(bc), "time_cost": float(tc)})
        out.sort(key=lambda d: -d["score"])
        return out

    def _gain(self, prep, x, y, r, hs, hy, hx, z1, z2, u_out, u_det, u_ang, H0, m):
        b = self.b
        sig_s, sig_b = m.eff(1.0)
        neg = b.lik_detect_neg(x, y, r)
        Hs = []
        for k in range(len(hs)):
            dx, dy = hx[k] - x, hy[k] - y
            d2 = dx * dx + dy * dy
            if d2 <= r * r and u_det[k] < m.p_detect:
                Hs.append(self.h_detect)  # EVE would be pinned down
                continue
            if u_out[k] < m.eps:  # a spoofed reading: random bearing, typical strength
                bearing, strength = u_ang[k], m.A * 0.1 * math.exp(z1[k])
            else:
                ls = math.log(m.A) + m.log_gain[hs[k]] - math.log1p(d2 / m.d0 ** 2) + z1[k] * sig_s
                bearing, strength = math.atan2(dy, dx) + z2[k] * sig_b, math.exp(ls)
            q = b.p * b.lik_signal(prep, bearing, strength, 1.0) * neg
            z = q.sum()
            Hs.append(H0 if z <= 1e-300 else b.entropy(q / z))
        return H0 - float(np.mean(Hs))
