"""Simulated environment.

The world (walls, EVE, clues, decoys) is generated from the config + seed. The agent only
sees what `step()` returns. Hidden truth is available through `truth()` for evaluation and
visualisation only.

Observation model
-----------------
* Signal reading  : bearing + strength of the strongest source. Strength decays as
                    A * gain[status] / (1 + (d/d0)^2); both channels get Gaussian noise, part
                    of which is a fixed per-cell bias (multipath), so re-scanning the same cell
                    does not average the error away.
* Spoofed signals : with probability `decoy_prob` a reading comes from a decoy emitter.
* Scan(r)         : low-noise reading, reveals clues within r, detects EVE within r (p_detect).
* Move            : cheap, noisier passive reading, reveals clues within 1 cell.
"""
import math

import numpy as np

from .grid import INF, bfs

STATUSES = ("active", "dormant")


class WallEEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        seed = int(cfg["seed"])
        self.seed = seed
        wrng = np.random.default_rng(seed)
        self.obs_rng = np.random.default_rng(seed + 1_000_003)
        m = cfg["map"]
        self.W, self.H = int(m["width"]), int(m["height"])
        self.base = tuple(m["base"])
        self.stations = [self.base] + [tuple(s) for s in m["stations"]]
        self._build_map(wrng)
        self._place_entities(wrng)

        res = cfg["resources"]
        self.capacity = float(res["battery_capacity"])
        self.time_budget = float(res["time_budget"])
        self.battery = self.capacity
        self.time_used = 0.0
        self.pos = self.base
        self._bias = {}
        self.revealed = set()
        self.detected = False
        self._init_signal()

    # ------------------------------------------------------------------ world generation
    def _build_map(self, rng):
        walls = rng.random((self.H, self.W)) < self.cfg["map"]["obstacle_density"]
        for sx, sy in self.stations:
            walls[max(0, sy - 1):sy + 2, max(0, sx - 1):sx + 2] = False
        dist, _ = bfs(walls, self.base)
        for sx, sy in self.stations[1:]:
            if dist[sy, sx] >= INF:  # carve an L-shaped corridor if a station is walled in
                bx, by = self.base
                walls[by, min(bx, sx):max(bx, sx) + 1] = False
                walls[min(by, sy):max(by, sy) + 1, sx] = False
                dist, _ = bfs(walls, self.base)
        walls |= dist >= INF  # unreachable pockets become walls
        self.walls = walls
        self.dist_base = dist

    def _place_entities(self, rng):
        W, H, cfg = self.W, self.H, self.cfg
        free = [(x, y) for y in range(H) for x in range(W) if not self.walls[y, x]]
        e = cfg["eve"]
        if e["pos"] is not None:
            self.eve_pos = tuple(e["pos"])
        else:
            far = [c for c in free if math.dist(c, self.base) >= 0.4 * W]
            self.eve_pos = far[rng.integers(len(far))]
        if e["status"] is not None:
            self.eve_status = e["status"]
        else:
            pr = np.array([e["status_prior"][s] for s in STATUSES], float)
            self.eve_status = STATUSES[rng.choice(2, p=pr / pr.sum())]

        # clues: half placed near EVE (informative), the rest anywhere
        cc = cfg["clues"]
        names = list(cc["types"])
        probs = np.array([cc["types"][n] for n in names], float)
        probs /= probs.sum()
        taken = {self.eve_pos, self.base, *self.stations}
        near = [c for c in free if c not in taken and math.dist(c, self.eve_pos) <= cc["near_eve_radius"]]
        anyc = [c for c in free if c not in taken]
        n = int(cc["count"])
        n_near = int(round(cc["near_eve_fraction"] * n))
        self.clues = []
        for i in range(n):
            pool = near if i < n_near and near else anyc
            pos = pool[rng.integers(len(pool))]
            while pos in taken:
                pos = pool[rng.integers(len(pool))]
            taken.add(pos)
            kind = names[rng.choice(len(names), p=probs)]
            self.clues.append({"id": i, "type": kind, "pos": pos,
                               "payload": self._make_payload(kind, pos, rng)})

        sn = cfg["sensors"]
        cand = [c for c in free if math.dist(c, self.eve_pos) >= 8 and c not in taken]
        k = min(int(sn["decoy_count"]), len(cand))
        self.decoys = [cand[i] for i in rng.choice(len(cand), size=k, replace=False)] if k else []

    def _make_payload(self, kind, pos, rng):
        cc = self.cfg["clues"]
        dx, dy = self.eve_pos[0] - pos[0], self.eve_pos[1] - pos[1]
        if kind == "tracks":  # a noisy bearing towards EVE
            b = math.atan2(dy, dx) + rng.normal(0, cc["tracks_sigma"])
            return {"bearing_deg": round(math.degrees(b) % 360, 1)}
        if kind == "debris":  # a noisy range to EVE
            return {"distance": round(max(0.0, math.hypot(dx, dy) + rng.normal(0, cc["debris_sigma"])), 1)}
        ok = rng.random() < cc["log_accuracy"]  # "log": a status report, sometimes wrong
        other = [s for s in STATUSES if s != self.eve_status][0]
        return {"status": self.eve_status if ok else other}

    # ------------------------------------------------------------------ sensing
    def _bias_for(self, cell):
        if cell not in self._bias:
            g = np.random.default_rng([self.seed, cell[0], cell[1], 77])
            self._bias[cell] = g.normal(0, 1, 2)
        return self._bias[cell]

    def _read_signal(self, mult, allow_decoy=True):
        sn = self.cfg["sensors"]
        src, gain, is_decoy = self.eve_pos, sn["status_gain"][self.eve_status], False
        if allow_decoy and self.decoys and self.obs_rng.random() < sn["decoy_prob"]:
            src = self.decoys[self.obs_rng.integers(len(self.decoys))]
            gain, is_decoy = 1.0, True  # decoys always look like an active EVE
        dx, dy = src[0] - self.pos[0], src[1] - self.pos[1]
        s_true = sn["amplitude"] * gain / (1.0 + (dx * dx + dy * dy) / sn["d0"] ** 2)
        share = sn["bias_share"]
        fresh, bias = math.sqrt(1 - share) * mult, math.sqrt(share)
        zb = self._bias_for(self.pos)
        ls = math.log(s_true) + sn["sigma_logstrength"] * (bias * zb[0] + fresh * self.obs_rng.normal())
        b = math.atan2(dy, dx) + sn["sigma_bearing"] * (bias * zb[1] + fresh * self.obs_rng.normal())
        sig = {"bearing_deg": round(math.degrees(b) % 360, 2),
               "strength": round(math.exp(ls), 4), "timestamp": self.time_used}
        return sig, is_decoy

    def _init_signal(self):
        s = self.cfg["initial_signal"]
        if s["mode"] == "explicit":
            self.initial_signal = {"bearing_deg": float(s["bearing_deg"]),
                                   "strength": float(s["strength"]),
                                   "timestamp": float(s.get("timestamp", 0))}
        else:
            sig, _ = self._read_signal(self.cfg["sensors"]["initial_noise_mult"], allow_decoy=False)
            self.initial_signal = sig

    def _reveal(self, radius):
        out = []
        for c in self.clues:
            if c["id"] in self.revealed:
                continue
            if math.dist(c["pos"], self.pos) <= radius + 1e-9:
                self.revealed.add(c["id"])
                out.append({"id": c["id"], "type": c["type"], "pos": list(c["pos"]),
                            "payload": dict(c["payload"])})
        return out

    # ------------------------------------------------------------------ dynamics
    def action_cost(self, action):
        c = self.cfg["resources"]["costs"]
        k = action["type"]
        if k == "move":
            return c["move"]["battery"], c["move"]["time"]
        if k == "scan":
            return c["scan"]["battery"] + c["scan"]["battery_per_radius"] * action["radius"], c["scan"]["time"]
        if k == "recharge":
            return 0, c["recharge"]["time"]
        raise ValueError(k)

    def step(self, action):
        """Apply an action. Returns an observation dict (see keys below)."""
        obs = {"ok": True, "signal": None, "clues": [], "detected": None, "message": "", "debug": {}}
        kind = action["type"]
        if kind == "move":
            tx, ty = action["to"]
            x, y = self.pos
            if abs(tx - x) + abs(ty - y) != 1 or not (0 <= tx < self.W and 0 <= ty < self.H) or self.walls[ty, tx]:
                return self._fail(obs, "invalid move")
        elif kind == "recharge":
            if self.pos not in self.stations:
                return self._fail(obs, "not at a charging station")
        elif kind != "scan":
            return self._fail(obs, f"unknown action {kind}")
        cb, ct = self.action_cost(action)
        if self.battery < cb or self.time_used + ct > self.time_budget + 1e-9:
            return self._fail(obs, "insufficient battery or time")

        self.battery -= cb
        self.time_used += ct
        sn = self.cfg["sensors"]
        if kind == "move":
            self.pos = (int(action["to"][0]), int(action["to"][1]))
            obs["signal"], dec = self._read_signal(sn["passive_noise_mult"])
            obs["debug"]["decoy_reading"] = dec
            obs["clues"] = self._reveal(1.0)
        elif kind == "scan":
            r = action["radius"]
            obs["signal"], dec = self._read_signal(1.0)
            obs["debug"]["decoy_reading"] = dec
            obs["clues"] = self._reveal(r)
            if math.dist(self.pos, self.eve_pos) <= r and self.obs_rng.random() < sn["p_detect"]:
                acc = sn["detect_status_accuracy"]
                other = [s for s in STATUSES if s != self.eve_status][0]
                status = self.eve_status if self.obs_rng.random() < acc else other
                obs["detected"] = {"pos": list(self.eve_pos), "status": status}
                self.detected = True
        else:
            self.battery = min(self.capacity, self.battery + self.cfg["resources"]["costs"]["recharge"]["amount"])
        obs.update(pos=list(self.pos), battery=self.battery, time_used=self.time_used,
                   done=self.time_budget - self.time_used < 1)
        return obs

    def _fail(self, obs, msg):
        obs.update(ok=False, message=msg, pos=list(self.pos), battery=self.battery,
                   time_used=self.time_used, done=False)
        return obs

    # ------------------------------------------------------------------ interfaces
    def public_info(self):
        """Everything the agent is told up front (the mission briefing)."""
        return {"W": self.W, "H": self.H, "walls": self.walls.copy(), "base": self.base,
                "stations": list(self.stations), "start": self.base,
                "initial_signal": dict(self.initial_signal)}

    def truth(self):
        return {"eve_pos": self.eve_pos, "eve_status": self.eve_status,
                "clues": [dict(c) for c in self.clues], "decoys": list(self.decoys)}

    def commit(self, est_xy, est_status):
        """Score the agent's final answer against the hidden truth."""
        err = math.dist(est_xy, self.eve_pos)
        return {"error_cells": round(err, 2),
                "location_correct": err <= self.cfg["evaluation"]["tolerance"],
                "status_correct": est_status == self.eve_status,
                "true_pos": list(self.eve_pos), "true_status": self.eve_status}
