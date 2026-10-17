"""Autonomous agents.

BaseAgent   : bookkeeping + the battery "turn back" invariant (always keep enough battery to
              reach a charging station, plus a reserve).
BayesAgent  : maintains a belief over EVE, plans by expected information gain per cost,
              replans on new evidence and logs why it chose each action.
GreedyAgent : baseline with no belief. Inverts the latest reading into a point estimate and
              walks towards it. Used in the benchmark.
"""
import math

import numpy as np

from .belief import STATUSES, Belief, SensorModel
from .grid import bfs, path_from_parent
from .planner import InfoPlanner


class BaseAgent:
    name = "base"

    def __init__(self, info, cfg, seed=0):
        self.info, self.cfg = info, cfg
        self.walls, self.W, self.H = info["walls"], info["W"], info["H"]
        self.stations = [tuple(s) for s in info["stations"]]
        res = cfg["resources"]
        self.res, self.costs = res, res["costs"]
        self.cap, self.reserve, self.T = res["battery_capacity"], res["reserve"], res["time_budget"]
        self.pos = tuple(info["start"])
        self.battery, self.time_used = float(self.cap), 0.0
        self.dist_station = np.minimum.reduce([bfs(self.walls, s)[0] for s in self.stations])
        self.detection = None
        self.rng = np.random.default_rng(seed + 99)

    @property
    def time_left(self):
        return self.T - self.time_used

    def ret_cost(self, cell):
        return self.dist_station[cell[1], cell[0]] * self.costs["move"]["battery"]

    def scan_battery(self, r):
        s = self.costs["scan"]
        return s["battery"] + s["battery_per_radius"] * r

    def at_station(self):
        return self.pos in self.stations

    def observe(self, action, obs):
        if obs["ok"]:
            self.pos, self.battery, self.time_used = tuple(obs["pos"]), obs["battery"], obs["time_used"]

    def dec(self, mode, action, reasoning, **kw):
        d = {"mode": mode, "action": action, "reasoning": reasoning, "target": None, "gain": None,
             "alternatives": [], "strategy_change": False, "strategy_note": None, "trigger": None}
        d.update(kw)
        return d

    def snapshot(self):
        return None

    def belief_summary(self):
        return {}


# =============================================================================== Bayesian agent
class BayesAgent(BaseAgent):
    name = "bayes"

    def __init__(self, info, cfg, seed=0):
        super().__init__(info, cfg, seed)
        self.a = cfg["agent"]
        self.model = SensorModel(cfg)
        self.belief = Belief(self.walls, cfg["eve"]["status_prior"], self.model)
        self.planner = InfoPlanner(self.belief, cfg, info, self.dist_station)
        self.scanned_mask = np.zeros((self.H, self.W), bool)
        self.scanned_positions = []
        self.clues_known = []
        self.plan, self.new_clue, self.prev_mode, self.prev_target = None, False, None, None
        self.last_ranking = []
        s = info["initial_signal"]
        self._apply_signal(self.pos, s, self.model.initial_mult)

    # ------------------------------------------------------------ evidence
    def _apply_signal(self, pos, sig, mult):
        prep = self.belief.prep_signal(*pos)
        lik = self.belief.lik_signal(prep, math.radians(sig["bearing_deg"]), sig["strength"], mult)
        self.belief.update(lik)

    def observe(self, action, obs):
        super().observe(action, obs)
        if not obs["ok"]:
            self.plan = None
            return
        kind, b = action["type"], self.belief
        if obs["signal"]:
            self._apply_signal(self.pos, obs["signal"], self.model.passive_mult if kind == "move" else 1.0)
        for c in obs["clues"]:
            b.update(b.lik_clue(c))
            self.clues_known.append(dict(c))
            self.new_clue = True
        if kind == "scan":
            r = action["radius"]
            area = (b.XX - self.pos[0]) ** 2 + (b.YY - self.pos[1]) ** 2 <= r * r
            self.scanned_mask |= area
            self.scanned_positions.append(self.pos)
            if obs["detected"]:
                d = obs["detected"]
                self.detection = d
                b.update(b.lik_detect_pos(d["pos"][0], d["pos"][1], d["status"]))
            else:
                b.update(b.lik_detect_neg(self.pos[0], self.pos[1], r))
        if self.plan:
            self.plan["age"] += 1
            if kind == "move":
                self.plan["i"] += 1
                if self.plan["mode"] == "return" and self.pos == self.plan["path"][-1]:
                    self.plan["done"] = True
            else:
                self.plan["done"] = True

    # ------------------------------------------------------------ decisions
    def decide(self):
        b = self.belief
        conf = b.confidence(self.a["confidence_radius"])
        if self.detection and conf >= self.a["conclude_confidence"]:
            return self.dec("conclude", {"type": "conclude"},
                            f"EVE detected by scan at {tuple(self.detection['pos'])}; posterior mass near the "
                            f"estimate is {conf:.2f} (>= {self.a['conclude_confidence']}). Concluding.")
        if self.time_left < 1:
            return self.dec("conclude", {"type": "conclude"}, "Time budget exhausted. Reporting best estimate.")
        trigger = self._trigger()
        if trigger:
            self._make_plan(trigger)
        return self._act()

    def _trigger(self):
        p = self.plan
        if p is None:
            return "start" if self.prev_mode is None else "replan"
        if p["done"]:
            return "plan complete"
        if self.new_clue:
            return "new clue"
        if abs(self.belief.entropy() - p["H0"]) > self.a["replan_entropy_delta"]:
            return "belief shift"
        if p["age"] >= self.a["replan_every"]:
            return "periodic review"
        return None

    def _status_str(self):
        return ", ".join(f"P({k})={v:.2f}" for k, v in self.belief.status_probs().items())

    def _make_plan(self, trigger):
        b, a, rc = self.belief, self.a, self.costs["recharge"]
        self.new_clue = False
        dist_r, parent = bfs(self.walls, self.pos)
        cands = self.planner.rank(self.pos, self.battery, self.time_left, dist_r,
                                  self.scanned_mask, self.scanned_positions, self.rng)
        self.last_ranking = cands[:5]
        at_st = self.at_station()
        can_rc = at_st and self.battery < self.cap - 1e-9 and self.time_left >= rc["time"] + 3
        H0, conf = b.entropy(), b.confidence(a["confidence_radius"])
        head = (f"Belief peak {b.map_cell()} (conf {conf:.2f}, entropy {H0:.2f} nats, {self._status_str()}). ")
        plan = {"H0": H0, "age": 0, "i": 0, "done": False, "trigger": trigger, "alts": cands[1:4]}

        if cands and cands[0]["gain"] >= a["min_gain"]:
            best = cands[0]
            # would topping up first unlock a clearly better (longer-range) scan?
            full_best = None
            if can_rc:
                topped = min(self.cap, self.battery + rc["amount"])
                alt = self.planner.rank(self.pos, topped, self.time_left - rc["time"], dist_r,
                                        self.scanned_mask, self.scanned_positions, self.rng)
                if alt:
                    full_best = alt[0]
                    full_best_score = full_best["gain"] / (full_best["cost"] + rc["time"] / self.T)
            if can_rc and full_best is not None and (
                    full_best_score > best["score"] * 1.1 or self.battery <= a["recharge_below"] * self.cap * 0.5):
                plan.update(mode="recharge", target=self.pos,
                            reason=head + f"At a station with battery {self.battery:.0f}/{self.cap}: recharging unlocks "
                                          f"a better scan, r={full_best['radius']} at {full_best['target']} "
                                          f"(score {full_best_score:.1f} vs {best['score']:.1f} without).")
            else:
                tgt = best["target"]
                margin = self.battery - best["battery_cost"] - self.dist_station[tgt[1], tgt[0]] * \
                    self.costs["move"]["battery"]
                plan.update(mode="explore", target=tgt, radius=best["radius"], gain=best["gain"],
                            path=path_from_parent(parent, self.pos, tgt),
                            reason=head + f"Best information per cost: scan r={best['radius']} at {tgt} "
                                          f"(expected gain {best['gain']:.2f} nats, travel {best['travel']}, "
                                          f"score {best['score']:.1f}). Battery margin after action: "
                                          f"{margin:.0f} above return+reserve.")
        elif cands:
            plan.update(mode="conclude", target=None,
                        reason=head + f"No scan is expected to cut uncertainty by more than {a['min_gain']} nats.")
        elif can_rc:
            plan.update(mode="recharge", target=self.pos,
                        reason=head + f"No scan is affordable at battery {self.battery:.0f}; recharging.")
        elif not at_st and self.battery < self.cap - 1e-9:
            st = min(self.stations, key=lambda s: dist_r[s[1], s[0]])
            d = int(dist_r[st[1], st[0]])
            if self.time_left >= d * self.costs["move"]["time"] + rc["time"] + 3:
                plan.update(mode="return", target=st, path=path_from_parent(parent, self.pos, st),
                            reason=head + f"Battery {self.battery:.0f} cannot fund another scan and still get home "
                                          f"(station {d} steps away, reserve {self.reserve}). Turning back to recharge.")
            else:
                plan.update(mode="conclude", target=None,
                            reason=head + "No affordable scan and not enough time to recharge. Concluding.")
        else:
            plan.update(mode="conclude", target=None, reason=head + "No feasible action within the budgets. Concluding.")

        # strategy-change bookkeeping
        note = None
        if self.prev_mode is not None and plan["mode"] != self.prev_mode:
            note = f"mode {self.prev_mode} -> {plan['mode']} (trigger: {trigger})"
        elif (plan["mode"] == "explore" and self.prev_target is not None
              and math.dist(plan["target"], self.prev_target) > 6):
            note = f"retargeted {self.prev_target} -> {plan['target']} (trigger: {trigger})"
        plan["change"] = note
        self.prev_mode = plan["mode"]
        if plan["mode"] == "explore":
            self.prev_target = plan["target"]
        self.plan = plan

    def _act(self):
        p, mb = self.plan, self.costs["move"]["battery"]
        change, note = False, p.pop("change", None)
        if note:
            change = True
        kw = dict(target=p.get("target"), gain=p.get("gain"), strategy_change=change, strategy_note=note,
                  trigger=p["trigger"],
                  alternatives=[{"target": list(c["target"]), "radius": c["radius"], "gain": round(c["gain"], 3),
                                 "score": round(c["score"], 2)} for c in p["alts"]])
        if p["mode"] == "conclude":
            return self.dec("conclude", {"type": "conclude"}, p["reason"], **kw)
        if p["mode"] == "recharge":
            return self.dec("recharge", {"type": "recharge"}, p["reason"], **kw)
        path, i = p["path"], p["i"]
        first = p["age"] == 0
        if p["mode"] == "explore" and i >= len(path) - 1:
            return self.dec("explore", {"type": "scan", "radius": p["radius"]},
                            p["reason"] if first else f"Arrived at {p['target']}; scanning r={p['radius']}.", **kw)
        if i >= len(path) - 1:
            p["done"] = True
            return self._act_replan()
        nxt = path[i + 1]
        if self.battery - mb < self.ret_cost(nxt) + self.reserve:   # paranoid invariant check
            return self._act_replan()
        why = p["reason"] if first else f"Continuing to {p['target']} ({len(path) - 1 - i} cells left)."
        return self.dec(p["mode"], {"type": "move", "to": list(nxt)}, why, **kw)

    def _act_replan(self):
        self.plan = None
        self._make_plan("safety")
        return self._act()

    # ------------------------------------------------------------ outputs
    def estimate(self):
        status = max(self.belief.status_probs().items(), key=lambda kv: kv[1])[0]
        return self.belief.map_cell(), status

    def belief_summary(self):
        b = self.belief
        return {"map_cell": list(b.map_cell()), "entropy": round(b.entropy(), 3),
                "confidence": round(b.confidence(self.a["confidence_radius"]), 3),
                "status": {k: round(v, 3) for k, v in b.status_probs().items()},
                "top_modes": [[x, y, round(m, 3)] for x, y, m in b.top_modes(3)]}

    def snapshot(self):
        return self.belief.marginal().astype(np.float32)


# =============================================================================== greedy baseline
class GreedyAgent(BaseAgent):
    name = "greedy"

    def __init__(self, info, cfg, seed=0):
        super().__init__(info, cfg, seed)
        sn = cfg["sensors"]
        self.A, self.d0 = sn["amplitude"], sn["d0"]
        self.sr = 3
        self.since_scan = 99
        self.est = (float(self.pos[0]), float(self.pos[1]))
        self._from_signal(info["initial_signal"])

    def _from_signal(self, sig):
        s = max(sig["strength"], 1e-6)
        d = self.d0 * math.sqrt(max(self.A / s - 1.0, 0.0))   # assumes an *active* EVE
        th = math.radians(sig["bearing_deg"])
        x = min(max(self.pos[0] + d * math.cos(th), 0), self.W - 1)
        y = min(max(self.pos[1] + d * math.sin(th), 0), self.H - 1)
        self.est = (x, y)

    def observe(self, action, obs):
        super().observe(action, obs)
        if not obs["ok"]:
            return
        self.since_scan = 0 if action["type"] == "scan" else self.since_scan + 1
        if obs["signal"]:
            self._from_signal(obs["signal"])
        if obs["detected"]:
            self.detection = obs["detected"]
            self.est = tuple(obs["detected"]["pos"])

    def decide(self):
        if self.detection:
            return self.dec("conclude", {"type": "conclude"}, "EVE detected by scan.")
        if self.time_left < 1:
            return self.dec("conclude", {"type": "conclude"}, "Time budget exhausted.")
        mb = self.costs["move"]["battery"]
        rc0 = self.costs["recharge"]
        if self.at_station() and self.battery < self.cap - 1e-9 and self.time_left >= rc0["time"] + 3:
            return self.dec("recharge", {"type": "recharge"}, "At a station: topping up to full before heading out.")
        dist_r, parent = bfs(self.walls, self.pos)
        reach = [(x, y) for y in range(self.H) for x in range(self.W) if dist_r[y, x] < 10**6]
        tgt = min(reach, key=lambda c: math.dist(c, self.est))
        sb, st = self.scan_battery(self.sr), self.costs["scan"]["time"]
        scan_ok = self.battery - sb >= self.ret_cost(self.pos) + self.reserve and self.time_left >= st
        near = math.dist(self.pos, self.est) <= self.sr
        tg = dict(target=tgt)
        if near and scan_ok and self.since_scan >= 3:
            return self.dec("explore", {"type": "scan", "radius": self.sr},
                            f"Estimate {tuple(round(v) for v in self.est)} is within scan range; scanning.", **tg)
        if self.pos != tgt:
            nxt = path_from_parent(parent, self.pos, tgt)[1]
        else:  # standing on the estimate with nothing found: wander to a neighbour
            opts = [(self.pos[0] + dx, self.pos[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            opts = [c for c in opts if 0 <= c[0] < self.W and 0 <= c[1] < self.H and not self.walls[c[1], c[0]]]
            nxt = opts[self.rng.integers(len(opts))]
        if self.battery - mb >= self.ret_cost(nxt) + self.reserve and self.time_left >= self.costs["move"]["time"]:
            return self.dec("explore", {"type": "move", "to": list(nxt)},
                            f"Heading towards latest point estimate {tuple(round(v) for v in self.est)}.", **tg)
        rc = self.costs["recharge"]
        if self.at_station():
            if self.battery < self.cap - 1e-9 and self.time_left >= rc["time"]:
                return self.dec("recharge", {"type": "recharge"}, "Battery too low to continue; recharging.")
            return self.dec("conclude", {"type": "conclude"}, "Cannot make further progress within budgets.")
        s = min(self.stations, key=lambda c: dist_r[c[1], c[0]])
        return self.dec("return", {"type": "move", "to": list(path_from_parent(parent, self.pos, s)[1])},
                        "Battery reserve reached; returning to a charging station.", target=s)

    def estimate(self):
        status = self.detection["status"] if self.detection else "active"
        return (int(round(self.est[0])), int(round(self.est[1]))), status

    def belief_summary(self):
        return {"map_cell": [int(round(self.est[0])), int(round(self.est[1]))]}


AGENTS = {"bayes": BayesAgent, "greedy": GreedyAgent}
