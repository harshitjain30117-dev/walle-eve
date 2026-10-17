"""Run with `pytest -q` (or `python tests/test_walle_eve.py` if pytest is not installed)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from walle_eve.belief import Belief, SensorModel  # noqa: E402
from walle_eve.config import load_config  # noqa: E402
from walle_eve.env import WallEEnv  # noqa: E402
from walle_eve.grid import INF, bfs  # noqa: E402
from walle_eve.run import run_mission  # noqa: E402


def test_env_is_deterministic():
    a, b = WallEEnv(load_config(None, {"seed": 5})), WallEEnv(load_config(None, {"seed": 5}))
    assert a.eve_pos == b.eve_pos and (a.walls == b.walls).all()
    assert a.initial_signal == b.initial_signal


def test_world_is_connected_and_entities_reachable():
    for seed in range(5):
        e = WallEEnv(load_config(None, {"seed": seed}))
        dist, _ = bfs(e.walls, e.base)
        assert dist[e.eve_pos[1], e.eve_pos[0]] < INF
        assert all(dist[s[1], s[0]] < INF for s in e.stations)
        assert all(dist[c["pos"][1], c["pos"][0]] < INF for c in e.clues)


def test_env_rejects_invalid_actions_and_charges_costs():
    e = WallEEnv(load_config(None, {"seed": 1}))
    assert not e.step({"type": "move", "to": [e.pos[0] + 5, e.pos[1]]})["ok"]
    b0 = e.battery
    o = e.step({"type": "scan", "radius": 3})
    assert o["ok"] and e.battery == b0 - 5 and e.time_used == 2
    assert e.step({"type": "move", "to": [e.pos[0] + 1, e.pos[1]]})["signal"] is not None


def test_belief_concentrates_on_truth_with_low_noise():
    cfg = load_config(None, {"seed": 2, "sensors": {"decoy_prob": 0.0}})
    e = WallEEnv(cfg)
    b = Belief(e.walls, cfg["eve"]["status_prior"], SensorModel(cfg))
    h0 = b.entropy()
    for pos in [(5, 5), (30, 5), (30, 25), (10, 25)]:
        if e.walls[pos[1], pos[0]]:
            continue
        e.pos = pos
        sig, _ = e._read_signal(1.0, allow_decoy=False)
        b.update(b.lik_signal(b.prep_signal(*pos), math.radians(sig["bearing_deg"]), sig["strength"], 1.0))
    assert b.entropy() < h0 - 2
    assert math.dist(b.map_cell(), e.eve_pos) < 6
    assert abs(b.p.sum() - 1) < 1e-9


def test_belief_survives_a_spoofed_reading():
    cfg = load_config(None, {"seed": 2})
    e = WallEEnv(cfg)
    b = Belief(e.walls, cfg["eve"]["status_prior"], SensorModel(cfg))
    b.update(b.lik_signal(b.prep_signal(2, 15), math.radians(10.0), 50.0, 1.0))   # absurdly strong reading
    assert np.isfinite(b.p).all() and abs(b.p.sum() - 1) < 1e-9


def test_full_mission_respects_budgets_and_never_strands_the_robot():
    for seed in (1, 4):
        cfg = load_config(None, {"seed": seed})
        r = run_mission(cfg, "bayes", snapshots=False)
        env = r["env"]
        assert env.time_used <= cfg["resources"]["time_budget"]
        for t in r["trace"]:
            assert t["battery"] >= 0
            if t["action"]["type"] in ("move", "scan"):
                cell = tuple(t["pos"])
                need = r["agent"].dist_station[cell[1], cell[0]]
                assert t["battery"] >= need, "robot could not have returned to a station"
        assert all("reasoning" in t and t["reasoning"] for t in r["trace"])


def test_bayes_finds_eve_on_default_seeds():
    ok = 0
    for seed in range(1, 5):
        r = run_mission(load_config(None, {"seed": seed}), "bayes", snapshots=False)
        ok += r["evaluation"]["location_correct"]
    assert ok >= 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
