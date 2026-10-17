"""Mission config loading (YAML or JSON) with defaults."""
import copy
import json
from pathlib import Path

DEFAULTS = {
    "seed": 7,
    "map": {
        "width": 40, "height": 30, "obstacle_density": 0.08,
        "base": [2, 15], "stations": [[20, 15]],   # base is always a charging station
    },
    # Hidden truth. null => drawn from the seed.
    "eve": {"pos": None, "status": None,
            "status_prior": {"active": 0.6, "dormant": 0.4}},
    # "auto": reading taken at the base from the hidden world. "explicit": use values below.
    "initial_signal": {"mode": "auto", "bearing_deg": None, "strength": None, "timestamp": 0},
    "resources": {
        "battery_capacity": 60, "time_budget": 240, "reserve": 3,
        "costs": {
            "move": {"battery": 1, "time": 1},
            "scan": {"battery": 2, "time": 2, "battery_per_radius": 1},
            "recharge": {"amount": 30, "time": 6},
        },
    },
    "clues": {"count": 6, "types": {"tracks": 0.4, "debris": 0.35, "log": 0.25},
              "tracks_sigma": 0.45, "debris_sigma": 1.5, "log_accuracy": 0.85,
              "near_eve_fraction": 0.5, "near_eve_radius": 14},
    "sensors": {
        "amplitude": 100.0, "d0": 4.0,
        "status_gain": {"active": 1.0, "dormant": 0.35},   # dormant EVE is a weaker emitter
        "sigma_logstrength": 0.25, "sigma_bearing": 0.25,  # total noise of a scan reading
        "bias_share": 0.35,            # share of noise variance that is fixed per grid cell
        "passive_noise_mult": 2.0, "initial_noise_mult": 1.5,
        "decoy_count": 2, "decoy_prob": 0.15,              # spoofed readings
        "p_detect": 0.9, "detect_status_accuracy": 0.95,
    },
    "agent": {
        "type": "bayes",
        "conclude_confidence": 0.9, "confidence_radius": 2.0,
        "candidate_stride": 3, "scan_radii": [2, 4], "mc_samples": 6,
        "assumed_outlier_prob": 0.15, "min_gain": 0.01,
        "replan_entropy_delta": 0.3, "replan_every": 5,
        "revisit_penalty": 0.35, "clue_bonus": 0.12, "find_bonus": 3.0, "recharge_below": 0.5,
    },
    "evaluation": {"tolerance": 3.0},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, overrides=None):
    data = {}
    if path:
        text = Path(path).read_text()
        if str(path).endswith(".json"):
            data = json.loads(text)
        else:
            import yaml
            data = yaml.safe_load(text) or {}
    return deep_merge(deep_merge(DEFAULTS, data), overrides or {})
