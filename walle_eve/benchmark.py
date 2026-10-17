"""Generalisation benchmark: Bayesian planner vs greedy baseline on random missions.

    python -m walle_eve.benchmark --seeds 20 --out outputs

Each seed draws a new map, EVE position/status, clues and decoys. Conditions:
    clean  : no spoofed signals
    decoys : 15% of readings spoofed (default mission)
    hard   : 30% spoofed, 4 decoys, noisier sensors, tighter battery/time
"""
import argparse
import csv
import json
import statistics as st
import time
from pathlib import Path

from .config import load_config
from .run import run_mission

CONDITIONS = {
    "clean": {"sensors": {"decoy_prob": 0.0, "decoy_count": 0}},
    "decoys": {},
    "hard": {"sensors": {"decoy_prob": 0.30, "decoy_count": 4, "sigma_bearing": 0.35, "sigma_logstrength": 0.30},
             "resources": {"battery_capacity": 50, "time_budget": 180},
             "agent": {"assumed_outlier_prob": 0.3}},
}


def run_one(base_cfg, cond, seed, agent):
    from .config import deep_merge
    cfg = deep_merge(deep_merge(base_cfg, CONDITIONS[cond]), {"seed": seed})
    r = run_mission(cfg, agent, snapshots=False)
    e, s = r["evaluation"], r["summary"]
    return {"condition": cond, "agent": agent, "seed": seed, "error": e["error_cells"],
            "loc_ok": int(e["location_correct"]), "status_ok": int(e["status_correct"]),
            "detected": int(s["detected"]), "time_used": s["time_used"], "battery_used": s["battery_used"],
            "scans": s["scans"], "recharges": s["recharges"],
            "confidence": r["estimate"]["confidence"] if r["estimate"]["confidence"] is not None else ""}


def summarise(rows):
    out = []
    for cond in CONDITIONS:
        for agent in ("bayes", "greedy"):
            rs = [r for r in rows if r["condition"] == cond and r["agent"] == agent]
            if not rs:
                continue
            n = len(rs)
            out.append({
                "condition": cond, "agent": agent, "n": n,
                "location_success": sum(r["loc_ok"] for r in rs) / n,
                "status_accuracy": sum(r["status_ok"] for r in rs) / n,
                "both_correct": sum(r["loc_ok"] and r["status_ok"] for r in rs) / n,
                "detected": sum(r["detected"] for r in rs) / n,
                "median_error": st.median(r["error"] for r in rs),
                "mean_time": st.mean(r["time_used"] for r in rs),
                "mean_battery": st.mean(r["battery_used"] for r in rs),
            })
    return out


def to_markdown(summary, n_seeds):
    L = [f"Benchmark over {n_seeds} random missions per cell (new map, EVE, clues and decoys each seed).\n",
         "| condition | agent | location correct | status correct | found by scan | median error (cells) | mean time used | mean battery used |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['condition']} | {s['agent']} | {s['location_success']:.0%} | {s['status_accuracy']:.0%} | "
                 f"{s['detected']:.0%} | {s['median_error']:.1f} | {s['mean_time']:.0f} | {s['mean_battery']:.0f} |")
    return "\n".join(L) + "\n"


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    conds = list(dict.fromkeys(s["condition"] for s in summary))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "location_success", "EVE located within tolerance"),
                           (axes[1], "mean_time", "mean time used (budget differs in 'hard')")):
        x = np.arange(len(conds))
        for i, agent in enumerate(("bayes", "greedy")):
            vals = [next(s[key] for s in summary if s["condition"] == c and s["agent"] == agent) for c in conds]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=agent,
                   color=("tab:blue" if agent == "bayes" else "tab:gray"))
        ax.set_xticks(x)
        ax.set_xticklabels(conds)
        ax.set_title(title)
        if key == "location_success":
            ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sample.yaml")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=100, help="first seed")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    base = load_config(args.config)
    base["initial_signal"] = {"mode": "auto"}   # every seed gets its own initial reading
    rows, t0 = [], time.time()
    for cond in CONDITIONS:
        for seed in range(args.start, args.start + args.seeds):
            for agent in ("bayes", "greedy"):
                rows.append(run_one(base, cond, seed, agent))
            print(f"[{time.time() - t0:5.0f}s] {cond} seed {seed} done", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary, args.seeds)
    (out / "benchmark.md").write_text(md)
    plot(summary, out / "benchmark.png")
    print("\n" + md)


if __name__ == "__main__":
    main()
