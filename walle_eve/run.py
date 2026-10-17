"""Run one mission and write the trace, report and visualisations.

    python -m walle_eve.run --config configs/sample.yaml --out outputs
    python -m walle_eve.run --config configs/sample.yaml --agent greedy --seed 3
"""
import argparse
import json
from pathlib import Path

from .agent import AGENTS
from .config import load_config
from .env import WallEEnv


def run_mission(cfg, agent_type=None, snapshots=True, max_iters=2000):
    """Simulate a mission. Returns a result dict (trace, estimate, evaluation, summary, ...)."""
    agent_type = agent_type or cfg["agent"]["type"]
    env = WallEEnv(cfg)
    info = env.public_info()
    agent = AGENTS[agent_type](info, cfg, seed=cfg["seed"])
    trace, snaps = [], []

    def record(step, action, decision, obs):
        rec = {
            "step": step, "t": env.time_used, "battery": env.battery, "pos": list(env.pos),
            "mode": decision["mode"], "action": action,
            "observation": None if obs is None else {
                "signal": obs["signal"], "clues": obs["clues"], "detected": obs["detected"],
                "ok": obs["ok"], "message": obs["message"]},
            "belief": agent.belief_summary(),
            "decision": {"target": decision["target"], "expected_gain": decision["gain"],
                         "trigger": decision["trigger"], "alternatives": decision["alternatives"]},
            "reasoning": decision["reasoning"],
            "strategy_change": decision["strategy_change"], "strategy_note": decision["strategy_note"],
            "debug_truth": {} if obs is None else obs["debug"],   # ground truth, not seen by the agent
        }
        trace.append(rec)
        if snapshots:
            snaps.append(agent.snapshot())

    # step 0: the initial signal reading, already folded into the agent's belief
    record(0, {"type": "init", "signal": info["initial_signal"]},
           agent.dec("init", None, "Initial signal reading incorporated into the belief."), None)

    for step in range(1, max_iters + 1):
        decision = agent.decide()
        action = decision["action"]
        if action["type"] == "conclude":
            record(step, action, decision, None)
            break
        obs = env.step(action)
        agent.observe(action, obs)
        record(step, action, decision, obs)
        if not obs["ok"] or obs["done"]:
            break

    est_xy, est_status = agent.estimate()
    evaluation = env.commit(est_xy, est_status)
    acts = [t["action"]["type"] for t in trace]
    summary = {
        "agent": agent_type, "seed": cfg["seed"], "steps": len(trace) - 1,
        "time_used": env.time_used, "battery_used": sum(
            env.action_cost(t["action"])[0] for t in trace if t["action"]["type"] in ("move", "scan")),
        "moves": acts.count("move"), "scans": acts.count("scan"), "recharges": acts.count("recharge"),
        "clues_found": len(env.revealed), "detected": env.detected,
        "strategy_changes": sum(1 for t in trace if t["strategy_change"]),
        "final_battery": env.battery,
    }
    conf = trace[-1]["belief"].get("confidence")
    return {"config": cfg, "agent_name": agent_type, "agent": agent, "env": env, "info": info,
            "trace": trace, "snapshots": snaps, "estimate": {"pos": list(est_xy), "status": est_status,
                                                             "confidence": conf},
            "evaluation": evaluation, "summary": summary, "truth": env.truth()}


def main():
    from . import report, viz
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sample.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--agent", choices=list(AGENTS), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    over = {"seed": args.seed} if args.seed is not None else None
    cfg = load_config(args.config, over)
    res = run_mission(cfg, args.agent)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if res["agent_name"] == "bayes" else f"_{res['agent_name']}"

    with open(out / f"sample_trace{tag}.jsonl", "w") as f:
        for rec in res["trace"]:
            f.write(json.dumps(rec) + "\n")
    (out / f"sample_report{tag}.md").write_text(report.build_report(res))
    (out / f"summary{tag}.json").write_text(json.dumps(
        {"summary": res["summary"], "estimate": res["estimate"], "evaluation": res["evaluation"]}, indent=2))
    (out / f"path_ascii{tag}.txt").write_text(viz.ascii_map(res))
    if not args.no_plots:
        viz.plot_overview(res, out / f"path{tag}.png")
        viz.plot_timeline(res, out / f"timeline{tag}.png")
        if res["snapshots"] and res["snapshots"][0] is not None:
            viz.plot_belief_evolution(res, out / f"belief_evolution{tag}.png")

    print(viz.ascii_map(res))
    e, s = res["evaluation"], res["summary"]
    print(f"\nagent={res['agent_name']} seed={cfg['seed']} steps={s['steps']} time={s['time_used']:.0f} "
          f"scans={s['scans']} recharges={s['recharges']} clues={s['clues_found']}")
    print(f"estimate={res['estimate']['pos']} status={res['estimate']['status']} "
          f"conf={res['estimate']['confidence']} | truth={e['true_pos']} {e['true_status']} "
          f"| error={e['error_cells']} cells, location_correct={e['location_correct']}, "
          f"status_correct={e['status_correct']}")
    print(f"outputs written to {out}/")


if __name__ == "__main__":
    main()
