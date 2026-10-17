"""Final mission report: evidence, inferred location/status, confidence, recommended next step."""
import math

from .belief import STATUSES
from .grid import bfs


def spoof_analysis(res):
    """Flag readings that the final belief cannot explain (> 3 sigma): suspected spoofed signals."""
    ag = res["agent"]
    if not hasattr(ag, "model"):
        return None
    m, cfg = ag.model, res["config"]
    (ex, ey), status = res["estimate"]["pos"], res["estimate"]["status"]
    gain = cfg["sensors"]["status_gain"][status]
    flagged, tp = [], 0
    base = res["trace"][0]
    readings = [(tuple(res["info"]["start"]), res["info"]["initial_signal"], m.initial_mult, 0, None)]
    for t in res["trace"][1:]:
        o = t["observation"]
        if o and o["signal"]:
            readings.append((tuple(t["pos"]), o["signal"], m.passive_mult if t["action"]["type"] == "move" else 1.0,
                             t["step"], t["debug_truth"].get("decoy_reading")))
    for pos, sig, mult, step, truth_decoy in readings:
        dx, dy = ex - pos[0], ey - pos[1]
        exp_ls = math.log(m.A * gain / (1 + (dx * dx + dy * dy) / m.d0 ** 2))
        ss, sb = m.eff(mult)
        zs = (math.log(sig["strength"]) - exp_ls) / ss
        db = (math.radians(sig["bearing_deg"]) - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi
        zb = db / sb
        if max(abs(zs), abs(zb)) > 3:
            flagged.append((step, pos, round(zs, 1), round(zb, 1), truth_decoy))
            tp += bool(truth_decoy)
    true_decoys = sum(1 for _, _, _, _, td in readings if td)
    return {"n": len(readings), "flagged": flagged, "true_positive": tp, "true_decoys": true_decoys}


def recommendation(res):
    ag, info, cfg = res["agent"], res["info"], res["config"]
    est, status, conf = tuple(res["estimate"]["pos"]), res["estimate"]["status"], res["estimate"]["confidence"]
    mb = cfg["resources"]["costs"]["move"]["battery"]
    dist, _ = bfs(info["walls"], ag.pos)
    d = int(dist[est[1], est[0]])
    home = int(ag.dist_station[est[1], est[0]])
    need = (d + home) * mb + cfg["resources"]["reserve"]
    lines = []
    if res["summary"]["detected"]:
        lines.append(f"EVE is confirmed at {est} ({status}). Dispatch WALL-E: the target is {d} moves from its "
                     f"current cell {ag.pos}, {home} moves from the nearest charging station.")
        if ag.battery >= need:
            lines.append(f"Battery {ag.battery:.0f} covers the round trip ({need:.0f} needed incl. reserve); go now.")
        else:
            lines.append(f"Battery {ag.battery:.0f} is below the {need:.0f} needed for a round trip including reserve; "
                         f"recharge at the nearest station first.")
        lines.append("EVE appears **dormant** (weak emitter): bring a power-transfer / wake-up routine and expect a "
                     "faint approach signal." if status == "dormant" else
                     "EVE appears **active**: approach and open a communication channel.")
    else:
        lines.append(f"EVE has **not** been confirmed. Best hypothesis: {est} ({status}), confidence "
                     f"{(conf if conf is not None else float('nan')):.2f}.")
        if hasattr(ag, "belief"):
            modes = ag.belief.top_modes(3)
            lines.append("Next step: scan the highest-mass hypotheses in order: " + "; ".join(
                f"{(x, y)} (posterior mass {mass:.2f} within 4 cells, {int(dist[y, x])} moves away)"
                for x, y, mass in modes) + ".")
        else:
            lines.append(f"Next step: scan around {est}, {d} moves from WALL-E's current cell.")
        lines.append("Recharge at a station before the leg if battery is below the round-trip need "
                     f"of {need:.0f}.")
    return lines


def build_report(res):
    cfg, s, ev, est = res["config"], res["summary"], res["evaluation"], res["estimate"]
    ag, trace = res["agent"], res["trace"]
    L = []
    L.append(f"# Mission report: WALL-E and EVE's Signal\n")
    L.append(f"Agent: **{res['agent_name']}** | seed: {cfg['seed']} | map {cfg['map']['width']}x{cfg['map']['height']} | "
             f"decoy probability {cfg['sensors']['decoy_prob']}\n")

    L.append("## Outcome\n")
    L.append(f"- **Inferred location:** {tuple(est['pos'])}")
    if hasattr(ag, "belief"):
        sp = ag.belief.status_probs()
        L.append(f"- **Inferred status:** {est['status']} ({', '.join(f'P({k})={v:.2f}' for k, v in sp.items())})")
        L.append(f"- **Confidence:** {est['confidence']:.2f} (posterior mass within "
                 f"{cfg['agent']['confidence_radius']} cells of the estimate)")
        L.append(f"- **Residual entropy:** {ag.belief.entropy():.2f} nats (start: {trace[0]['belief']['entropy']:.2f})")
    else:
        L.append(f"- **Inferred status:** {est['status']} (baseline does not model status)")
    L.append(f"- **Direct detection by scan:** {'yes' if s['detected'] else 'no'}")
    L.append(f"- **Resources:** {s['time_used']:.0f}/{cfg['resources']['time_budget']} time, "
             f"{s['battery_used']:.0f} battery spent, {s['moves']} moves, {s['scans']} scans, "
             f"{s['recharges']} recharges, final battery {s['final_battery']:.0f}\n")

    L.append("## Evidence found\n")
    L.append(f"- Initial signal: bearing {res['info']['initial_signal']['bearing_deg']:.0f} deg, "
             f"strength {res['info']['initial_signal']['strength']:.2f} at the base.")
    clues = [(t["step"], c) for t in trace if t["observation"] for c in t["observation"]["clues"]]
    if clues:
        for step, c in clues:
            pl = c["payload"]
            what = (f"bearing {pl['bearing_deg']:.0f} deg towards EVE" if c["type"] == "tracks" else
                    f"EVE about {pl['distance']:.1f} cells away" if c["type"] == "debris" else
                    f"status log says **{pl['status']}** (about 85% reliable)")
            L.append(f"- Step {step}: `{c['type']}` clue at {tuple(c['pos'])}: {what}.")
    else:
        L.append("- No clues were discovered.")
    for t in trace:
        d = t["observation"] and t["observation"]["detected"]
        if d:
            L.append(f"- Step {t['step']}: **scan detected EVE** at {tuple(d['pos'])}, reported status {d['status']}.")
    sp = spoof_analysis(res)
    if sp:
        L.append(f"- Spoof check: {len(sp['flagged'])} of {sp['n']} signal readings are inconsistent (> 3 sigma) "
                 f"with the final estimate and are treated as suspected decoys.")
    L.append("")

    L.append("## Strategy changes\n")
    ch = [t for t in trace if t["strategy_change"]]
    if ch:
        for t in ch:
            L.append(f"- Step {t['step']} (t={t['t']:.0f}, battery {t['battery']:.0f}): {t['strategy_note']}")
    else:
        L.append("- None (single plan).")
    L.append("")

    L.append("## Recommended next step\n")
    for line in recommendation(res):
        L.append(f"- {line}")
    L.append("")

    L.append("## Ground-truth check (simulation only)\n")
    L.append(f"- True position {tuple(ev['true_pos'])}, true status {ev['true_status']}.")
    L.append(f"- Error {ev['error_cells']} cells (tolerance {cfg['evaluation']['tolerance']}): "
             f"location {'correct' if ev['location_correct'] else 'WRONG'}, "
             f"status {'correct' if ev['status_correct'] else 'WRONG'}.")
    if sp:
        L.append(f"- Spoofed readings actually received: {sp['true_decoys']}; flagged correctly: {sp['true_positive']}; "
                 f"false alarms: {len(sp['flagged']) - sp['true_positive']}.")
    L.append(f"- Hidden decoys at {res['truth']['decoys']}; "
             f"{s['clues_found']}/{cfg['clues']['count']} clues discovered.\n")
    return "\n".join(L)
