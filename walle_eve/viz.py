"""Visualisations: ASCII map, path + belief overview, belief evolution, resource timeline."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

CLUE_MARK = {"tracks": "^", "debris": "s", "log": "D"}


def _path(res):
    return [tuple(t["pos"]) for t in res["trace"]] if res["trace"] else []


def _scans(res):
    return [(tuple(t["pos"]), t["action"]["radius"]) for t in res["trace"] if t["action"]["type"] == "scan"]


def ascii_map(res):
    info, truth = res["info"], res["truth"]
    H, W = info["H"], info["W"]
    g = [["#" if info["walls"][y, x] else "." for x in range(W)] for y in range(H)]

    def put(c, ch):
        g[c[1]][c[0]] = ch

    for c in _path(res):
        put(c, "+")
    for c, _ in _scans(res):
        put(c, "o")
    for cl in truth["clues"]:
        put(cl["pos"], "C" if cl["id"] in res["env"].revealed else "c")
    for d in truth["decoys"]:
        put(d, "d")
    for s in info["stations"]:
        put(s, "S")
    put(info["base"], "H")
    ex, ey = res["estimate"]["pos"]
    put(truth["eve_pos"], "E")
    put((ex, ey), "@" if (ex, ey) == tuple(truth["eve_pos"]) else "X")
    legend = ("legend: # wall | + path | o scan | H base | S station | C clue found | c clue missed | "
              "d decoy | E EVE (truth) | X estimate | @ estimate==truth")
    return "\n".join("".join(r) for r in g) + "\n" + legend


def _base_axes(ax, res):
    info = res["info"]
    H, W = info["H"], info["W"]
    img = np.where(info["walls"], 0.2, 0.94)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=(-0.5, W - 0.5, -0.5, H - 0.5))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_overview(res, path):
    info, truth = res["info"], res["truth"]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _base_axes(ax, res)
    if res["snapshots"] and res["snapshots"][-1] is not None:
        m = np.ma.masked_less_equal(res["snapshots"][-1], 0.01 * float(res["snapshots"][-1].max()))
        im = ax.imshow(m, cmap="YlOrRd", alpha=0.85, origin="lower", vmin=0, vmax=float(max(m.max(), 1e-3)),
                       extent=(-0.5, info["W"] - 0.5, -0.5, info["H"] - 0.5))
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="final belief P(EVE in cell)")

    pts = np.array(_path(res), float)
    if len(pts) > 1:
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="winter", linewidths=2)
        lc.set_array(np.arange(len(segs)))
        ax.add_collection(lc)
    for (x, y), r in _scans(res):
        ax.add_patch(plt.Circle((x, y), r, fill=False, ec="black", lw=0.7, alpha=0.55, ls=":"))
    for cl in truth["clues"]:
        found = cl["id"] in res["env"].revealed
        ax.scatter(*cl["pos"], marker=CLUE_MARK[cl["type"]], s=70, zorder=5,
                   facecolors="lime" if found else "none", edgecolors="lime" if found else "gray", linewidths=1.5)
    for d in truth["decoys"]:
        ax.scatter(*d, marker="x", c="red", s=80, zorder=5)
    for s in info["stations"]:
        ax.scatter(*s, marker="s", c="orange", s=90, zorder=6, edgecolors="k")
    ax.scatter(*info["base"], marker="H", c="orange", s=140, zorder=6, edgecolors="k")
    ax.scatter(*truth["eve_pos"], marker="*", c="gold", s=330, zorder=7, edgecolors="k")
    ax.scatter(*res["estimate"]["pos"], marker="D", c="cyan", s=70, zorder=8, edgecolors="k")
    n = 0
    for t in res["trace"]:
        if t["strategy_change"]:
            n += 1
            ax.annotate(str(n), t["pos"], xytext=(4, 4), textcoords="offset points", color="white", fontsize=9,
                        bbox=dict(boxstyle="circle,pad=0.15", fc="crimson", ec="none"), zorder=9)
    handles = [
        plt.Line2D([], [], marker="*", color="gold", mec="k", ls="", ms=14, label="EVE (truth)"),
        plt.Line2D([], [], marker="D", color="cyan", mec="k", ls="", label="estimate"),
        plt.Line2D([], [], marker="x", color="red", ls="", label="decoy emitter"),
        plt.Line2D([], [], marker="^", color="lime", ls="", label="clue found (hollow = missed)"),
        plt.Line2D([], [], marker="s", color="orange", mec="k", ls="", label="charging station / base"),
        plt.Line2D([], [], color="black", lw=0.8, ls=":", label="scan radius"),
        plt.Line2D([], [], marker="o", color="crimson", ls="", label="strategy change (numbered)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=8, frameon=False)
    e = res["evaluation"]
    ax.set_title(f"{res['agent_name']} agent, seed {res['config']['seed']}: path (blue to green = time), "
                 f"error {e['error_cells']} cells")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_belief_evolution(res, path, n=6):
    snaps, trace, info = res["snapshots"], res["trace"], res["info"]
    idx = np.unique(np.linspace(0, len(snaps) - 1, n).astype(int))
    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        _base_axes(ax, res)
        ax.imshow(np.ma.masked_less_equal(snaps[i], 0.01 * float(snaps[i].max())), cmap="YlOrRd", alpha=0.9, origin="lower",
                  extent=(-0.5, info["W"] - 0.5, -0.5, info["H"] - 0.5))
        ax.scatter(*trace[i]["pos"], c="cyan", s=50, edgecolors="k", zorder=5)
        ax.scatter(*res["truth"]["eve_pos"], marker="*", c="gold", s=200, edgecolors="k", zorder=6)
        b = trace[i]["belief"]
        ax.set_title(f"step {trace[i]['step']}  H={b['entropy']:.2f}  conf={b['confidence']:.2f}", fontsize=10)
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle("Belief over EVE's location (cyan = WALL-E, star = EVE truth)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_timeline(res, path):
    tr = res["trace"]
    steps = [t["step"] for t in tr]
    has_b = "entropy" in tr[0]["belief"]
    fig, axes = plt.subplots(2 if has_b else 1, 1, figsize=(11, 6 if has_b else 3.5), sharex=True)
    axes = np.atleast_1d(axes)
    k = 0
    if has_b:
        axes[0].plot(steps, [t["belief"]["entropy"] for t in tr], c="purple")
        axes[0].set_ylabel("belief entropy (nats)")
        k = 1
    ax = axes[k]
    ax.plot(steps, [t["battery"] for t in tr], c="tab:green")
    ax.set_ylabel("battery")
    ax.set_xlabel("step")
    for a in axes:
        for t in tr:
            if t["action"]["type"] == "recharge":
                a.axvspan(t["step"] - 0.5, t["step"] + 0.5, color="orange", alpha=0.25)
            if t["strategy_change"]:
                a.axvline(t["step"], color="crimson", ls="--", lw=0.8)
    for t in tr:
        if t["action"]["type"] == "scan":
            ax.scatter(t["step"], t["battery"], marker="o", c="k", s=14, zorder=5)
        if t["observation"] and t["observation"]["clues"]:
            ax.scatter(t["step"], t["battery"], marker="^", c="lime", edgecolors="k", s=70, zorder=6)
        if t["observation"] and t["observation"]["detected"]:
            ax.scatter(t["step"], t["battery"], marker="*", c="gold", edgecolors="k", s=200, zorder=7)
    axes[0].set_title("Timeline: orange = recharge, dashed red = strategy change, black dot = scan, "
                      "green triangle = clue, star = detection", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
