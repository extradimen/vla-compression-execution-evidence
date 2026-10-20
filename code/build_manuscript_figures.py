"""Render manuscript figures from frozen project evidence; no estimation here."""

from pathlib import Path
import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
analysis = json.loads((ROOT / "data/frozen_analysis.json").read_text())
arms = analysis["arms"]

# Consistent Okabe--Ito semantics across the manuscript: BF16/reference and
# backbone are blue; packed/development routes and complete-process cost are
# orange; the offline target is green. Labels, shapes and hatches repeat all
# essential distinctions so colour alone is never necessary.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
INK = "#243447"
GRID = "#D9E1E8"
PALE_BLUE = "#E8F3F8"
PALE_ORANGE = "#FFF3E1"
PALE_GREEN = "#EAF5EE"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})

# Two aligned pathways within one figure: a final closed loop and a separate
# development-only preparation path. Text extents are checked below.
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig0, ax0 = plt.subplots(figsize=(8.3, 5.0))
ax0.set(xlim=(0, 12), ylim=(0, 6.90))
ax0.axis("off")
box_texts = []

def lane(y, h, header_y, header_color, title):
    ax0.add_patch(FancyBboxPatch((0.24, y), 11.52, h,
                                 boxstyle="round,pad=0.035", linewidth=0.8,
                                 edgecolor="#A8BAC6", facecolor="#FCFDFE", zorder=0))
    ax0.add_patch(Rectangle((0.28, header_y), 11.44, 0.34,
                            linewidth=0, facecolor=header_color, zorder=1))
    ax0.text(0.52, header_y + 0.17, title, ha="left", va="center",
             fontsize=8.8, fontweight="semibold", color=INK, zorder=2)

def box(x, y, w, h, label, color, fontsize=8.1):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                 linewidth=0.9, edgecolor=INK, facecolor=color))
    txt = ax0.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                   fontsize=fontsize, linespacing=1.2)
    box_texts.append((txt, x, y, w, h))

def arrow(x1, y1, x2, y2, dashed=False):
    ax0.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color=INK,
                                 linestyle="--" if dashed else "-"))

lane(2.95, 3.70, 6.24, PALE_BLUE, "Independent final experiment: synchronous closed loop")
lane(0.12, 2.23, 2.00, PALE_ORANGE, "Development diagnostic: offline preparation")
upper = [
    (0.55, "LIBERO-spatial\n8 tasks / 20 states\n3 seeds", PALE_BLUE),
    (2.875, "Image, instruction\nand robot state", "#F4F7F9"),
    (5.20, "Frozen VLA\nperception\nbackbone", PALE_BLUE),
    (7.525, "Action head\nD=4 denoising\nsteps", PALE_BLUE),
    (9.85, "Execute E=1 or 4\nactions; reobserve", PALE_BLUE),
]
for x, label, color in upper:
    box(x, 5.05, 1.65, 0.93, label, color, fontsize=7.8)
for a, b in zip(upper, upper[1:]):
    arrow(a[0] + 1.73, 5.515, b[0] - 0.08, 5.515)
box(2.5, 4.07, 7.0, 0.58,
    "Precision inside the same frozen policy:\nU = BF16   |   P4 = rotated packed W4 / dynamic A8", PALE_ORANGE)
arrow(6.0, 4.99, 6.0, 4.70, dashed=True)
arrow(8.2, 4.99, 8.2, 4.70, dashed=True)
box(0.55, 3.10, 10.9, 0.38,
    "Final evidence: success | all-attempt episode time | GPU cost | latency | allocator peak",
    PALE_BLUE, fontsize=8.2)
arrow(10.675, 4.99, 10.675, 3.55)
ax0.text(0.55, 3.73, "Loop: reobserve until success, failure, or step limit",
         fontsize=7.9, color=INK)

# The final-grid finding motivates, but does not validate, the later diagnostic.
arrow(6.0, 3.08, 6.0, 2.39)
ax0.text(6.35, 2.67, "Final finding motivates mechanism study; not confirmation",
         fontsize=7.6, color=INK, va="center")

box(0.55, 0.70, 3.0, 0.98, "GPU producer converts\n84 backbone modules", PALE_ORANGE)
box(4.50, 0.70, 3.0, 0.98,
    "SHA-verified bundle\nweights / scales / rotations / bias", PALE_ORANGE)
box(8.45, 0.70, 3.0, 0.98,
    "CPU assembles final model\nthen moves it to GPU", PALE_GREEN)
arrow(3.63, 1.19, 4.42, 1.19)
arrow(7.58, 1.19, 8.37, 1.19)
ax0.text(0.55, 0.32, "Producer cost and memory peak are reported separately from target inference.",
         fontsize=8.0, color=INK)
fig0.tight_layout(pad=0.35)
fig0.canvas.draw()
renderer = fig0.canvas.get_renderer()
for txt, x, y, w, h in box_texts:
    label_bounds = txt.get_window_extent(renderer)
    lo = ax0.transData.transform((x, y))
    hi = ax0.transData.transform((x + w, y + h))
    if (label_bounds.x0 < lo[0] or label_bounds.x1 > hi[0]
            or label_bounds.y0 < lo[1] or label_bounds.y1 > hi[1]):
        raise ValueError(f"Figure 1 box overflow: {txt.get_text()!r}")
fig0.savefig(HERE.parent / "figures/figure0.pdf")
fig0.savefig(HERE.parent / "figures/figure0.svg")
fig0.savefig(HERE.parent / "figures/figure0.png", dpi=320)
plt.close(fig0)

# Frozen paired effects: both series are P4-minus-BF16 contrasts, so the two
# execution settings use warm hues plus different markers. The input already
# contains task-level effects and the frozen cluster-bootstrap intervals.
fig1, axs1 = plt.subplots(1, 2, figsize=(11, 3.2), constrained_layout=True)
contrast_keys = ["compression_at_E1", "compression_at_E4"]
contrast_labels = ["P4 - BF16, E1", "P4 - BF16, E4"]
contrast_colors = [BLUE, ORANGE]
contrast_markers = ["o", "s"]
paired_panels = [
    ("episode_wall_seconds_all_attempts_difference", "Episode wall-time difference (s)", 1),
    ("success_rate_difference", "Success-rate difference (percentage points)", 100),
]
for ax, (metric, xlabel, multiplier) in zip(axs1, paired_panels):
    ax.axvline(0, color=INK, linewidth=0.9)
    for i, (key, label, color, marker) in enumerate(zip(
            contrast_keys, contrast_labels, contrast_colors, contrast_markers)):
        values = analysis["paired_contrasts"][key][metric]
        task = np.asarray(values["per_task"]) * multiplier
        offsets = np.linspace(-0.13, 0.13, len(task))
        ax.scatter(task, np.full(len(task), i) + offsets, s=22, alpha=0.55,
                   color=color, marker=marker, zorder=2)
        lo, hi = np.asarray(values["task_cluster_percentile_95ci"]) * multiplier
        estimate = values["estimate"] * multiplier
        ax.errorbar(estimate, i, xerr=[[estimate - lo], [hi - estimate]],
                    fmt=marker, color=color, capsize=5, markersize=9,
                    markerfacecolor="white" if i else color,
                    markeredgewidth=1.5, linewidth=2, zorder=4)
    ax.set_yticks(range(2), contrast_labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
axs1[0].set_title("A. Completion time")
axs1[1].set_title("B. Task success")
fig1.suptitle("Effect of compression under two execution settings", fontweight="bold")
for suffix in ("pdf", "svg"):
    fig1.savefig(HERE.parent / f"figures/figure1.{suffix}", bbox_inches="tight")
fig1.savefig(HERE.parent / "figures/figure1.png", dpi=320, bbox_inches="tight")
plt.close(fig1)

# The separate development map reads the already archived review-pack table;
# plotting here changes presentation only, without selecting or re-estimating.
with (ROOT / "data/table-3-development-diagnostics.csv").open(
        newline="") as source:
    development_rows = list(csv.DictReader(source))
fig2, ax2 = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
for i, row in enumerate(development_rows):
    name = row["configuration"]
    latency = float(row["steady_inference_s_per_call"])
    peak = float(row["full_target_gpu_allocator_peak_gib"])
    color = BLUE if name == "BF16" else GREEN if name == "Offline backbone" else ORANGE
    marker = "s" if name == "BF16" else "D" if name == "Offline backbone" else "o"
    ax2.scatter(peak, latency, color=color, marker=marker, s=85,
                edgecolor=INK, linewidth=0.55, zorder=3)
    ax2.annotate(name, (peak, latency), xytext=(5, 3 if i % 2 else -11),
                 textcoords="offset points", fontsize=9)
bf16_row = next(row for row in development_rows if row["configuration"] == "BF16")
ax2.axvline(float(bf16_row["full_target_gpu_allocator_peak_gib"]),
            color=INK, linestyle="--", linewidth=0.9, alpha=0.65)
ax2.axhline(float(bf16_row["steady_inference_s_per_call"]),
            color=INK, linestyle="--", linewidth=0.9, alpha=0.65)
ax2.set(xlabel="Full target-process GPU allocator peak (GiB)",
        ylabel="Steady inference per call (s)",
        title="Development diagnosis: memory and latency are distinct objectives")
ax2.grid(color=GRID, linewidth=0.55)
ax2.set_axisbelow(True)
for suffix in ("pdf", "svg"):
    fig2.savefig(HERE.parent / f"figures/figure2.{suffix}", bbox_inches="tight")
fig2.savefig(HERE.parent / "figures/figure2.png", dpi=320, bbox_inches="tight")
plt.close(fig2)

labels = ["U / E1", "P4 / E1", "U / E4", "P4 / E4"]
keys = ["U_E1", "P4_E1", "U_E4", "P4_E4"]
success = [100 * arms[k]["task_equal_success_rate"] for k in keys]
wall = [arms[k]["episode_wall_seconds_all_attempts"]["mean"] for k in keys]
cost = [arms[k]["controller_final_gpu_seconds_per_success"] for k in keys]
latency = [1000 * arms[k]["inference_latency_seconds"]["mean"] for k in keys]

fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.6), constrained_layout=True)
metrics = [
    (success, "Success (%)", (0, 100), "A"),
    (wall, "All-attempt episode time (s)", (0, 85), "B"),
    (cost, "Controlled GPU seconds / success", (0, 95), "C"),
    (latency, "Mean inference latency (ms/call)", (0, 630), "D"),
]
colors = [BLUE, ORANGE, BLUE, ORANGE]
hatches = ["", "//", "", "//"]
for ax, (vals, ylabel, ylim, panel) in zip(axes.flat, metrics):
    x = np.arange(4)
    bars = ax.bar(x, vals, width=0.63, color=colors, edgecolor=INK, linewidth=0.55)
    for b, hatch in zip(bars, hatches):
        b.set_hatch(hatch)
    ax.set_ylim(*ylim)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(panel, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(i, v + ylim[1] * 0.025, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
fig.savefig(HERE.parent / "figures/figure3.pdf")
fig.savefig(HERE.parent / "figures/figure3.svg")
fig.savefig(HERE.parent / "figures/figure3.png", dpi=320)
plt.close(fig)

# Descriptive module attribution from one development trace. The three panels
# are deliberately not added together: module counts, calls, nominal bytes and
# inclusive CPU scopes answer different engineering questions.
attribution = json.loads((ROOT / "data/module-attribution.json").read_text())
groups = attribution["groups"]
group_keys = ["backbone", "action_head"]
group_labels = ["Backbone (84 modules)", "Action head (32 modules)"]
module_values = [groups[k]["modules"] for k in group_keys]
call_values = [groups[k]["profiled_calls"] for k in group_keys]
saved_mib = [groups[k]["nominal_bytes_saved_vs_bf16_weight"] / 2**20 for k in group_keys]
cpu_ms = [groups[k]["module_cpu_total_scope_ms"] for k in group_keys]

fig4, axs4 = plt.subplots(1, 3, figsize=(8.1, 3.15), constrained_layout=True)
panels4 = [
    (call_values, "Profiled calls", "A", "calls"),
    (saved_mib, "Nominal weight saving (MiB)", "B", "MiB"),
    (cpu_ms, "Inclusive CPU scope (ms)", "C", "ms"),
]
for ax, (vals, xlabel, panel, unit) in zip(axs4, panels4):
    y = np.arange(2)
    bars4 = ax.barh(y, vals, color=[BLUE, ORANGE], height=0.56,
                    edgecolor=INK, linewidth=0.6)
    bars4[1].set_hatch("//")
    ax.set_yticks(y, group_labels if ax is axs4[0] else ["", ""])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_xlabel(xlabel)
    ax.set_title(panel, loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for yi, v in enumerate(vals):
        ax.text(v + max(vals) * 0.025, yi, f"{v:.1f}" if unit != "calls" else str(v),
                va="center", fontsize=8)
for suffix in ("pdf", "svg"):
    fig4.savefig(HERE.parent / f"figures/figure4.{suffix}")
fig4.savefig(HERE.parent / "figures/figure4.png", dpi=320)
plt.close(fig4)

# The complete target-process peak and steady inference peak use different
# measurement windows. The offline producer is a separate process and is
# therefore annotated, never folded into the target-process bars.
lifecycle = json.loads((ROOT / "data/memory-lifecycle.json").read_text())
life_rows = {r["name"]: r for r in lifecycle["rows"]}
offline = json.loads((ROOT / "data/offline-target-summary.json").read_text())
producer = json.loads((ROOT / "data/offline-producer-result.json").read_text())
life_order = ["BF16", "P4_original", "backbone_only", "offline_target"]
life_labels = ["BF16", "Full P4\non-device", "Backbone-only\non-device", "Offline bundle\ntarget"]
full_peaks = [life_rows[k]["lifecycle_peak_gib"] for k in life_order[:-1]] + [offline["gpu_peak_allocated_bytes"] / 2**30]
infer_peaks = [life_rows[k]["inference_peak_gib"] for k in life_order[:-1]] + [offline["inference_only_peak_allocated_bytes"] / 2**30]
fig5, ax5 = plt.subplots(figsize=(8.1, 3.75), constrained_layout=True)
x5 = np.arange(4)
w5 = 0.34
b1 = ax5.bar(x5 - w5/2, infer_peaks, w5, label="Inference window", color=BLUE,
             edgecolor=INK, linewidth=0.6)
b2 = ax5.bar(x5 + w5/2, full_peaks, w5, label="Complete target process", color=ORANGE,
             edgecolor=INK, linewidth=0.6, hatch="//")
for bars in (b1, b2):
    for b in bars:
        ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.12,
                 f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.6)
ax5.set_xticks(x5, life_labels)
ax5.set_ylim(0, 12)
ax5.set_ylabel("GPU allocator peak (GiB)")
ax5.set_title("Target-process memory depends on preparation boundary", loc="left", fontsize=10)
ax5.spines[["top", "right"]].set_visible(False)
ax5.grid(axis="y", color=GRID, linewidth=0.55)
ax5.set_axisbelow(True)
ax5.legend(ncol=2, loc="upper right", frameon=False, fontsize=8)
producer_peak_gib = producer["gpu_peak_allocated_bytes"] / 2**30
for suffix in ("pdf", "svg"):
    fig5.savefig(HERE.parent / f"figures/figure5.{suffix}")
fig5.savefig(HERE.parent / "figures/figure5.png", dpi=320)
plt.close(fig5)

manifest = {
    "sources": ["outputs/quantvla-packed-w4a8-final-analysis-v1.json",
                "data/table-3-development-diagnostics.csv",
                "data/module-attribution.json",
                "data/memory-lifecycle.json",
                "data/offline-target-summary.json",
                "data/offline-producer-result.json"],
    "transformations": "Figures 1 and 3 use frozen paired contrasts and direct arm means; Figure 2 reads the archived development table. Figure 3 multiplies success fractions by 100. No new inference.",
    "palette": {"reference_BF16_backbone": BLUE, "packed_or_development": ORANGE,
                "paired_E1_effect": BLUE, "paired_E4_effect": ORANGE,
                "offline_target": GREEN,
                "ink": INK, "grid": GRID},
    "panels": ["success", "all-attempt wall time", "controlled GPU seconds per success", "mean inference latency"],
    "figure0": "study pathway diagram, no quantitative inference",
    "figure4": "one development trace: calls, nominal module weight savings, inclusive CPU scope; no causal timing decomposition",
    "figure5": f"one development condition: direct allocator peaks by measurement window; offline producer separate at {producer_peak_gib:.3f} GiB",
}
(HERE.parent / "figures/figure-provenance-v2.json").write_text(json.dumps(manifest, indent=2) + "\n")
