"""Figure of Chapter 14."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED, RED, LEARNED_TEXT, VALUE_TEXT

PURPLE = "#6639ba"
fig = Figure(920, 400, legend=False)
fig.box(30, 50, 300, 290, "Stage 1, run once at θ_{old}", EDGE)
fig.text(180, 120, "forward message d_{old}", LEARNED_TEXT, "middle", 13,
         bold=True)
fig.text(180, 142, "where the old policy goes", EDGE, "middle", 12)
fig.text(180, 200, "backward message A_{old}", VALUE_TEXT, "middle", 13,
         bold=True)
fig.text(180, 222, "how good each action is,", EDGE, "middle", 12)
fig.text(180, 238, "under the old policy", EDGE, "middle", 12)
fig.text(180, 300, "both describe π_{old}, not π_θ", INK, "middle", 12)
fig.arrow((330, 195), (430, 195), gap=4, color=EDGE)
fig.text(380, 180, "reused", EDGE, "middle", 12)

# The space of parameters, with a trust region around theta_old.
fig.box(430, 50, 460, 290, "Stage 2: several updates of θ from the same messages",
        PURPLE)
cx, cy = 600, 215
fig.raw(f'<ellipse cx="{cx}" cy="{cy}" rx="125" ry="85" fill="#f3ecff" '
        f'stroke="{PURPLE}" stroke-width="2" stroke-dasharray="6 4"/>')
fig.text(cx, cy + 110, "trust region: KL(π_{old} ‖ π_θ) ≤ D_{max}", PURPLE,
         "middle", 12, bold=True)
points = [(cx - 70, cy - 5), (cx - 38, cy + 14), (cx - 2, cy + 26),
          (cx + 36, cy + 30), (cx + 74, cy + 26)]
for p, q in zip(points[:-1], points[1:]):
    fig.arrow(p, q, gap=5, color=PURPLE)
for i, p in enumerate(points):
    fig.raw(f'<circle cx="{p[0]}" cy="{p[1]}" r="5" fill="{PURPLE}"/>')
fig.text(points[0][0] - 12, points[0][1] - 12, "θ_{old}", INK, "end", 13,
         bold=True)
fig.text(points[-1][0] + 12, points[-1][1] + 5, "θ_{new}", INK, "start", 13,
         bold=True)
fig.arrow(points[0], (cx + 235, cy - 120), gap=6, color=RED, dashed=True)
fig.text(cx + 275, cy - 135, "one large step", RED, "end", 12, bold=True)
fig.text(cx + 278, cy - 60, "outside, the old", RED, "end", 12)
fig.text(cx + 278, cy - 44, "messages are stale", RED, "end", 12)
fig.text(cx + 278, cy + 50, "at θ_{new}:", INK, "end", 12)
fig.text(cx + 278, cy + 66, "run Stage 1 again", INK, "end", 12)
fig.save("TrustRegion.svg", "Reusing messages inside a trust region",
         "On the left, Stage 1 is run once at theta old and produces a "
         "forward and a backward message. On the right, Stage 2 makes "
         "several small updates of theta from those same messages, all "
         "inside a dashed region around theta old where the KL divergence "
         "from the old policy is at most D max. A long dashed red arrow "
         "shows one large step that leaves the region, where the old "
         "messages are stale. At theta new, Stage 1 is run again.")
