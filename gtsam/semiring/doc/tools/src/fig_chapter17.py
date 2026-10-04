"""Figures of Chapter 17."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import (Figure, MUTED, EDGE, RED, INK, VALUE_TEXT, LEARNED_TEXT)

PURPLE = "#6639ba"

# 1. The bucket of the action, before and after the soft elimination.
fig = Figure(1010, 400)
# Left: the bucket.
fig.variable("s", (110, 120), "s", "state")
fig.variable("a", (300, 250), "a", "action")
fig.variable("theta", (110, 300), "θ", "parameter")
fig.factor((205, 185), "probability", ["s", "a", "theta"],
           "π_θ(a | s)", offset=(16, -10), where="start")
fig.factor((300, 120), "learned", ["s", "a"], "(1, Q̂(s, a))", "above")
fig.ring("a")
fig.text(205, 60, "the bucket of the action", EDGE, "middle", 12, bold=True)
# Right: the result.
fig.variable("s2", (560, 120), "s", "state")
fig.variable("a2", (750, 250), "a", "done")
fig.arrow("s2", "a2")
fig.factor((470, 120), "learned", ["s2"], "(1, soft max of Q̂)", "above")
fig.text(640, 60, "after eliminating it by soft maximum", EDGE, "middle", 12,
         bold=True)
fig.text(690, 170, "conditional:", INK, "start", 12)
fig.text(690, 188, "(π_θ(a | s), Q̂ − soft max)", INK, "start", 12)
fig.text(560, 300, "tilted policy  q(a | s) = π_θ(a | s) · exp(soft advantage / η)",
         RED, "start", 12, bold=True)
fig.arrow((410, 190), (470, 190), gap=0, color=EDGE)
fig.save("TiltedConditional.svg",
         "Eliminating the action by soft maximum",
         "On the left, the bucket of the action: the policy factor, joined to "
         "the parameter theta, and a learned value factor holding Q. On the "
         "right, the result of eliminating the action by soft maximum: a "
         "value factor on the state holding the soft maximum of Q, and a "
         "conditional on the action whose value channel is the soft "
         "advantage. Reweighting the policy by the exponential of the soft "
         "advantage divided by the temperature gives the tilted policy q.")

# 2. The loop: E-step and M-step.
fig = Figure(900, 320, legend=False)
boxes = [
    (40, 90, 230, "Stage 1, part 1: evaluate", ["backward message Q̂", "of the current policy π_θ,", "exact or learned"]),
    (330, 90, 230, "Stage 1, part 2: tilt (E-step)", ["eliminate a by soft maximum:", "q(a | s) ∝ π_θ(a | s) e^{Â/η}", "a target policy, not parametric"]),
    (620, 90, 240, "Stage 2: fit (M-step)", ["weighted maximum likelihood:", "θ maximizes the average of", "q(a | s) log π_θ(a | s)"]),
]
for x, y, w, title, lines in boxes:
    fig.box(x, y, w, 120, title, EDGE)
    for i, line in enumerate(lines):
        fig.text(x + w / 2, y + 52 + 18 * i, line, INK, "middle", 12)
fig.arrow((270, 150), (330, 150), gap=3, color=EDGE)
fig.arrow((560, 150), (620, 150), gap=3, color=EDGE)
fig.arrow((740, 210), (155, 210), bend=0.16, gap=3, color=PURPLE)
fig.text(450, 296, "the fitted policy is evaluated next", PURPLE, "middle", 12)
fig.text(450, 50, "Policy improvement as expectation-maximization", INK,
         "middle", 14, bold=True)
fig.save("EmLoop.svg", "The E-step and the M-step as the two stages",
         "Three boxes in a loop. Stage 1 first evaluates the current policy, "
         "giving the backward message Q, and then eliminates the action by "
         "soft maximum, giving the tilted policy q. Stage 2 fits the "
         "parametric policy to q by weighted maximum likelihood. The fitted "
         "policy is evaluated next.")
