"""Figures of Chapter 18."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, LEARNED_TEXT, MUTED, INK

# 1. System identification: one variable, one factor per transition.
fig = Figure(860, 330)
fig.variable("theta", (430, 80), "θ_p", "parameter", radius=28)
labels = ["(x, u, x′)^{(1)}", "(x, u, x′)^{(2)}", "(x, u, x′)^{(3)}", None,
          "(x, u, x′)^{(M)}"]
for i, x in enumerate([130, 280, 430, 580, 730]):
    if labels[i] is None:
        fig.text(x, 205, "…", EDGE, "middle", 24)
        continue
    fig.factor((x, 200), "sampled", ["theta"], labels[i], "below")
fig.text(480, 62, "the parameters (F, B) of the dynamics: the unknown", EDGE,
         "start", 12)
fig.text(430, 258, "one factor per observed transition, with error x′ − F x − B u",
         EDGE, "middle", 12)
fig.save("SystemIdGraph.svg", "System identification as a factor graph",
         "One variable, the parameters theta p of the dynamics, joined to one "
         "factor for each observed transition. Each factor holds the error of "
         "predicting the next state of that transition.")

# 2. The control graph on the learned factor: theta_p joins every dynamics
#    factor, as theta joined every policy factor in Chapter 4.
fig = Figure(900, 390)
for t, x in enumerate([100, 400, 700]):
    fig.variable(f"x{t}", (x, 150), f"x_{t}", "state")
for t, x in enumerate([250, 550]):
    fig.variable(f"u{t}", (x, 280), f"u_{t}", "action")
fig.variable("theta", (400, 40), "θ_p", "parameter")
fig.factor((100, 70), "probability", ["x0"], "p(x_0)", "right")
for t, x in enumerate([250, 550]):
    fig.curve("theta", (x, 60), (x, 150))
    fig.factor((x, 150), "learned", [f"x{t}", f"u{t}", f"x{t + 1}"],
               f"p(x_{t + 1} | x_{t}, u_{t}; θ_p)", offset=(12, 24),
               where="start")
    fig.factor((x - 150, 280), "value", [f"x{t}", f"u{t}"],
               f"r(x_{t}, u_{t})", "below")
fig.factor((790, 150), "value", ["x2"], "r(x_2)", "above")
fig.text(445, 30, "one unknown, shared by every dynamics factor", EDGE,
         "start", 12)
fig.save("LearnedDynamicsGraph.svg",
         "The control graph with a learned dynamics factor",
         "The decision graph of the two-move line problem. Its dynamics "
         "factors are drawn in teal: they are learned. Both depend on the "
         "same parameter theta p, a variable joined to every dynamics factor.")
