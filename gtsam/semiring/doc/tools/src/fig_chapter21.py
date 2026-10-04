"""Figures of Chapter 21."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, LEARNED_TEXT, MUTED, RED

# 1. Planning in a learned model, over a short horizon, with a learned value
#    at its end.
fig = Figure(900, 400)
for t, x in enumerate([100, 400, 700]):
    fig.variable(f"x{t}", (x, 140), f"x_{t}", "state")
for t, x in enumerate([250, 550]):
    fig.variable(f"u{t}", (x, 270), f"u_{t}", "action")
    fig.factor((x, 140), "learned", [f"x{t}", f"u{t}", f"x{t + 1}"],
               "learned dynamics", "above")
    fig.factor((x - 150, 270), "learned", [f"x{t}", f"u{t}"],
               "reward", "below")
    fig.text(x, 316, "max", RED, "middle", 12, bold=True)
fig.factor((800, 140), "learned", ["x2"], "learned value V(x_2)", "below")
fig.text(100, 96, "the real state, now", INK, "middle", 12)
fig.box(40, 40, 820, 300, "Stage 1 in the model: plan h moves ahead, "
        "apply u_0, observe the real x_1, plan again", EDGE)
fig.save("PlanningInLearnedModel.svg", "Planning in a learned model",
         "A short chain of two moves starting from the real current state. "
         "Its dynamics and reward factors are learned, and a learned value "
         "factor closes the chain at its end. The actions are chosen by a "
         "maximum, the first action is applied, and the plan is made again "
         "from the next real state.")

# 2. Short imagined rollouts branching from real states.
fig = Figure(900, 380)
for t, x in enumerate([90, 300, 510, 720]):
    fig.variable(f"x{t}", (x, 90), f"x_{t}", "state")
for t, x in enumerate([195, 405, 615]):
    fig.factor((x, 90), "sampled", [f"x{t}", f"x{t + 1}"])
fig.text(810, 95, "…", EDGE, "middle", 22)
fig.text(405, 46, "real transitions, stored in the buffer", INK, "middle", 12)
for name, x0 in [("a", 90), ("b", 510)]:
    root = "x0" if name == "a" else "x2"
    fig.variable(f"{name}1", (x0 + 130, 230), "x̂", "state", radius=20)
    fig.variable(f"{name}2", (x0 + 310, 230), "x̂", "state", radius=20)
    fig.factor((x0 + 50, 160), "learned", [root, f"{name}1"])
    fig.factor((x0 + 220, 230), "learned", [f"{name}1", f"{name}2"])
fig.text(445, 290, "imagined: h = 2 steps of the learned model, "
         "started from a real state", LEARNED_TEXT, "middle", 12)
fig.save("ImaginedRollouts.svg", "Short imagined rollouts from real states",
         "A chain of real states joined by real transitions from the "
         "buffer. From two of the real states, a short chain of two imagined "
         "states branches off, joined by learned dynamics factors.")
