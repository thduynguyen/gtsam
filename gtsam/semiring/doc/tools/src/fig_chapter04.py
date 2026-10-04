"""Figures of Chapter 4."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, chain, MUTED, EDGE, RED

# 1. The chain without policy factors: which sum each variable gets.
fig = Figure(900, 400)
where = chain(fig, 2, y_state=130, y_action=270, policy=False)
for name, order in [("s2", 1), ("s1", 3), ("s0", 5)]:
    x, y = fig.pos[name]
    fig.text(x, y - 36, f"{order}. average", EDGE, "middle", 12, bold=True)
for name, order in [("a1", 2), ("a0", 4)]:
    x, y = fig.pos[name]
    fig.text(x, y + 46, f"{order}. max", RED, "middle", 12, bold=True)
fig.save("DecisionGraph.svg", "The decision graph of the two-move problem",
         "The factor graph of Chapter 1 without its policy factors. The "
         "states are eliminated by averaging and the actions by maximizing, "
         "in the order last state, last action, middle state, first action, "
         "first state.")

# 2. A shared parameter joins all policy factors.
fig = Figure(900, 430)
where = chain(fig, 2, y_state=120, y_action=250,
              policy_label="π_θ(a_{t} | s_{t})")
fig.variable("theta", (325, 350), "θ", "parameter")
fig.curve("theta", (150, 330), where["policy0"])
fig.curve("theta", (450, 330), where["policy1"])
fig.text(368, 356, "one decision, shared by every policy factor", EDGE,
         "start", 12)
fig.save("GlobalDecision.svg", "A policy parameter as one global decision",
         "The factor graph of Chapter 1, in which each policy factor depends "
         "on a parameter theta. Theta is one extra variable joined to every "
         "policy factor.")
