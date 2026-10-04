"""Figures of Chapter 23."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, chain, EDGE, INK, LEARNED_TEXT

# 1. The graph: reward factors with parameters.
fig = Figure(900, 450)
where = chain(fig, 2, y_state=120, y_action=250, policy=False, reward=False,
              final=False)
for t in range(2):
    centre = (100 + 300 * t, 250)
    where[f"reward{t}"] = centre
    fig.factor(centre, "learned", [f"s{t}", f"a{t}"],
               f"r(s_{t}, a_{t}; θ_r)", offset=(0, -16), where="end")
where["final"] = (790, 120)
fig.factor(where["final"], "learned", ["s2"], "r(s_2; θ_r)", "above")
fig.variable("theta", (450, 360), "θ_r", "parameter")
fig.curve("theta", (150, 370), where["reward0"])
fig.curve("theta", (410, 300), where["reward1"])
fig.curve("theta", (800, 370), where["final"])
fig.save("InverseGraph.svg", "Reward factors with parameters",
         "The decision graph of the track without policy factors. The two "
         "reward factors of the moves and the final reward factor are drawn "
         "in teal: they are learned. All three are joined to one parameter "
         "node, theta r.")

# 2. The loop of inverse reinforcement learning.
fig = Figure(900, 330, legend=False)
fig.box(40, 40, 330, 105, "Demonstrations of the expert", EDGE)
fig.text(205, 100, "count the features along each trajectory", INK,
         "middle", 12)
fig.text(205, 122, "→ counted features", EDGE, "middle", 12)
fig.box(40, 185, 330, 105, "Stage 1: eliminate, with θ_r fixed", EDGE)
fig.text(205, 240, "backward: soft action values, soft-optimal policy", INK,
         "middle", 12)
fig.text(205, 258, "forward: visitations d_t", INK, "middle", 12)
fig.text(205, 278, "→ expected features of the model", EDGE, "middle", 12)
fig.box(560, 110, 300, 110, "Stage 2: update θ_r", "#6639ba")
fig.text(710, 165, "θ_r ← θ_r + α (counted − expected)", INK, "middle", 13)
fig.text(710, 190, "until the features match", EDGE, "middle", 12)
fig.arrow((370, 95), (560, 145), gap=4, color=EDGE)
fig.arrow((370, 225), (560, 180), gap=4, color=EDGE)
fig.arrow((700, 222), (372, 266), bend=0.22, gap=4, color="#6639ba")
fig.text(560, 296, "new reward factors", "#6639ba", "middle", 12)
fig.save("InverseLoop.svg", "The loop of inverse reinforcement learning",
         "Three boxes. The demonstrations give counted features. Stage 1 "
         "eliminates the graph with the reward parameters fixed and gives the "
         "expected features of the model. Stage 2 moves the reward parameters "
         "by the difference of the two and passes new reward factors back to "
         "Stage 1.")
