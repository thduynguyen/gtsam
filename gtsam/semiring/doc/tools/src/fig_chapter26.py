"""Figures of Chapter 26."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED

# The graph of simulation training with a randomized model.
fig = Figure(900, 520)
ys, ya = 190, 320
for t, x in enumerate([110, 410, 710]):
    fig.variable(f"x{t}", (x, ys), f"x_{t}", "state")
for t, x in enumerate([260, 560]):
    fig.variable(f"u{t}", (x, ya), f"u_{t}", "action")
fig.variable("model", (410, 60), "θ_p", "parameter")
fig.variable("theta", (335, 430), "θ", "parameter")
fig.factor((30, ys), "probability", ["x0"], "p(x_0)", "above")
fig.factor((540, 60), "probability", ["model"],
           "p(θ_p): a new model for every episode", "right")
where = {}
for t, x in enumerate([260, 560]):
    fig.factor((x, ys), "sampled", [f"x{t}", f"u{t}", f"x{t + 1}"],
               f"p(x_{t + 1} | x_{t}, u_{t}; θ_p)", offset=(0, 30),
               where="start")
    fig.edge("model", (x, ys))
    where[t] = (x - 75, (ys + ya) / 2)
    fig.factor(where[t], "probability", [f"x{t}", f"u{t}"],
               f"π_θ(u_{t} | x_{t})", offset=(-14, 22), where="end")
    fig.factor((x - 150, ya), "value", [f"x{t}", f"u{t}"],
               f"r(x_{t}, u_{t})", "below")
fig.factor((800, ys), "value", ["x2"], "r(x_2)", "above")
fig.curve("theta", (160, 410), where[0])
fig.curve("theta", (460, 400), where[1])
fig.text(375, 452, "one policy for all the models", EDGE, "start", 12)
fig.save("RandomizedSimulation.svg",
         "Training in a simulator with a randomized model",
         "The chain of states and actions with rewards. The dynamics factors "
         "are white: they are known only through simulator samples. Both are "
         "joined to a model parameter node that has a prior and is drawn "
         "anew for every episode. The policy factors are joined to one policy "
         "parameter node, shared by all models.")
