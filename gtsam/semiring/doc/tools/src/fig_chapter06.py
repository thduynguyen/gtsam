"""Figures of Chapter 6."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, VALUE_TEXT, LEARNED_TEXT, INK

# The LQG graph: observations above, actions below, and the two passes.
fig = Figure(900, 500)
xs = [130, 430, 730]
for t, x in enumerate(xs):
    fig.variable(f"x{t}", (x, 210), f"x_{t}", "state")
for t, x in enumerate(xs[:2]):
    fig.variable(f"y{t}", (x, 95), f"y_{t}", "observation")
    fig.variable(f"u{t}", (x + 150, 335), f"u_{t}", "action")
fig.factor((45, 210), "probability", ["x0"], "p(x_0)", "above")
for t, x in enumerate(xs[:2]):
    fig.factor((x, 152), "probability", [f"x{t}", f"y{t}"],
               f"p(y_{t} | x_{t})", "right")
    fig.factor((x + 150, 210), "probability", [f"x{t}", f"u{t}", f"x{t + 1}"],
               f"p(x_{t + 1} | x_{t}, u_{t})", "above")
    fig.factor((x, 335), "value", [f"x{t}", f"u{t}"], f"r(x_{t}, u_{t})",
               "below")
fig.factor((820, 210), "value", ["x2"], "r(x_2)", "above")
fig.arrow((130, 38), (760, 38), gap=0, color=LEARNED_TEXT)
fig.text(445, 28, "forward pass, probability channel: the Kalman filter "
         "gives the estimate of x_t", LEARNED_TEXT, "middle", 12, bold=True)
fig.arrow((760, 410), (130, 410), gap=0, color=VALUE_TEXT)
fig.text(445, 430, "backward pass, value channel: the Riccati recursion "
         "gives the gain K_t", VALUE_TEXT, "middle", 12, bold=True)
fig.save("LqgGraph.svg", "The factor graph of an LQG problem",
         "A chain of three states with two actions below it and two "
         "observations above it. Each observation is joined to its state by "
         "an observation factor. A forward arrow above the graph marks the "
         "Kalman filter, computed on the probability channel, and a backward "
         "arrow below the graph marks the Riccati recursion, computed on the "
         "value channel.")
