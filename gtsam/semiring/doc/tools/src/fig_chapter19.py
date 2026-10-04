"""Figures of Chapter 19."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, LEARNED_TEXT, MUTED, INK, RED, VALUE_TEXT

# 1. The hill: height cos(x), so the slope pushes with sin(x).
fig = Figure(860, 300, legend=False)
px = lambda x: 430 + 88 * x
py = lambda x: 150 - 55 * math.cos(x)
points = " ".join(f"{px(-4.4 + 0.1 * i):.1f},{py(-4.4 + 0.1 * i):.1f}"
                  for i in range(89))
fig.raw(f'<polyline points="{points}" fill="none" stroke="{EDGE}" '
        f'stroke-width="3"/>')
fig.raw(f'<line x1="40" y1="250" x2="820" y2="250" stroke="{MUTED}" '
        f'stroke-width="1.5"/>')
for value, label in [(-math.pi, "−π"), (0, "0"), (2, "2"), (math.pi, "π")]:
    fig.raw(f'<line x1="{px(value):.1f}" y1="245" x2="{px(value):.1f}" '
            f'y2="255" stroke="{MUTED}" stroke-width="1.5"/>')
    fig.text(px(value), 272, label, EDGE, "middle", 13)
fig.text(805, 272, "x", EDGE, "middle", 14)
fig.text(px(0), py(0) - 16, "top of the hill: the goal", INK, "middle", 13)
fig.text(px(math.pi), py(math.pi) + 24, "valley", EDGE, "middle", 12)
fig.text(px(-math.pi), py(-math.pi) + 24, "valley", EDGE, "middle", 12)
fig.raw(f'<circle cx="{px(2):.1f}" cy="{py(2) - 13:.1f}" r="12" '
        f'fill="#ddebff" stroke="#1a4fa0" stroke-width="2"/>')
fig.arrow((px(2) + 16, py(2) - 20), (px(2) + 86, py(2) + 22), gap=0,
          color=RED)
fig.text(px(2) + 40, py(2) - 26, "the slope pushes: sin x", RED, "start", 12)
fig.arrow((px(2) - 16, py(2) - 22), (px(2) - 74, py(2) - 62), gap=0,
          color="#1e7b34")
fig.text(px(2) - 60, py(2) - 78, "the action u", "#1e7b34", "start", 12)
fig.save("HillWorld.svg", "A robot on a hill",
         "A hill whose top is at position zero, with valleys at plus and "
         "minus pi. A robot at position two is pushed down the slope, away "
         "from the top, and its action pushes it back up.")

# 2. The graph: a learned dynamics factor, and a Gaussian forward message.
fig = Figure(900, 470)
for t, x in enumerate([100, 400, 700]):
    fig.variable(f"x{t}", (x, 170), f"x_{t}", "state")
    fig.text(x, 62, f"N(μ_{t}, Σ_{t})", LEARNED_TEXT, "middle", 13, bold=True)
for t, x in enumerate([250, 550]):
    fig.variable(f"u{t}", (x, 300), f"u_{t}", "action")
    fig.factor((x, 170), "learned", [f"x{t}", f"u{t}", f"x{t + 1}"],
               "Gaussian process", "above")
    fig.factor((x - 75, 235), "probability", [f"x{t}", f"u{t}"],
               f"π_θ(u_{t} | x_{t})", offset=(-14, 22), where="end")
    fig.factor((x - 150, 300), "value", [f"x{t}", f"u{t}"],
               f"r(x_{t}, u_{t})", "below")
    fig.arrow((x - 95, 57), (x + 95, 57), gap=0, color=LEARNED_TEXT)
    fig.text(x, 44, "moment matching", LEARNED_TEXT, "middle", 12)
fig.factor((790, 170), "value", ["x2"], "r(x_2)", "above")
fig.text(100, 86, "the forward message", LEARNED_TEXT, "middle", 12)
fig.edge((100, 95), (100, 143), dashed=True)
fig.edge((400, 72), (400, 143), dashed=True)
fig.edge((700, 72), (700, 143), dashed=True)
fig.variable("theta", (325, 385), "θ", "parameter")
fig.curve("theta", (150, 365), (175, 235))
fig.curve("theta", (450, 365), (475, 235))
fig.save("PilcoGraph.svg", "The graph of PILCO",
         "The chain of states and actions with policy factors that depend "
         "on a parameter theta, reward factors, and dynamics factors drawn "
         "in teal: each is a Gaussian process learned from transitions. "
         "Above each state is its forward message, a Gaussian, carried from "
         "one state to the next by moment matching.")
