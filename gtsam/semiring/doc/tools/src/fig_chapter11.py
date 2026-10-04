"""Figures of Chapter 11."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED, LEARNED_TEXT, VALUE_TEXT

# 1. The graph: the dynamics are known only through samples.
fig = Figure(900, 430)
for t, x in enumerate([100, 400, 700]):
    fig.variable(f"s{t}", (x, 120), f"s_{t}", "state")
for t, x in enumerate([250, 550]):
    fig.variable(f"a{t}", (x, 250), f"a_{t}", "action")
fig.factor((100, 40), "sampled", ["s0"], "p(s_0)", "right")
policies = {}
for t, x in enumerate([250, 550]):
    fig.factor((x, 120), "sampled", [f"s{t}", f"a{t}", f"s{t + 1}"],
               f"p(s_{t + 1} | s_{t}, a_{t})", "above")
    policies[t] = (x - 75, 185)
    fig.factor(policies[t], "probability", [f"s{t}", f"a{t}"],
               f"π_θ(a_{t} | s_{t})", offset=(-14, 22), where="end")
    fig.factor((x - 150, 250), "value", [f"s{t}", f"a{t}"],
               f"r(s_{t}, a_{t})", "below")
fig.factor((790, 120), "value", ["s2"], "r(s_2)", "above")
fig.variable("theta", (325, 350), "θ", "parameter")
fig.curve("theta", (150, 330), policies[0])
fig.curve("theta", (450, 330), policies[1])
fig.save("MonteCarloGraph.svg",
         "The graph of the track when the dynamics can only be sampled",
         "The factor graph of the two-move track with a policy parameter "
         "theta. The policy and reward factors are known. The prior and the "
         "two dynamics factors are drawn as white squares: they cannot be "
         "read, only sampled by running the simulator.")

# 2. Particles: 100 rollouts of the coin flip through the track.
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])
dynamics = np.zeros((3, 2, 3))
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
rng = np.random.default_rng(0)
M = 100
s = np.zeros((M, 3), dtype=int)
s[:, 0] = (rng.random(M)[:, None] > prior.cumsum()).sum(axis=1)
for t in range(2):
    a = (rng.random(M) < 0.5).astype(int)
    u = rng.random(M)
    s[:, t + 1] = (u[:, None] > dynamics[s[:, t], a].cumsum(axis=1)).sum(axis=1)

fig = Figure(860, 410, legend=False)
xs = [190, 430, 670]
ys = {0: 310, 1: 210, 2: 110}  # cell 2, the charger, on top
exact = {0: [0.5, 0.5, 0.0], 1: [0.5, 0.3, 0.2], 2: [0.42, 0.34, 0.24]}
for t in range(2):
    for cell in range(3):
        for nxt in range(3):
            count = int(((s[:, t] == cell) & (s[:, t + 1] == nxt)).sum())
            if count:
                fig.raw(f'<line x1="{xs[t]}" y1="{ys[cell]}" x2="{xs[t + 1]}" '
                        f'y2="{ys[nxt]}" stroke="#7aa7e0" '
                        f'stroke-width="{0.35 * count:.1f}" '
                        f'stroke-linecap="round" opacity="0.75"/>')
for t in range(3):
    fig.text(xs[t], 40, f"step {t}", INK, "middle", 14, bold=True)
    for cell in range(3):
        count = int((s[:, t] == cell).sum())
        radius = 13 + 2.4 * np.sqrt(count)
        fig.raw(f'<circle cx="{xs[t]}" cy="{ys[cell]}" r="{radius:.1f}" '
                f'fill="#ddebff" stroke="#1a4fa0" stroke-width="2"/>')
        fig.text(xs[t], ys[cell] + 5, str(count), INK, "middle", 14, bold=True)
        fig.text(xs[t], ys[cell] - radius - 7,
                 f"exact d_{t} = {exact[t][cell]:g}", EDGE, "middle", 12)
for cell in range(3):
    fig.text(60, ys[cell] + 5, f"cell {cell}", EDGE, "middle", 13)
fig.text(430, 384, "100 rollouts of the coin flip. The number in a circle "
         "counts the rollouts in that cell; line widths count the moves.",
         EDGE, "middle", 12)
fig.save("MonteCarloParticles.svg",
         "The forward message as particles",
         "Three columns for the steps 0, 1 and 2 and three rows for the "
         "cells. A circle shows how many of 100 sampled rollouts are in that "
         "cell at that step, next to the exact visitation probability. Lines "
         "between columns show the sampled moves, wider where more rollouts "
         "took them.")
