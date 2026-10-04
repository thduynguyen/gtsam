"""Figures of Chapter 20."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED

PURPLE = "#6639ba"

fig = Figure(900, 560)
fig.variable("theta", (450, 258), "θ", "parameter")
fig.text(484, 263, "one global policy π_θ", EDGE, "start", 12)
rows = [("a", 80, 170, "local problem 1: start x_0^{(1)}", 34),
        ("b", 440, 350, "local problem 2: start x_0^{(2)}", 492)]
for name, y_state, y_action, title, y_title in rows:
    fig.text(60, y_title, title, EDGE, "start", 12, bold=True)
    for t, x in enumerate([120, 400, 680]):
        fig.variable(f"{name}x{t}", (x, y_state), f"x_{t}", "state",
                     radius=20)
    for t, x in enumerate([260, 540]):
        fig.variable(f"{name}u{t}", (x, y_action), f"u_{t}", "action",
                     radius=20)
        fig.factor((x, y_state), "learned",
                   [f"{name}x{t}", f"{name}u{t}", f"{name}x{t + 1}"])
        fig.factor((x - 140, y_action), "value", [f"{name}x{t}",
                                                  f"{name}u{t}"])
        # The agreement factor, joined to x_t, u_t and theta.
        centre = (x - 70, (y_state + y_action) / 2)
        fig.edge(centre, f"{name}x{t}")
        fig.edge(centre, f"{name}u{t}")
        fig.curve(centre, (centre[0] + 10, 258), "theta", dashed=True)
        fig.raw(f'<rect x="{centre[0] - 8}" y="{centre[1] - 8}" width="16" '
                f'height="16" fill="#ffffff" stroke="{PURPLE}" '
                f'stroke-width="3"/>')
    fig.factor((770, y_state), "value", [f"{name}x2"])
fig.raw(f'<rect x="652" y="250" width="14" height="14" fill="#ffffff" '
        f'stroke="{PURPLE}" stroke-width="3"/>')
fig.text(676, 255, "agreement factor: u_t = −K_t x_t,", PURPLE, "start", 12)
fig.text(676, 271, "with a dual variable ν_t", PURPLE, "start", 12)
fig.save("GuidedPolicySearchGraph.svg", "The graph of guided policy search",
         "Two local trajectory problems, one per start position, each a "
         "chain of states and actions with dynamics factors fitted locally "
         "and reward factors. In each, an agreement factor joins every "
         "state and its action to the parameter theta of one global policy, "
         "which is shared by all local problems.")
