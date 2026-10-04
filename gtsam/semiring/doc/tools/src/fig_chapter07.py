"""Figures of Chapter 7."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, chain, EDGE, VALUE_TEXT, LEARNED_TEXT, RED

# The line with a linear-Gaussian policy whose gain is the parameter.
fig = Figure(900, 500)
where = chain(fig, 2, y_state=170, y_action=300, state="x", action="u",
              policy_label="π_K(u_{t} | x_{t})")
fig.variable("K", (325, 400), "K", "parameter")
fig.curve("K", (150, 380), where["policy0"])
fig.curve("K", (450, 380), where["policy1"])
fig.text(365, 405, "the gains: one per move, or one shared", EDGE, "start",
         12)
fig.arrow((100, 48), (700, 48), gap=0, color=LEARNED_TEXT)
fig.text(430, 36, "forward messages: the state is Gaussian, with "
         "second moment E[x_t x_t^⊤]", LEARNED_TEXT, "middle", 12, bold=True)
fig.arrow((830, 245), (590, 300), bend=-0.15, gap=0, color=VALUE_TEXT)
fig.text(770, 228, "backward messages: the advantage,", VALUE_TEXT, "middle",
         12, bold=True)
fig.text(770, 244, "", VALUE_TEXT)
fig.text(790, 318, "a quadratic with blocks H_{uu}, H_{ux}", VALUE_TEXT,
         "middle", 12, bold=True)
fig.save("LinearPolicyGraph.svg",
         "The line with a linear-Gaussian policy and its gain as a parameter",
         "The factor graph of the robot on a line with two moves. Each policy "
         "factor is a Gaussian whose mean is minus a gain times the state. "
         "The gains form a parameter node joined to both policy factors. A "
         "forward arrow marks the Gaussian state marginals and a backward "
         "arrow marks the quadratic advantages.")
