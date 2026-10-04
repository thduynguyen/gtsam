"""Figures of Chapter 5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import (Figure, chain, MUTED, EDGE, RED, INK, VALUE_TEXT,
                     LEARNED_TEXT)

# 1. Forward and backward messages meeting at one policy factor.
fig = Figure(900, 470)
where = chain(fig, 2, y_state=150, y_action=280,
              policy_label="π_θ(a_{t} | s_{t})")
fig.variable("theta", (325, 380), "θ", "parameter")
fig.curve("theta", (150, 360), where["policy0"])
fig.curve("theta", (450, 360), where["policy1"])
# Forward message into s1, backward message into a1.
fig.arrow((130, 60), (385, 118), bend=-0.12, gap=0, color=LEARNED_TEXT)
fig.text(250, 38, "forward message d_1(s): how often the robot is in s",
         LEARNED_TEXT, "middle", 12, bold=True)
fig.arrow((800, 250), (582, 282), bend=-0.15, gap=0, color=VALUE_TEXT)
fig.text(740, 235, "backward message A_1(s, a):", VALUE_TEXT, "middle", 12,
         bold=True)
fig.text(740, 251, "", VALUE_TEXT, "middle", 12)
fig.text(775, 300, "how good the action is", VALUE_TEXT, "middle", 12,
         bold=True)
fig.raw(f'<circle cx="{where["policy1"][0]}" cy="{where["policy1"][1]}" '
        f'r="20" fill="none" stroke="{RED}" stroke-width="2" '
        f'stroke-dasharray="6 4"/>')
fig.text(590, 205, "local derivative ∇π_θ", RED, "start", 12, bold=True)
fig.save("PolicyGradientMessages.svg",
         "The policy gradient as forward and backward messages",
         "The factor graph of the track with a parameter theta joined to both "
         "policy factors. At the second policy factor three things meet: the "
         "forward message, the marginal of the state; the backward message, "
         "the advantage of the action; and the derivative of the policy "
         "factor itself with respect to theta.")

# 2. The two stages.
fig = Figure(900, 330, legend=False)
fig.box(40, 50, 380, 210, "Stage 1: eliminate, with θ fixed at θ_k", EDGE)
fig.box(560, 50, 300, 210, "Stage 2: update θ", "#6639ba")
for i, x in enumerate([110, 230, 350]):
    fig.variable(f"s{i}", (x, 130), f"s_{i}", "state", radius=20)
for i, x in enumerate([170, 290]):
    fig.variable(f"a{i}", (x, 205), f"a_{i}", "action", radius=20)
    fig.factor((x, 130), "probability", [f"s{i}", f"a{i}", f"s{i + 1}"])
    fig.factor((x - 30, 168), "probability", [f"s{i}", f"a{i}"])
    fig.factor((x - 60, 205), "value", [f"s{i}", f"a{i}"])
fig.variable("theta", (710, 150), "θ", "parameter")
fig.text(710, 210, "θ_{k+1} = update(θ_k; messages)", INK, "middle", 13)
fig.text(710, 232, "greedy, gradient, natural gradient,", EDGE, "middle", 12)
fig.text(710, 248, "trust region, EM, sampling", EDGE, "middle", 12)
fig.arrow((420, 120), (560, 120), gap=4, color=EDGE)
fig.text(490, 98, "J(θ_k), forward and", EDGE, "middle", 12)
fig.text(490, 112, "backward messages", EDGE, "middle", 12)
fig.arrow((560, 200), (420, 200), gap=4, color="#6639ba")
fig.text(490, 222, "new policy", "#6639ba", "middle", 12)
fig.text(490, 236, "factors π_θ", "#6639ba", "middle", 12)
fig.save("TwoStages.svg", "The two-stage framework",
         "Two boxes joined by two arrows. Stage 1 holds the factor graph of "
         "states and actions and eliminates it with the parameter theta "
         "fixed; it passes the expected return and the forward and backward "
         "messages to Stage 2. Stage 2 updates theta and passes new policy "
         "factors back to Stage 1.")
