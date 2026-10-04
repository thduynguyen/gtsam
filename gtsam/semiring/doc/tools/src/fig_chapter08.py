"""Figures of Chapter 8."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, chain, EDGE, RED, VALUE_TEXT

# 1. Control as inference: the rewards become ordinary factors.
fig = Figure(900, 400)
where = chain(fig, 2, y_state=130, y_action=270, reward=False, final=False)
for t in range(2):
    fig.factor((100 + 300 * t, 270), "probability", [f"s{t}", f"a{t}"],
               f"exp(r(s_{t}, a_{t}) / η)", "below")
fig.factor((790, 130), "probability", ["s2"], "exp(r(s_2) / η)", "above")
fig.text(735, 236, "every factor is now a probability factor:", EDGE,
         "middle", 12)
fig.text(735, 254, "an ordinary factor graph, as in SLAM", EDGE, "middle", 12)
fig.save("InferenceGraph.svg", "Control as inference: rewards as factors",
         "The factor graph of the track, with a base policy factor on each "
         "state and action. Each reward factor has been replaced by a "
         "probability factor, the exponential of the reward divided by the "
         "temperature, so that all factors are of one kind.")

# 2. Which sum each variable gets, in the four formulations.
fig = Figure(900, 450, legend=False)
fig.variable("x", (150, 110), "s", "state")
fig.variable("u", (300, 230), "a", "action")
fig.variable("n", (450, 110), "s′", "state")
fig.factor((300, 110), "probability", ["x", "u", "n"], "p(s′ | s, a)", "above")
fig.factor((225, 170), "probability", ["x", "u"], "π(a | s)",
           offset=(-14, 22), where="end")
fig.factor((150, 230), "value", ["x", "u"], "r(s, a)", "below")
fig.factor((560, 110), "value", ["n"], "(1, V(s′))", "above")
rows = [("", "the action a", "the next state s′", EDGE, True),
        ("optimal control (Ch. 4, 6)", "maximum", "average", EDGE, False),
        ("soft control", "soft maximum", "average", "#1e7b34", False),
        ("risk-sensitive control", "maximum", "tilted mean", "#1a4fa0", False),
        ("control as inference", "soft maximum", "tilted mean", RED, False)]
y = 322
for name, action, state, color, bold in rows:
    fig.text(60, y, name, color, "start", 13, bold=bold or True)
    fig.text(420, y, action, color, "middle", 13, bold=bold)
    fig.text(640, y, state, color, "middle", 13, bold=bold)
    y += 24
fig.raw('<line x1="50" y1="330" x2="760" y2="330" stroke="#d0d7de"/>')
fig.text(60, 296, "How each variable is summed out:", EDGE, "start", 12)
fig.save("TiltedSums.svg", "Which sum eliminates each variable",
         "One time step of the graph: a state, its action and the next "
         "state, with the dynamics, base policy and reward factors and the "
         "value of the future. Below it a table: optimal control uses a "
         "maximum over the action and an average over the next state; soft "
         "control a soft maximum and an average; risk-sensitive control a "
         "maximum and a tilted mean; control as inference a soft maximum and "
         "a tilted mean.")
