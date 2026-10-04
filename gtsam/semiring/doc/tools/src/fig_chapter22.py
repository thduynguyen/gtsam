"""Figures of Chapter 22."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, MUTED, EDGE, RED, LEARNED_TEXT

# 1. The graph of a partially observed problem.
fig = Figure(900, 470)
for t, x in enumerate([110, 410, 710]):
    fig.variable(f"s{t}", (x, 170), f"s_{t}", "state")
for t, x in enumerate([260, 560]):
    fig.variable(f"a{t}", (x, 300), f"a_{t}", "action")
for t, x in enumerate([110, 410]):
    fig.variable(f"y{t}", (x, 50), f"y_{t}", "observation")
    fig.factor((x, 110), "probability", [f"s{t}", f"y{t}"],
               f"p(y_{t} | s_{t})", "right")
fig.factor((30, 170), "probability", ["s0"], "p(s_0)", "above")
for t, x in enumerate([260, 560]):
    fig.factor((x, 170), "probability", [f"s{t}", f"a{t}", f"s{t + 1}"],
               f"p(s_{t + 1} | s_{t}, a_{t})", "above")
    fig.factor((x - 150, 300), "value", [f"s{t}", f"a{t}"],
               f"r(s_{t}, a_{t})", "below")
fig.factor((800, 170), "value", ["s2"], "r(s_2)", "above")
# What each decision may depend on.
fig.arrow("y0", "a0", bend=0.18, color=LEARNED_TEXT, dashed=True)
fig.arrow("y1", "a1", bend=0.18, color=LEARNED_TEXT, dashed=True)
fig.text(700, 350, "dashed arrows: what a decision sees: the reading,",
         LEARNED_TEXT, "middle", 12)
fig.text(700, 366, "never the state. It also remembers earlier readings.", LEARNED_TEXT,
         "middle", 12)
fig.save("PomdpGraph.svg", "A partially observed decision problem",
         "The decision graph of Chapter 4 with an observation node attached "
         "to each of the first two states by a sensor factor. Dashed arrows "
         "run from each observation to the action chosen after it: an action "
         "may depend on the readings made so far, and on nothing else.")

# 2. The filter: the forward message with evidence.
fig = Figure(900, 330)
fig.variable("s0", (140, 170), "s_0", "done")
fig.variable("s1", (480, 170), "s_1", "state")
fig.factor((60, 170), "probability", ["s0"], "p(s_0)", "above")
fig.factor((140, 80), "probability", ["s0"], "p(y_0 | s_0), y_0 fixed", "right")
fig.factor((310, 170), "probability", ["s0", "s1"],
           "p(s_1 | s_0, a_0), a_0 fixed", "above")
fig.factor((480, 80), "probability", ["s1"], "p(y_1 | s_1), y_1 fixed", "right")
fig.arrow((175, 215), (445, 215), bend=0.12, gap=0, color=LEARNED_TEXT)
fig.text(310, 262, "forward message: d_1(s_1 | y_0), the predicted belief",
         LEARNED_TEXT, "middle", 12, bold=True)
fig.text(690, 165, "marginal of s_1:", EDGE, "middle", 12)
fig.text(690, 183, "the belief d_1(s_1 | y_0, y_1)", EDGE, "middle", 12,
         bold=True)
fig.save("BeliefFilter.svg", "The belief as a forward message",
         "Two states joined by the dynamics factor with the action fixed. "
         "Each state also has a sensor factor with its reading fixed. "
         "Eliminating the first state sends a forward message to the second, "
         "the predicted belief; multiplying it by the second sensor factor "
         "and normalizing gives the belief.")
