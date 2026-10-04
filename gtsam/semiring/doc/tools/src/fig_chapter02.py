"""Figures of Chapter 2: two steps of max-sum elimination."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bookfig
from bookfig import Figure, EDGE, VALUE_TEXT

# In max-sum every entry is a value; a lifted probability is 0 or -infinity.
bookfig.LEGEND_TEXT["probability"] = "lifted probability: 0 if possible, −∞ if not"
bookfig.LEGEND_TEXT["value"] = "value"
bookfig.LEGEND_TEXT["done"] = "eliminated; arrows: its conditional"

S1, A1, S2 = (140, 140), (320, 270), (500, 140)
DYNAMICS, REWARD, FINAL = (320, 140), (140, 270), (620, 140)


def base(height=380):
    return Figure(900, height)


# 1. Before: the bucket of the last state.
fig = base()
fig.variable("s1", S1, "s_1", "state")
fig.variable("a1", A1, "a_1", "action")
fig.variable("s2", S2, "s_2", "state")
fig.factor(DYNAMICS, "probability", ["s1", "a1", "s2"],
           "p(s_2 | s_1, a_1), lifted", "above")
fig.factor(REWARD, "value", ["s1", "a1"], "r(s_1, a_1)", "below")
fig.factor(FINAL, "value", ["s2"], "r(s_2)", "above")
fig.ring("s2")
fig.text(560, 215, "bucket of s_2: add the two factors,", EDGE, "start", 12)
fig.text(560, 233, "then keep the best s_2", EDGE, "start", 12)
fig.save("MaxSumStart.svg", "The last move of the track, before elimination",
         "The state s1, the action a1 and the last state s2. The dynamics "
         "factor, lifted to zero or minus infinity, joins all three. A value "
         "factor holding the reward of the move joins s1 and a1, and a value "
         "factor holding the final reward attaches to s2, which is about to "
         "be eliminated.")

# 2. After eliminating the last state.
fig = base()
fig.variable("s1", S1, "s_1", "state")
fig.variable("a1", A1, "a_1", "action")
fig.factor(DYNAMICS, "value", ["s1", "a1"], "φ(s_1, a_1)", "above")
fig.text(30, 40, "new factor: φ(s_1, a_1) = the best r(s_2) among the possible "
         "s_2", VALUE_TEXT, "start", 13)
fig.factor(REWARD, "value", ["s1", "a1"], "r(s_1, a_1)", "below")
fig.variable("s2", S2, "s_2", "done")
fig.arrow("s1", "s2", bend=0.25)
fig.arrow("a1", "s2", bend=0.0)
fig.text(545, 146, "c(s_2 | s_1, a_1): the outcome that was assumed", EDGE,
         "start", 12)
fig.ring("a1")
fig.text(375, 275, "bucket of a_1: add the two factors,", EDGE, "start", 12)
fig.text(375, 293, "then keep the best a_1", EDGE, "start", 12)
fig.save("MaxSumNextState.svg", "After eliminating the last state by maximum",
         "The last state is eliminated and keeps a conditional given s1 and "
         "a1. The dynamics and final reward factors are replaced by one new "
         "value factor joining s1 and a1, holding the best final reward among "
         "the outcomes that are possible. The action is about to be "
         "eliminated.")

# 3. After eliminating the action.
fig = base()
fig.variable("s1", S1, "s_1", "state")
fig.factor((40, 140), "value", ["s1"], "φ(s_1)", "below")
fig.text(30, 40, "new factor: φ(s_1) = the best of r(s_1, a_1) + φ(s_1, a_1) "
         "over a_1", VALUE_TEXT, "start", 13)
fig.variable("a1", A1, "a_1", "done")
fig.variable("s2", S2, "s_2", "done")
fig.arrow("s1", "a1", bend=0.0)
fig.arrow("s1", "s2", bend=0.25)
fig.arrow("a1", "s2", bend=0.0)
fig.text(360, 280, "c(a_1 | s_1): the best action, and the regret of the other",
         EDGE, "start", 12)
fig.text(545, 146, "c(s_2 | s_1, a_1)", EDGE, "start", 12)
fig.save("MaxSumAction.svg", "After eliminating the last action by maximum",
         "The action is eliminated and keeps a conditional given s1: the best "
         "action. The remaining factors are replaced by one new value factor "
         "on s1, holding the best total over the actions.")
