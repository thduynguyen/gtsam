"""Figures of Chapter 25."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, VALUE_TEXT, INK

# One step: the value factor holds a distribution per state.
fig = Figure(900, 400)
fig.variable("s", (190, 170), "s", "state")
fig.variable("a", (350, 290), "a", "action")
fig.variable("n", (510, 170), "s′", "state")
fig.factor((350, 170), "probability", ["s", "a", "n"], "p(s′ | s, a)", "above")
fig.factor((270, 230), "probability", ["s", "a"], "π(a | s)",
           offset=(-14, 22), where="end")
fig.factor((190, 290), "value", ["s", "a"], "r(s, a): all mass at z = r",
           "below")
fig.factor((630, 170), "value", ["n"], None)
fig.text(650, 150, "in: for each s′, a distribution", VALUE_TEXT, "start")
fig.text(650, 168, "over the reward still to come", VALUE_TEXT, "start")
fig.factor((70, 170), "value", ["s"], None)
fig.text(50, 128, "out: for each s,", VALUE_TEXT, "start")
fig.text(50, 146, "a distribution", VALUE_TEXT, "start")
fig.arrow((630, 156), (70, 110), bend=-0.16, gap=6, color=VALUE_TEXT,
          dashed=True)
fig.text(350, 30, "multiply with the reward: shift by r (a convolution)",
         VALUE_TEXT, "middle", 12)
fig.text(350, 48, "sum out s′ and a: mix, weighted by probability",
         VALUE_TEXT, "middle", 12)
fig.save("DistributionalBackup.svg", "One step with distributions as values",
         "A state, its action and the next state, with dynamics, policy and "
         "reward factors. The value factor entering on the next state holds, "
         "for each next state, a distribution over the reward still to come. "
         "Multiplying with the reward factor shifts each distribution by the "
         "reward, and summing out the next state and the action mixes the "
         "shifted distributions. The value factor leaving on the state holds "
         "a distribution for each state.")
