"""Figures of Chapter 12."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED, LEARNED_TEXT, VALUE_TEXT

# 1. One sampled transition: the estimate at s is pulled toward a target
#    built from the estimate at s'.
fig = Figure(840, 380)
fig.variable("s", (170, 160), "s", "state")
fig.variable("a", (330, 280), "a", "action")
fig.variable("n", (490, 160), "s′", "state")
fig.factor((330, 160), "sampled", ["s", "a", "n"], "p(s′ | s, a): one sample",
           "above")
fig.factor((250, 220), "probability", ["s", "a"], "π(a | s)",
           offset=(-14, 22), where="end")
fig.factor((170, 280), "value", ["s", "a"], "r(s, a)", "below")
fig.factor((610, 160), "learned", ["n"], "V̂(s′)", "below")
fig.factor((60, 160), "learned", ["s"], "V̂(s)", "below")
fig.arrow((610, 146), (60, 146), bend=-0.2, gap=6, color=LEARNED_TEXT,
          dashed=True)
fig.text(335, 62, "target for V̂(s):  r + γ V̂(s′)", LEARNED_TEXT, "middle",
         13, bold=True)
fig.save("BootstrapBackup.svg",
         "One sampled transition and the bootstrapped target",
         "A state, its action and the next state. The dynamics factor is "
         "white: only one sampled next state is seen. A learned value factor "
         "sits on the next state and another on the state. A dashed arrow "
         "from the first to the second carries the target, the reward plus "
         "gamma times the estimate at the next state.")

# 2. Targets that use n sampled rewards and then the estimate.
fig = Figure(880, 400)
xs = [210, 330, 450, 570, 690]
rows = [(90, "n = 1 (TD)", 1), (190, "n = 3", 3), (290, "Monte Carlo", 5)]
first = True
for y, label, n in rows:
    fig.text(40, y + 5, label, INK, "start", 13, bold=True)
    for i in range(min(n, 4) + 1):
        name = f"{label}{i}"
        sub = "t" if i == 0 else f"t+{i}"
        fig.variable(name, (xs[i], y), f"s_{{{sub}}}", "state", radius=20)
    for i in range(min(n, 4)):
        sub = "t" if i == 0 else f"t+{i}"
        fig.factor(((xs[i] + xs[i + 1]) / 2, y), "value",
                   [f"{label}{i}", f"{label}{i + 1}"],
                   "r_t" if i == 0 else f"γ r_{{{sub}}}" if i == 1
                   else f"γ^{i} r_{{{sub}}}", "above", size=14)
    last = f"{label}{min(n, 4)}"
    if n <= 4:
        fig.factor((xs[n] + 70, y), "learned", [last],
                   f"γ^{n} V̂(s_{{t+{n}}})" if n > 1 else "γ V̂(s_{t+1})",
                   "above", size=14)
    else:
        fig.text(xs[4] + 55, y + 6, "…  to the end", EDGE, "start", 13)
        fig.edge((xs[4] + 20, y), (xs[4] + 48, y))
fig.save("NStepReturns.svg", "Targets with n sampled rewards",
         "Three rows. In the first, one sampled reward is followed by the "
         "learned value of the next state. In the second, three sampled "
         "rewards are followed by the learned value three steps later. In "
         "the third, the sampled rewards run to the end of the episode and "
         "no learned value is used.")
