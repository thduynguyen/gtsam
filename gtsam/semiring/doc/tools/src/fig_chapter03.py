"""Figures of Chapter 3."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, MUTED, EDGE, VALUE_TEXT

# 1. The chain with a termination outcome.
fig = Figure(900, 400)
for t, x in enumerate([100, 400, 700]):
    fig.variable(f"s{t}", (x, 120), f"s_{t}", "state")
for t, x in enumerate([250, 550]):
    fig.variable(f"a{t}", (x, 250), f"a_{t}", "action")
fig.variable("end", (850, 250), "∅", "terminal")
fig.factor((100, 40), "probability", ["s0"], "p(s_0)", "right")
for t, x in enumerate([250, 550]):
    fig.factor((x, 120), "probability", [f"s{t}", f"a{t}", f"s{t + 1}"],
               f"γ p(s_{t + 1} | s_{t}, a_{t})", "above")
    fig.factor((x - 75, 185), "probability", [f"s{t}", f"a{t}"],
               f"π(a_{t} | s_{t})", offset=(-14, 22), where="end")
    fig.factor((x - 150, 250), "value", [f"s{t}", f"a{t}"],
               f"r(s_{t}, a_{t})", "below")
fig.arrow((258, 128), "end", bend=-0.32, start_gap=0)
fig.arrow((558, 128), "end", bend=-0.12, start_gap=0)
fig.text(760, 318, "ended, with probability 1 − γ", MUTED, "middle", 12)
fig.text(800, 125, "…", EDGE, "middle", 22)
fig.save("DiscountedChain.svg", "A discounted chain with a termination outcome",
         "The chain of states and actions of Chapter 1. Each dynamics factor "
         "continues to the next state with probability gamma, and with "
         "probability one minus gamma leads to an extra state, ended, drawn "
         "in grey, where nothing more is collected.")

# 2. One stationary step: the same value function in and out.
fig = Figure(820, 370)
fig.variable("s", (160, 150), "s", "state")
fig.variable("a", (320, 270), "a", "action")
fig.variable("n", (480, 150), "s′", "state")
fig.factor((320, 150), "probability", ["s", "a", "n"], "γ p(s′ | s, a)", "above")
fig.factor((240, 210), "probability", ["s", "a"], "π(a | s)",
           offset=(-14, 22), where="end")
fig.factor((160, 270), "value", ["s", "a"], "r(s, a)", "below")
fig.factor((600, 150), "value", ["n"], "in: (1, V(s′))", "below")
fig.factor((50, 150), "value", ["s"], "out: (1, V(s))", "below")
fig.arrow((600, 136), (50, 136), bend=-0.2, gap=6, color=VALUE_TEXT,
          dashed=True)
fig.text(325, 62, "eliminate s′, then a: the same function V comes out",
         VALUE_TEXT, "middle", 12)
fig.save("StationaryBackup.svg", "One step of a stationary chain",
         "A state, its action and the next state, with the dynamics, policy "
         "and reward factors. A value factor holding V enters on the next "
         "state. After eliminating the next state and the action, a value "
         "factor holding the same function V leaves on the state.")
