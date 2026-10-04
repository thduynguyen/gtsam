"""Figures of Chapter 15."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import (Figure, MUTED, EDGE, RED, INK, VALUE_TEXT, LEARNED_TEXT)

# 1. One sampled transition and the two tables it touches.
fig = Figure(900, 400)
fig.variable("s", (120, 130), "s", "state")
fig.variable("a", (280, 260), "a", "action")
fig.variable("n", (460, 130), "s′", "state")
fig.variable("b", (620, 260), "a′", "action")
fig.factor((280, 130), "sampled", ["s", "a", "n"],
           "one sample of p(s′ | s, a)", "above")
fig.factor((120, 260), "value", ["s", "a"], "r(s, a)", "below")
fig.factor((200, 195), "learned", ["s", "a"], "Q̂(s, a): updated",
           offset=(-14, 22), where="end")
fig.factor((540, 195), "learned", ["n", "b"], "Q̂(s′, a′): read",
           offset=(16, -8), where="start")
fig.text(660, 300, "SARSA: a′ sampled from the policy", EDGE, "start", 12,
         bold=True)
fig.text(660, 318, "Q-learning: a′ = the best move in s′", RED, "start", 12,
         bold=True)
fig.save("SampledBackup.svg", "One sampled transition in value-based control",
         "A state, the action taken, the sampled next state and a next "
         "action. The dynamics factor is known only through this one sample. "
         "A learned factor holds the estimate of Q at the state and action "
         "and is updated; a second copy is read at the next state and next "
         "action, where SARSA samples the next action from the policy and "
         "Q-learning takes the best one.")

# 2. The loop through the replay buffer.
fig = Figure(900, 300, legend=False)
boxes = [
    (30, 90, 170, "data policy π_D", ["acts in the world;", "mostly greedy,", "sometimes random"]),
    (250, 90, 170, "buffer D", ["old transitions", "(s, a, r, s′):", "old forward messages"]),
    (470, 90, 190, "Stage 1", ["bootstrapped backup:", "Q̂(s, a) toward", "r + γ max Q̂(s′, a′)"]),
    (710, 90, 160, "Stage 2", ["greedy: the best", "move of Q̂", "in every state"]),
]
for x, y, w, title, lines in boxes:
    fig.box(x, y, w, 120, title, EDGE)
    for i, line in enumerate(lines):
        fig.text(x + w / 2, y + 52 + 18 * i, line, INK, "middle", 12)
fig.arrow((200, 150), (250, 150), gap=3, color=EDGE)
fig.arrow((420, 150), (470, 150), gap=3, color=EDGE)
fig.arrow((660, 150), (710, 150), gap=3, color=EDGE)
fig.arrow((790, 210), (115, 210), bend=0.16, gap=3, color="#6639ba")
fig.text(450, 282, "the new greedy policy collects the next transitions",
         "#6639ba", "middle", 12)
fig.text(450, 50, "Q-learning with a replay buffer", INK, "middle", 14,
         bold=True)
fig.save("ReplayLoop.svg", "The two stages of Q-learning with a replay buffer",
         "Four boxes in a loop. The data policy acts and stores transitions "
         "in the buffer. Stage 1 draws old transitions from the buffer and "
         "moves the estimate of Q toward the bootstrapped target. Stage 2 "
         "makes the policy greedy. The new policy collects the next "
         "transitions.")
