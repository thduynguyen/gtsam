"""Figures of Chapter 16."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import (Figure, MUTED, EDGE, RED, INK, VALUE_TEXT, LEARNED_TEXT)

PURPLE = "#6639ba"

# 1. One step with a deterministic policy and a learned critic.
fig = Figure(900, 430)
fig.variable("x", (140, 120), "x", "state")
fig.variable("u", (330, 260), "u", "action")
fig.variable("n", (520, 120), "x′", "state")
fig.variable("theta", (140, 330), "θ", "parameter")
fig.factor((330, 120), "sampled", ["x", "u", "n"],
           "one sample of p(x′ | x, u)", "above")
fig.factor((235, 190), "probability", ["x", "u", "theta"],
           "policy: u = μ_θ(x)", offset=(-16, -4), where="end")
fig.factor((600, 260), "learned", ["u"], "critic Q̂(x, u)", "above")
fig.edge((600, 260), "x", dashed=False)
fig.arrow((160, 356), (300, 290), bend=0.25, gap=0, color=PURPLE)
fig.text(250, 362, "1. dμ_θ/dθ: how the action moves with θ", PURPLE, "start",
         12, bold=True)
fig.arrow((590, 280), (362, 276), bend=0.12, gap=0, color=LEARNED_TEXT)
fig.text(490, 330, "2. dQ̂/du: which way the action should move",
         LEARNED_TEXT, "middle", 12, bold=True)
fig.save("DeterministicPolicyGradient.svg",
         "The deterministic policy gradient on one step",
         "A state, an action and a sampled next state. The policy factor is "
         "a hard constraint that sets the action to a function of the state "
         "with parameter theta. A learned critic factor joins the state and "
         "the action. Two arrows show the chain rule: the derivative of the "
         "action with respect to theta, and the derivative of the critic "
         "with respect to the action.")

# 2. The loop of DDPG.
fig = Figure(900, 320, legend=False)
boxes = [
    (30, 90, 180, "data policy", ["u = μ_θ(x) + noise;", "acts and stores", "transitions"]),
    (255, 90, 170, "buffer D", ["old transitions", "(x, u, r, x′):", "old forward messages"]),
    (470, 90, 200, "Stage 1: critic", ["Q̂(x, u) toward", "r + γ Q̂⁻(x′, μ⁻(x′)),", "from frozen copies"]),
    (715, 90, 160, "Stage 2: actor", ["θ moves along", "dμ_θ/dθ · dQ̂/du,", "averaged over D"]),
]
for x, y, w, title, lines in boxes:
    fig.box(x, y, w, 120, title, EDGE)
    for i, line in enumerate(lines):
        fig.text(x + w / 2, y + 52 + 18 * i, line, INK, "middle", 12)
fig.arrow((210, 150), (255, 150), gap=3, color=EDGE)
fig.arrow((425, 150), (470, 150), gap=3, color=EDGE)
fig.arrow((670, 150), (715, 150), gap=3, color=EDGE)
fig.arrow((795, 210), (120, 210), bend=0.16, gap=3, color=PURPLE)
fig.text(450, 296, "the new policy collects the next transitions", PURPLE,
         "middle", 12)
fig.text(450, 50, "DDPG: a learned critic and a deterministic actor", INK,
         "middle", 14, bold=True)
fig.save("DdpgLoop.svg", "The two stages of DDPG",
         "Four boxes in a loop. The data policy is the deterministic policy "
         "plus noise; it stores transitions in the buffer. Stage 1 moves the "
         "critic toward a bootstrapped target computed from frozen copies of "
         "the critic and the actor. Stage 2 moves the policy parameters "
         "along the product of the derivative of the action and the "
         "derivative of the critic, averaged over the buffer.")
