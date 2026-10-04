"""Figure of Chapter 13."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED, LEARNED_TEXT, VALUE_TEXT

PURPLE = "#6639ba"
fig = Figure(1010, 400)
fig.box(30, 40, 470, 270, "Stage 1: sample with θ_k, and learn the backward message",
        EDGE)
fig.box(620, 40, 360, 270, "Stage 2: the actor", PURPLE)
for i, x in enumerate([110, 250, 390]):
    fig.variable(f"s{i}", (x, 150), f"s_{i}", "state", radius=20)
    fig.factor((x, 90), "learned", [f"s{i}"], "V̂" if i else "critic V̂",
               "right", size=14)
for i, x in enumerate([180, 320]):
    fig.variable(f"a{i}", (x, 240), f"a_{i}", "action", radius=20)
    fig.factor((x, 150), "sampled", [f"s{i}", f"a{i}", f"s{i + 1}"])
    fig.factor((x - 35, 195), "probability", [f"s{i}", f"a{i}"])
    fig.factor((x - 70, 240), "value", [f"s{i}", f"a{i}"])
fig.text(445, 156, "…", EDGE, "middle", 20)
fig.text(265, 292, "the critic is updated from each sampled transition",
         LEARNED_TEXT, "middle", 12)
fig.variable("theta", (800, 140), "θ", "parameter")
fig.text(800, 205, "θ_{k+1} = θ_k + α (sampled gradient)", INK, "middle", 13)
fig.text(800, 232, "score of each visited (s, a)", EDGE, "middle", 12)
fig.text(800, 248, "times its TD residual", EDGE, "middle", 12)
fig.arrow((500, 120), (620, 120), gap=4, color=EDGE)
fig.text(560, 84, "visited states,", EDGE, "middle", 12)
fig.text(560, 99, "TD residuals", EDGE, "middle", 12)
fig.arrow((620, 220), (500, 220), gap=4, color=PURPLE)
fig.text(560, 242, "new policy", PURPLE, "middle", 12)
fig.text(560, 257, "factors π_θ", PURPLE, "middle", 12)
fig.save("ActorCritic.svg", "Actor-critic as two stages",
         "Two boxes joined by two arrows. Stage 1 holds the chain of states "
         "and actions with white dynamics factors, which can only be "
         "sampled, and a teal learned value factor, the critic, on every "
         "state. It passes the visited states and the TD residuals to Stage "
         "2, the actor, which updates theta by a sampled gradient step and "
         "passes new policy factors back.")
