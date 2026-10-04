"""Figures of Chapter 24."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookfig import Figure, EDGE, INK, MUTED, RED, VALUE_TEXT, LEARNED_TEXT

# 1. The graph of a bandit with an unknown model.
fig = Figure(900, 420)
fig.variable("theta", (450, 60), "θ_p", "parameter")
for t, x in enumerate([150, 450, 750]):
    fig.variable(f"a{t}", (x - 70, 300), f"a_{t}", "action")
    fig.variable(f"y{t}", (x + 70, 300), f"y_{t}", "observation")
    fig.factor((x, 190), "probability", [f"a{t}", f"y{t}", "theta"],
               f"p(y_{t} | a_{t}, θ_p)", offset=(14, 22), where="start")
    fig.factor((x + 70, 360 - 10), "value", [f"y{t}"], None)
    fig.text(x + 92, 355, f"r = y_{t}", VALUE_TEXT, "start")
fig.factor((560, 60), "probability", ["theta"], "p(θ_p)", "right")
fig.arrow("y0", "a1", bend=0.0, color=LEARNED_TEXT, dashed=True)
fig.arrow("y1", "a2", bend=0.0, color=LEARNED_TEXT, dashed=True)
fig.text(170, 60, "the unknown model: joined to every outcome factor", EDGE,
         "middle", 12)
fig.save("BanditGraph.svg", "A bandit with an unknown model",
         "Three pulls. Each pull has an action, the arm, and an observed "
         "outcome, joined by an outcome factor. One parameter node, the "
         "unknown success probabilities, is joined to every outcome factor "
         "and has a prior. Dashed arrows show that each action may depend on "
         "the outcomes before it.")

# 2. The first two levels of the tree of beliefs.
fig = Figure(900, 420, legend=False)


def node(x, y, lines, color=INK, width=150):
    fig.raw(f'<rect x="{x - width / 2}" y="{y - 24}" width="{width}" '
            f'height="48" rx="8" fill="#f6f8fa" stroke="{EDGE}" '
            f'stroke-width="1.5"/>')
    fig.text(x, y - 5, lines[0], color, "middle", 12, bold=True)
    fig.text(x, y + 13, lines[1], EDGE, "middle", 12)


def link(p, q, label, color=EDGE, shift=(0, 0)):
    fig.edge((p[0], p[1] + 24), (q[0], q[1] - 24))
    fig.text((p[0] + q[0]) / 2 + shift[0], (p[1] + q[1]) / 2 + shift[1], label,
             color, "middle", 12)


root = (450, 55)
pull_u, pull_k = (240, 170), (660, 170)
node(*root, ["U: 1, 1   K: 12, 8", "10 pulls left: best 6.369"], width=190)
node(*pull_u, ["pull U", "value 6.369"], RED, 130)
node(*pull_k, ["pull K", "value 6.314"], RED, 130)
link(root, pull_u, "max", RED, (-28, -4))
link(root, pull_k, "max", RED, (28, -4))
leaves = [((120, 300), ["U: 2, 1   K: 12, 8", "9 pulls left: 6.339"], "success 0.5"),
          ((360, 300), ["U: 1, 2   K: 12, 8", "9 pulls left: 5.400"], "failure 0.5"),
          ((540, 300), ["U: 1, 1   K: 13, 8", "9 pulls left"], "success 0.6"),
          ((780, 300), ["U: 1, 1   K: 12, 9", "9 pulls left"], "failure 0.4")]
for i, (position, lines, label) in enumerate(leaves):
    node(*position, lines, width=170)
    parent = pull_u if i < 2 else pull_k
    link(parent, position, label, EDGE, (-48 if i % 2 == 0 else 48, 0))
fig.text(450, 375, "max over the arm, average over the outcome; "
         "a belief is the four counts", MUTED, "middle", 12)
fig.save("BeliefTree.svg", "The first levels of the tree of beliefs",
         "A tree. The root is the starting belief, counts 1, 1 for arm U and "
         "12, 8 for arm K, with 10 pulls left. It branches by a maximum into "
         "pulling U, worth 6.369, and pulling K, worth 6.314. Each of those "
         "branches by an average into a success and a failure, which lead to "
         "four new beliefs with one count increased.")
