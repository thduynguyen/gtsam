"""Figures of Chapter 10. Run with the notebook environment's python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import logsumexp
from bookfig import Figure, chain, EDGE, INK, MUTED, RED

BLUE = "#1a4fa0"

# 1. The graph: an open-loop Gaussian policy over plans.
fig = Figure(900, 500)
where = chain(fig, 2, y_state=120, y_action=250, state="x", action="u",
              policy=False, final=False)
fig.text(790, 126, "…", EDGE, "middle", 22)
fig.variable("theta", (400, 400), "ū", "parameter")
for t, x in enumerate([250, 550]):
    centre = (x, 330)
    fig.edge("theta", centre)
    fig.factor(centre, "probability", [f"u{t}"],
               f"N(u_{t}; ū_{t}, Σ_e)", "right" if t else "left")
fig.text(440, 418, "the plan: the mean of the sampling distribution", EDGE,
         "start", 12)
fig.save("SamplingPolicyGraph.svg",
         "The decision graph with a sampling distribution over plans",
         "The decision graph of Chapter 9, with states x, actions u, dynamics "
         "and reward factors. Each action has one more factor, a Gaussian "
         "around a planned action that does not depend on the state. The "
         "planned actions form one parameter node, joined to all of these "
         "factors.")


# 2. Sampled rollouts on the weak motor, shaded by their weight.
def plot(fig, box, xlim, ylim, xticks, yticks, xlabel, ylabel):
    left, top, width, height = box

    def to(x, y):
        return (left + (x - xlim[0]) / (xlim[1] - xlim[0]) * width,
                top + height - (y - ylim[0]) / (ylim[1] - ylim[0]) * height)

    fig.raw(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
            f'fill="none" stroke="#d0d7de"/>')
    for x in xticks:
        px, py = to(x, ylim[0])
        fig.raw(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" '
                f'y2="{top + height}" stroke="#eaeef2"/>')
        fig.text(px, py + 16, f"{x:g}", EDGE, "middle", 11)
    for y in yticks:
        px, py = to(xlim[0], y)
        fig.raw(f'<line x1="{left}" y1="{py:.1f}" x2="{left + width}" '
                f'y2="{py:.1f}" stroke="#eaeef2"/>')
        fig.text(px - 6, py + 4, f"{y:g}", EDGE, "end", 11)
    fig.text(left + width / 2, top + height + 34, xlabel, EDGE, "middle", 12)
    fig.text(left, top - 10, ylabel, EDGE, "start", 12)
    return to


def polyline(fig, to, xs, ys, color, width, opacity=1.0, dash=None):
    points = " ".join(f"{px:.1f},{py:.1f}" for px, py in
                      (to(x, y) for x, y in zip(xs, ys)))
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    fig.raw(f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity:.2f}"{dashed}/>')


T, x_start, eta, variance = 4, 3.0, 0.5, 0.25


def states_of(plans):
    x = np.full(plans.shape[:-1], x_start)
    out = [x]
    for t in range(T):
        x = x + np.tanh(plans[..., t])
        out.append(x)
    return np.stack(out, axis=-1)


def returns_of(plans):
    s = states_of(plans)
    return -(np.sum(s[..., :-1] ** 2, -1) + np.sum(plans ** 2, -1) + s[..., -1] ** 2)


rng = np.random.default_rng(0)
plan = np.zeros(T)
for k in range(2):  # the samples of the second iteration are drawn
    plans = plan + rng.normal(0.0, np.sqrt(variance), (1000, T))
    returns = returns_of(plans)
    weights = np.exp(returns / eta - logsumexp(returns / eta))
    old_plan, plan = plan, weights @ plans

fig = Figure(780, 400, legend=False)
to = plot(fig, (70, 50, 420, 270), (0, 4), (-1, 3.2), [0, 1, 2, 3, 4],
          [-1, 0, 1, 2, 3], "step t", "state x_t")
shown = 80
strength = weights[:shown] / weights[:shown].max()
for states, s in zip(states_of(plans[:shown]), strength):
    polyline(fig, to, range(T + 1), states, RED, 1.5, 0.06 + 0.94 * s)
polyline(fig, to, range(T + 1), states_of(old_plan), MUTED, 2.5, dash="6 4")
polyline(fig, to, range(T + 1), states_of(plan), BLUE, 3)
legend = [(MUTED, "6 4", 1.0, "the current plan"),
          (RED, None, 0.35, "samples; darker = larger weight"),
          (BLUE, None, 1.0, "the new plan: their weighted mean")]
for row, (color, dash, opacity, label) in enumerate(legend):
    y = 90 + 28 * row
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    fig.raw(f'<line x1="520" y1="{y - 4}" x2="550" y2="{y - 4}" '
            f'stroke="{color}" stroke-width="2.5" stroke-opacity="{opacity}"{dashed}/>')
    fig.text(557, y, label, INK, "start", 12)
fig.save("MppiRollouts.svg", "Sampled rollouts of one MPPI iteration",
         "State against step for eighty rollouts sampled around the current "
         "plan on the weak motor, all starting at three. Rollouts with a "
         "high return are drawn darker. The rollout of their weighted mean "
         "runs through the dark ones and ends closer to the origin than the "
         "rollout of the current plan.")
print(old_plan, plan)
