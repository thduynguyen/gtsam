"""Figures of Chapter 9. Run with the notebook environment's python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import least_squares
from bookfig import Figure, chain, EDGE, INK, MUTED, RED, VALUE_TEXT, LEARNED_TEXT

BLUE, GREEN = "#1a4fa0", "#1e7b34"


def plot(fig, box, xlim, ylim, xticks, yticks, xlabel, ylabel):
    """Axes inside box = (left, top, width, height); returns a mapper."""
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


def curve(fig, to, xs, ys, color, width=2.5, dash=None, dots=False):
    points = " ".join(f"{px:.1f},{py:.1f}" for px, py in
                      (to(x, y) for x, y in zip(xs, ys)))
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    fig.raw(f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{dashed}/>')
    if dots:
        for x, y in zip(xs, ys):
            px, py = to(x, y)
            fig.raw(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" '
                    f'fill="{color}"/>')


# ---- numbers (the same computations as the notebook) ----
T, x_start = 4, 3.0
f = lambda x, u: x + np.tanh(u)


def rollout(controls, x0):
    states = [x0]
    for u in controls:
        states.append(f(states[-1], u))
    return np.array(states)


def total_reward(controls, x0):
    s = rollout(controls, x0)
    return -(np.sum(s[:-1] ** 2) + np.sum(np.square(controls)) + s[-1] ** 2)


def backward(states, controls):
    moves = len(controls)
    gains, change = np.zeros(moves), np.zeros(moves)
    P, grad = 1.0, -2 * states[-1]
    for t in reversed(range(moves)):
        B = 1 - np.tanh(controls[t]) ** 2
        Q_x, Q_u = -2 * states[t] + grad, -2 * controls[t] + B * grad
        H_uu, H_ux, H_xx = 1 + B * P * B, B * P, 1 + P
        gains[t], change[t] = H_ux / H_uu, 0.5 * Q_u / H_uu
        grad, P = Q_x - gains[t] * Q_u, H_xx - H_ux ** 2 / H_uu
    return gains, change


def ilqr(x0, moves, record=None):
    controls = np.zeros(moves)
    for _ in range(100):
        states, current = rollout(controls, x0), total_reward(controls, x0)
        if record is not None:
            record.append(states)
        gains, change = backward(states, controls)
        step = 1.0
        while True:
            x, cand = x0, np.zeros(moves)
            for t in range(moves):
                cand[t] = controls[t] + step * change[t] - gains[t] * (x - states[t])
                x = f(x, cand[t])
            gain = total_reward(cand, x0) - current
            if gain > 0 or step < 1e-8:
                break
            step /= 2
        if gain > 0:
            controls = cand
        if gain < 1e-8:
            break
    return controls


def map_first(x0, moves, variance):
    def residuals(z):
        u, x = z[:moves], np.concatenate([[x0], z[moves:]])
        return np.concatenate([x, u, (x[1:] - f(x[:-1], u)) / np.sqrt(2 * variance)])
    start = np.concatenate([np.zeros(moves), np.full(moves, x0)])
    return least_squares(residuals, start, xtol=1e-12, ftol=1e-12).x[0]


# 1. The graph.
fig = Figure(900, 400)
where = chain(fig, 2, y_state=130, y_action=270, state="x", action="u",
              policy=False, final=False)
fig.text(790, 136, "…", EDGE, "middle", 22)
fig.text(450, 40, "p(x_{t+1} | x_t, u_t) = N(x_{t+1}; f(x_t, u_t), Σ_w),  "
         "here f(x, u) = x + tanh(u): not linear in u", INK, "middle", 13)
fig.save("NonlinearChain.svg", "The decision graph with nonlinear dynamics",
         "The decision graph of Chapter 4, with states x and actions u, a "
         "dynamics factor and a reward factor per step and no policy "
         "factors. The dynamics factor is a Gaussian whose mean f of x and u "
         "is not a linear function.")

# 2. Linearizing the dynamics factor.
fig = Figure(760, 400, legend=False)
to = plot(fig, (70, 50, 620, 270), (-3, 3), (-1.6, 1.6), [-3, -2, -1, 0, 1, 2, 3],
          [-1, 0, 1], "command u", "displacement of one move")
us = np.linspace(-3, 3, 121)
curve(fig, to, us, np.tanh(us), BLUE)
for ubar, color, dy in [(0.0, MUTED, -1), (-1.27, RED, 1)]:
    slope = 1 - np.tanh(ubar) ** 2
    span = np.array([max(-3, ubar - 1.6), min(3, ubar + 1.6)])
    curve(fig, to, span, np.tanh(ubar) + slope * (span - ubar), color, 2, "6 4")
    px, py = to(ubar, np.tanh(ubar))
    fig.raw(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
px, py = to(0.0, 0.0)
fig.text(px + 12, py + 22, "linearized at u = 0: slope 1", MUTED, "start", 12)
px, py = to(-2.9, -0.4)
fig.text(px, py, "linearized at u = −1.27: slope 0.27", RED, "start", 12)
px, py = to(2.2, np.tanh(2.2))
fig.text(px, py + 24, "tanh(u): the true motor", BLUE, "middle", 12)
fig.save("WeakMotorLinearization.svg", "The motor curve and two linearizations",
         "The displacement tanh of u as a function of the command u, an "
         "S-shaped curve that levels off at plus and minus one. Two tangent "
         "lines: at u equal to zero, with slope one, and at u equal to minus "
         "1.27, with slope 0.27.")

# 3. The trajectories of the iterations.
record = []
solution = ilqr(x_start, T, record)
fig = Figure(760, 400, legend=False)
to = plot(fig, (70, 50, 430, 270), (0, 4), (0, 3.2), [0, 1, 2, 3, 4],
          [0, 1, 2, 3], "step t", "state x_t")
styles = [(0, MUTED, "start: do nothing"),
          (1, "#d9730d", "after 1 pass"),
          (2, GREEN, "after 2 passes"),
          (len(record) - 1, BLUE, "converged, after 13 passes")]
for row, (index, color, label) in enumerate(styles):
    curve(fig, to, range(T + 1), record[index], color, 2.5, dots=True)
    y = 90 + 28 * row
    fig.raw(f'<line x1="525" y1="{y - 4}" x2="555" y2="{y - 4}" '
            f'stroke="{color}" stroke-width="2.5"/>')
    fig.text(562, y, label, INK, "start", 12)
fig.save("IlqrIterations.svg", "The trajectories of the iLQR iterations",
         "The state at steps zero to four for four iterations. The first "
         "trajectory stays at three. After one pass it drops to 0.3 at the "
         "last step, overshooting; after two passes it is close to the "
         "converged trajectory, which ends at 0.41.")

# 4. The MPC policies of iLQR and of MAP.
grid = np.linspace(-1, 5, 61)
ilqr_policy = [ilqr(x, T)[0] for x in grid]
map_policy = [map_first(x, T, 0.2) for x in grid]
fig = Figure(760, 400, legend=False)
to = plot(fig, (70, 50, 430, 270), (-1, 5), (-1.8, 0.8), [-1, 0, 1, 2, 3, 4, 5],
          [-1.5, -1, -0.5, 0, 0.5], "state x", "first control of the plan")
curve(fig, to, grid, ilqr_policy, BLUE)
curve(fig, to, grid, map_policy, RED)
for row, (color, label) in enumerate([
        (BLUE, "iLQR: average over the slip"),
        (RED, "MAP: maximum over the slip")]):
    y = 90 + 28 * row
    fig.raw(f'<line x1="525" y1="{y - 4}" x2="555" y2="{y - 4}" '
            f'stroke="{color}" stroke-width="2.5"/>')
    fig.text(562, y, label, INK, "start", 12)
fig.text(525, 170, "MAP commands less: it counts", EDGE, "start", 12)
fig.text(525, 186, "on slips toward the origin.", EDGE, "start", 12)
fig.save("MpcPolicies.svg", "The first control of the plan, as a function of the state",
         "Two curves of the first control against the state, both passing "
         "through zero at the origin and negative for positive states. The "
         "curve of iLQR is steeper: at state three it commands minus 1.27, "
         "the curve of MAP minus 0.97.")
print(len(record), solution)
