# %% [markdown]
# # Chapter 21 examples: planning in learned models
#
# This notebook runs the examples of
# [Chapter 21](https://thduynguyen.github.io/gtsam/chapter21): how the error
# of a learned dynamics factor compounds along the chain, what short rollouts
# and ensembles do about it, and a small experiment in the style of PETS, an
# ensemble of learned models planned through with CEM.

# %%
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## A slightly wrong model of the line (Section 2)
#
# The line of Chapter 1, $x' = x + u + w$ with $w \sim N(0, 0.5)$, and a
# linear model fitted to 20 sampled transitions by least squares, as in
# Chapter 18.

# %%
SIGMA_W = 0.5


def sample_transitions(rng, count):
    """Two-move episodes from x0 ~ N(2, 1) with random actions u ~ N(0, 1)."""
    xs, us, ns = [], [], []
    while len(xs) < count:
        x = 2.0 + rng.normal()
        for t in range(2):
            u = rng.normal()
            n = x + u + np.sqrt(SIGMA_W) * rng.normal()
            xs.append(x), us.append(u), ns.append(n)
            x = n
    return np.array(xs[:count]), np.array(us[:count]), np.array(ns[:count])


def fit(x, u, x_next):
    """Least squares: the model (F, B), and its covariance."""
    Z = np.stack([x, u], axis=1)
    information = Z.T @ Z
    model = np.linalg.solve(information, Z.T @ x_next)
    residual = x_next - Z @ model
    variance = residual @ residual / (len(x) - 2)
    return model, variance * np.linalg.inv(information)


rng = np.random.default_rng(6)
x, u, x_next = sample_transitions(rng, 20)
model, covariance = fit(x, u, x_next)
print("learned model: F = %.3f, B = %.3f   (true: 1, 1)" % tuple(model))
assert np.allclose(model, [1.0806, 0.8388], atol=1e-3)

# %% [markdown]
# ## The error of the forward message grows with the horizon (Section 2)
#
# Start at $x_0 = 2$ and apply the same action $u = -0.2$ at every step. The
# true mean position after $h$ steps is $2 - 0.2 h$. The model's prediction is
# rolled forward $h$ times, each step feeding the next. An ensemble of five
# models, drawn from the estimate and its covariance, is rolled the same way.


# %%
def roll(F, B, steps, start=2.0, action=-0.2):
    """The mean position after some steps of the model x' = F x + B u."""
    position = start
    for _ in range(steps):
        position = F * position + B * action
    return position


members = np.random.default_rng(1).multivariate_normal(model, covariance, 5)
print("ensemble members (F, B):\n", members)
errors, spreads = {}, {}
for h in [1, 2, 3, 5, 8, 10]:
    true = roll(1.0, 1.0, h)
    predicted = roll(*model, h)
    ensemble = [roll(F, B, h) for F, B in members]
    errors[h], spreads[h] = abs(predicted - true), np.std(ensemble)
    print(f"h = {h:2d}: true {true:5.2f}   model {predicted:5.3f}   "
          f"error {errors[h]:.3f}   spread of the ensemble {spreads[h]:.3f}")
assert errors[10] > 9 * errors[1] and spreads[10] > 9 * spreads[1]

# %% [markdown]
# ## Short rollouts from real states (Section 4)
#
# The same question, the position at step 10, answered by rolling the model
# only the last $h$ steps, starting from the real mean position at step
# $10 - h$.

# %%
for h in [1, 3, 10]:
    start = roll(1.0, 1.0, 10 - h)  # a real state, h steps before the end
    predicted = roll(*model, h, start=start)
    print(f"model steps h = {h:2d}: starts at {start:4.1f}, predicts "
          f"{predicted:6.3f}, error {abs(predicted - roll(1.0, 1.0, 10)):.3f}")
branch_error = abs(roll(*model, 1, start=roll(1.0, 1.0, 9)) -
                   roll(1.0, 1.0, 10))
assert branch_error < 0.1 * errors[10]

# %% [markdown]
# ## The bound, checked on the endless track (Section 2)
#
# A dynamics table estimated from 20 sampled transitions for every cell and
# move. The forward messages of the true and the learned table drift apart by
# at most the one-step error $D_p$ per step, and the error of the return
# grows much faster than the horizon $1 / (1 - \gamma)$.

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])
dynamics = np.zeros((3, 2, 3))
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])
policy = np.full((3, 2), 0.5)  # the coin flip

rng = np.random.default_rng(0)
learned = np.zeros_like(dynamics)
for s in range(3):
    for a in range(2):
        outcomes = rng.choice(3, size=20, p=dynamics[s, a])
        learned[s, a] = np.bincount(outcomes, minlength=3) / 20
one_step_error = np.abs(learned - dynamics).sum(axis=2).max()
print("learned table p(s' | s, a):\n", learned.reshape(6, 3))
print("one-step error D_p =", one_step_error)

P_true = np.einsum("sa,sat->st", policy, dynamics)
P_model = np.einsum("sa,sat->st", policy, learned)
d_true, d_model = prior.copy(), prior.copy()
for h in range(1, 11):
    d_true, d_model = d_true @ P_true, d_model @ P_model
    distance = np.abs(d_true - d_model).sum()
    assert distance <= h * one_step_error
    if h in [1, 2, 5, 10]:
        print(f"h = {h:2d}: distance of the forward messages {distance:.3f}, "
              f"bound {h * one_step_error:.1f}")


def evaluate(table, gamma):
    """J of the coin flip, by solving the Bellman equation of Chapter 3."""
    P_pi = np.einsum("sa,sat->st", policy, table)
    r_pi = (policy * reward).sum(axis=1)
    return prior @ np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)


r_max = np.abs(reward).max()
for gamma in [0.5, 0.9, 0.99]:
    J_true, J_model = evaluate(dynamics, gamma), evaluate(learned, gamma)
    bound = r_max * one_step_error * gamma / (1 - gamma) ** 2
    print(f"gamma = {gamma}: horizon {1 / (1 - gamma):5.0f}, true J = "
          f"{J_true:7.3f}, model J = {J_model:7.3f}, error = "
          f"{abs(J_true - J_model):6.3f}, bound = {bound:.0f}")
    assert abs(J_true - J_model) <= bound
assert np.isclose(evaluate(dynamics, 0.9), 0.4385, atol=1e-4)

# %% [markdown]
# ## Planning in the model: the planner's gains (Section 3)
#
# Stage 1 in a learned model, in a receding horizon: at each move, find the
# action sequence for the remaining moves that the model, or the average of
# the ensemble's members, rates highest, and apply its first action. For
# linear models and quadratic rewards the best first action is linear in the
# position, $u = -K_t x$, and the gain has a closed form. It is used below to
# evaluate planners exactly; the CEM planner further down must agree with it.


# %%
def planner_gains(models):
    """Gains of receding-horizon planning with the average over the models."""
    models = np.atleast_2d(models)
    F, B = models[:, 0], models[:, 1]
    # Two moves left: minimize over (u0, u1), from x = 1, the average of
    # u0^2 + x1^2 + u1^2 + x2^2 with x1 = F + B u0 and x2 = F x1 + B u1.
    rows, targets = [np.eye(2)], [np.zeros(2)]
    for f, b in models:
        rows.append(np.array([[b, 0.0], [f * b, b]]) / np.sqrt(len(models)))
        targets.append(-np.array([f, f * f]) / np.sqrt(len(models)))
    plan = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets),
                           rcond=None)[0]
    # One move left: minimize u^2 + average of (F + B u)^2.
    return {0: -plan[0], 1: (F * B).mean() / (1 + (B * B).mean())}


def model_return(gains, F=1.0, B=1.0):
    """J of u_t = -K_t x_t under the model x' = F x + B u + w, exactly."""
    second, total = 5.0, 0.0  # E[x0^2] = 2^2 + 1
    for t in range(2):
        total += (1 + gains[t] ** 2) * second
        second = (F - B * gains[t]) ** 2 * second + SIGMA_W
    return -(total + second)


gains = planner_gains([1.0, 1.0])
print("planning in the true model: gains", gains, " J =", model_return(gains))
assert np.isclose(gains[0], 0.6) and np.isclose(gains[1], 0.5)
assert np.isclose(model_return(gains), -9.25)

# %% [markdown]
# ## The planner exploits the model; an ensemble does not (Section 5)
#
# Over 1000 independent data sets of each size: the return that the planner's
# own model promises for its plan, and the return that the plan collects on
# the real line. The ensemble has five members, each fitted to a resampling
# of the data (a bootstrap).


# %%
def bootstrap(rng, x, u, x_next, size=5):
    """An ensemble: refit the model to resampled transitions."""
    members = []
    while len(members) < size:
        index = rng.integers(len(x), size=len(x))
        Z = np.stack([x[index], u[index]], axis=1)
        if np.linalg.cond(Z.T @ Z) < 1e8:
            members.append(np.linalg.solve(Z.T @ Z, Z.T @ x_next[index]))
    return np.array(members)


medians = {}
for count in [6, 10, 20, 50]:
    rng = np.random.default_rng(5)
    rows = []
    for trial in range(1000):
        x, u, x_next = sample_transitions(rng, count)
        single, _ = fit(x, u, x_next)
        members = bootstrap(rng, x, u, x_next)
        gains_single = planner_gains(single)
        gains_ensemble = planner_gains(members)
        rows.append((
            model_return(gains_single, *single), model_return(gains_single),
            np.mean([model_return(gains_ensemble, F, B) for F, B in members]),
            model_return(gains_ensemble)))
    medians[count] = np.median(rows, axis=0)
    print(f"M = {count:2d}:  one model: predicted {medians[count][0]:6.2f}, "
          f"true {medians[count][1]:6.2f}   |   ensemble: predicted "
          f"{medians[count][2]:6.2f}, true {medians[count][3]:6.2f}")
# One model promises more than the best possible return, -9.25, and delivers
# less. The ensemble's promise is below what it delivers.
assert medians[6][0] > -9.25 > medians[6][1]
assert medians[6][2] < medians[6][3]

# %% [markdown]
# ## A small PETS: an ensemble, CEM, and a receding horizon (Section 6)
#
# One data set of 10 transitions, an ensemble of five models, and the
# cross-entropy method of Chapter 10 as the planner. All episodes are planned
# at once, one row per episode.


# %%
def cem(rng, positions, models, horizon, population=200, elites=20,
        iterations=6):
    """For each position, the first action of the best sampled plan."""
    count = len(positions)
    mean, std = np.zeros((count, horizon)), np.full((count, horizon), 2.0)
    for _ in range(iterations):
        plans = mean[:, None, :] + std[:, None, :] * rng.normal(
            size=(count, population, horizon))
        score = np.zeros((count, population))
        for F, B in np.atleast_2d(models):  # average over the members
            x = np.repeat(positions[:, None], population, axis=1)
            total = np.zeros((count, population))
            for t in range(horizon):
                total -= x ** 2 + plans[:, :, t] ** 2
                x = F * x + B * plans[:, :, t]
            score += (total - x ** 2) / len(np.atleast_2d(models))
        best = np.argsort(-score, axis=1)[:, :elites]
        chosen = np.take_along_axis(plans, best[:, :, None], axis=1)
        mean, std = chosen.mean(axis=1), chosen.std(axis=1) + 1e-6
    return mean[:, 0]


def run_on_real_line(models, episodes=10000, seed=7):
    """Receding-horizon control with CEM, on the real line. Returns J."""
    rng = np.random.default_rng(seed)
    x = 2.0 + rng.normal(size=episodes)
    total, used = np.zeros(episodes), []
    for t in range(2):
        u = cem(rng, x, models, horizon=2 - t)
        used.append(-(x @ u) / (x @ x))  # the gain CEM effectively applied
        total -= x ** 2 + u ** 2
        x = x + u + np.sqrt(SIGMA_W) * rng.normal(size=episodes)
    return (total - x ** 2).mean(), used


rng = np.random.default_rng(11)
x, u, x_next = sample_transitions(rng, 10)
single, _ = fit(x, u, x_next)
members = bootstrap(rng, x, u, x_next)
print("single model (F, B):", single)
print("ensemble members (F, B):\n", members)
results = {}
for name, models in [("true model", np.array([1.0, 1.0])),
                     ("one learned model", single),
                     ("ensemble of 5", members)]:
    exact_gains = planner_gains(models)
    sampled_J, cem_gains = run_on_real_line(models)
    results[name] = model_return(exact_gains)
    print(f"{name:18s}: planner gains ({exact_gains[0]:.3f}, "
          f"{exact_gains[1]:.3f}), gains applied by CEM ({cem_gains[0]:.3f}, "
          f"{cem_gains[1]:.3f}), true J exact {results[name]:.3f}, "
          f"sampled with CEM {sampled_J:.3f}")
    assert np.allclose([cem_gains[0], cem_gains[1]],
                       [exact_gains[0], exact_gains[1]], atol=0.03)
assert np.isclose(results["true model"], -9.25)
assert results["one learned model"] > -9.6 and results["ensemble of 5"] > -9.6
