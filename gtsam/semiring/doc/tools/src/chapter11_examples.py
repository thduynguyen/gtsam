# %% [markdown]
# # Chapter 11 examples: Monte Carlo messages
#
# This notebook runs the examples of
# [Chapter 11](https://thduynguyen.github.io/gtsam/chapter11). The dynamics of
# the track are hidden inside a simulator. The forward message becomes a set
# of sampled states, the backward message a set of sampled returns, and the
# gradient of Chapter 5 is estimated from them and compared with its exact
# value.

# %%
import itertools

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track, as a simulator (Section 1)
#
# The tables are those of Chapter 1. The algorithm of this chapter never reads
# `dynamics`: it only calls `rollouts`, which samples from it. The tables are
# used again at the end of each section, to compute the exact answer that the
# samples are compared with.

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
move_reward = np.array([[0.0, -1.0]] * 3)  # r(s, a)
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sample(probabilities, rng):
    """One sample per row of a table of probabilities."""
    u = rng.random(len(probabilities))
    return (u[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)


def rollouts(theta, M, rng):
    """Run the policy pi_theta(R | s) = sigmoid(theta_s) M times.

    Returns the states s[i, t], the actions a[i, t] and the rewards r[i, t]:
    two moves, and the final reward in the last column of r.
    """
    s = np.zeros((M, 3), dtype=int)
    a = np.zeros((M, 2), dtype=int)
    r = np.zeros((M, 3))
    s[:, 0] = sample(np.tile(prior, (M, 1)), rng)
    for t in range(2):
        a[:, t] = rng.random(M) < sigmoid(theta[s[:, t]])  # 1 = Right
        r[:, t] = move_reward[s[:, t], a[:, t]]
        s[:, t + 1] = sample(dynamics[s[:, t], a[:, t]], rng)
    r[:, 2] = final_reward[s[:, 2]]
    return s, a, r


theta = np.zeros(3)  # the coin flip
rng = np.random.default_rng(0)
s, a, r = rollouts(theta, 5, rng)
for i in range(5):
    print(f"rollout {i}: s = {s[i]}, a = {'LR'[a[i, 0]]}{'LR'[a[i, 1]]}, "
          f"rewards = {r[i]}, return = {r[i].sum():.0f}")

# %% [markdown]
# ## The exact messages, for comparison
#
# Stage 1 of Chapter 5, by the Bellman backup on the tables.


# %%
def exact_messages(theta):
    """J, Q_t, V_t, A_t and d_t for the logistic policy, from the tables."""
    right = sigmoid(theta)
    policy = np.stack([1 - right, right], axis=1)
    V = {2: final_reward}
    Q, A = {}, {}
    for t in [1, 0]:
        Q[t] = move_reward + dynamics @ V[t + 1]
        V[t] = (policy * Q[t]).sum(axis=1)
        A[t] = Q[t] - V[t][:, None]
    d = {0: prior}
    d[1] = np.einsum("s,sa,sat->t", d[0], policy, dynamics)
    d[2] = np.einsum("s,sa,sat->t", d[1], policy, dynamics)
    return prior @ V[0], Q, V, A, d


def exact_gradient(theta):
    """Forward message times local derivative times backward message."""
    _, _, _, A, d = exact_messages(theta)
    slope = sigmoid(theta) * (1 - sigmoid(theta))
    return sum(d[t] * slope * (A[t][:, R] - A[t][:, L]) for t in range(2))


J, Q, V, A, d = exact_messages(theta)
print("J =", J)
print("exact gradient =", exact_gradient(theta))
assert np.isclose(J, 1.4)
assert np.allclose(exact_gradient(theta), [0.15, 1.0, 0.35])

# %% [markdown]
# ## Forward message: particles (Section 2)
#
# The fraction of rollouts that are in cell $s$ at step $t$ estimates
# $d_t(s)$.

# %%
rng = np.random.default_rng(0)
for M in [100, 10000]:
    s, a, r = rollouts(theta, M, rng)
    particles = np.stack([np.bincount(s[:, t], minlength=3) / M
                          for t in range(3)])
    print(f"M = {M}: d_0, d_1, d_2 estimated by particles\n{particles}")
print("exact d_1 =", d[1], " exact d_2 =", d[2])
assert np.allclose(particles[1], d[1], atol=0.02)

# %% [markdown]
# ## Backward message: sampled returns (Section 2)
#
# The return from step $t$ on, $R_t$, is one sample of $Q_t(s_t, a_t)$.
# Averaging it over the rollouts that pass through $(s, a)$ estimates $Q_t$.

# %%
returns_to_go = np.stack([r.sum(axis=1), r[:, 1:].sum(axis=1)], axis=1)
for t in range(2):
    estimate = np.zeros((3, 2))
    for cell, move in itertools.product(range(3), range(2)):
        through = (s[:, t] == cell) & (a[:, t] == move)
        estimate[cell, move] = (returns_to_go[through, t].mean()
                                if through.any() else np.nan)
    print(f"Q_{t} from {M} rollouts =\n{estimate}\nexact Q_{t} =\n{Q[t]}")
    visited = ~np.isnan(estimate)
    assert np.allclose(estimate[visited], Q[t][visited], atol=0.3)

# A single return is a poor estimate: its spread around Q_0(1, R).
through = (s[:, 0] == 1) & (a[:, 0] == R)
print("returns after (s_0, a_0) = (1, R): mean",
      returns_to_go[through, 0].mean().round(3), " standard deviation",
      returns_to_go[through, 0].std().round(3))

# %% [markdown]
# ## REINFORCE: the gradient from rollouts (Section 3)
#
# Each rollout gives one sample of the gradient: the sum over its steps of the
# score term $g_t$ times a sampled value. Three choices of the value are
# compared: the whole return, the return from step $t$ on, and the latter
# minus a baseline $V_t(s_t)$.


# %%
def gradient_samples(theta, s, a, r, value):
    """One gradient sample per rollout, for one choice of the value."""
    M = len(s)
    samples = np.zeros((M, 3))
    whole = r.sum(axis=1)
    to_go = np.stack([whole, r[:, 1:].sum(axis=1)], axis=1)
    for t in range(2):
        # The score term of step t: d log pi(a_t | s_t) / d theta_{s_t}.
        g = a[:, t] - sigmoid(theta[s[:, t]])
        if value == "whole return":
            weight = whole
        elif value == "return from t":
            weight = to_go[:, t]
        else:  # return from t minus the baseline V_t(s_t)
            weight = to_go[:, t] - V[t][s[:, t]]
        np.add.at(samples, (np.arange(M), s[:, t]), g * weight)
    return samples


values = ["whole return", "return from t", "return from t minus baseline"]
rng = np.random.default_rng(0)
s, a, r = rollouts(theta, 200000, rng)
for value in values:
    samples = gradient_samples(theta, s, a, r, value)
    print(f"{value:30s} mean {samples.mean(axis=0)}  "
          f"std per rollout {samples.std(axis=0)}")
    assert np.allclose(samples.mean(axis=0), [0.15, 1.0, 0.35], atol=0.03)

# %% [markdown]
# The means agree with the exact gradient $(0.15, 1.0, 0.35)$; the spreads
# differ. Since the track is small, the spread of one rollout can also be
# computed exactly, by enumerating all trajectories.

# %%
right = sigmoid(theta)
policy = np.stack([1 - right, right], axis=1)
exact_std = {}
for value in values:
    mean, second = np.zeros(3), np.zeros(3)
    for s0, a0, s1, a1, s2 in itertools.product(
            range(3), range(2), range(3), range(2), range(3)):
        p = (prior[s0] * policy[s0, a0] * dynamics[s0, a0, s1] *
             policy[s1, a1] * dynamics[s1, a1, s2])
        if p == 0:
            continue
        one = gradient_samples(
            theta, np.array([[s0, s1, s2]]), np.array([[a0, a1]]),
            np.array([[move_reward[s0, a0], move_reward[s1, a1],
                       final_reward[s2]]]), value)[0]
        mean += p * one
        second += p * one ** 2
    exact_std[value] = np.sqrt(second - mean ** 2)
    print(f"{value:30s} exact mean {mean}  exact std {exact_std[value]}")
    assert np.allclose(mean, [0.15, 1.0, 0.35])
assert np.all(exact_std["return from t minus baseline"]
              < exact_std["return from t"])
assert np.allclose(exact_std["whole return"], [1.1948, 1.9170, 1.3793],
                   atol=1e-3)
assert np.allclose(exact_std["return from t"], [1.1790, 2.0341, 1.5500],
                   atol=1e-3)
assert np.allclose(exact_std["return from t minus baseline"],
                   [1.1440, 1.6167, 0.9434], atol=1e-3)

# %% [markdown]
# ## The error shrinks like one over the square root of M (Section 4)

# %%
rng = np.random.default_rng(1)
exact = np.array([0.15, 1.0, 0.35])
for M in [10, 100, 1000, 10000, 100000]:
    errors = {value: [] for value in values}
    for repeat in range(20):
        s, a, r = rollouts(theta, M, rng)
        for value in values:
            estimate = gradient_samples(theta, s, a, r, value).mean(axis=0)
            errors[value].append(np.linalg.norm(estimate - exact))
    print(f"M = {M:6d}: average error  " + "  ".join(
        f"{value}: {np.mean(errors[value]):.4f}" for value in values))
    last = {value: np.mean(errors[value]) for value in values}
assert last["return from t minus baseline"] < 0.02

# %% [markdown]
# ## A baseline costs no bias (Section 3)
#
# For any function $b(s)$, the average of $g_t\, b(s_t)$ is zero, because the
# derivatives of the policy sum to zero over the actions. Even an absurd
# baseline leaves the mean unchanged; it only changes the spread.

# %%
s, a, r = rollouts(theta, 200000, np.random.default_rng(2))
to_go = np.stack([r.sum(axis=1), r[:, 1:].sum(axis=1)], axis=1)
for name, baseline in [("b = 0", np.zeros(3)), ("b = V_t (exact)", None),
                       ("b = (50, -20, 7)", np.array([50.0, -20.0, 7.0]))]:
    samples = np.zeros((len(s), 3))
    for t in range(2):
        b = V[t] if baseline is None else baseline
        g = a[:, t] - sigmoid(theta[s[:, t]])
        np.add.at(samples, (np.arange(len(s)), s[:, t]),
                  g * (to_go[:, t] - b[s[:, t]]))
    print(f"{name:18s} mean {samples.mean(axis=0)}  "
          f"std {samples.std(axis=0)}")

# %% [markdown]
# ## The two stages, with sampled messages (Section 5)
#
# Stage 1 runs $M$ rollouts. Stage 2 takes a gradient step. The baseline is
# estimated from the same rollouts, as the average return from each cell at
# each step. The expected return of each iterate is evaluated exactly, for
# reporting only.


# %%
def reinforce_step(theta, M, rng):
    """One sampled gradient, with a baseline estimated from the rollouts."""
    s, a, r = rollouts(theta, M, rng)
    to_go = np.stack([r.sum(axis=1), r[:, 1:].sum(axis=1)], axis=1)
    gradient = np.zeros(3)
    for t in range(2):
        counts = np.bincount(s[:, t], minlength=3)
        sums = np.bincount(s[:, t], weights=to_go[:, t], minlength=3)
        baseline = sums / np.maximum(counts, 1)  # average return per cell
        g = a[:, t] - sigmoid(theta[s[:, t]])
        np.add.at(gradient, s[:, t], g * (to_go[:, t] - baseline[s[:, t]]))
    return gradient / M


rng = np.random.default_rng(0)
theta_k = np.zeros(3)
history = []
for k in range(201):
    history.append(exact_messages(theta_k)[0])
    theta_k = theta_k + 0.5 * reinforce_step(theta_k, 100, rng)
for k in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
    print(f"iteration {k:3d}: J = {history[k]:.4f}")
print("pi(R | s) =", sigmoid(theta_k))
assert history[-1] > 5.8

# The exact two-stage loop of Chapter 5, with the same step size.
theta_e = np.zeros(3)
exact_history = []
for k in range(201):
    exact_history.append(exact_messages(theta_e)[0])
    theta_e = theta_e + 0.5 * exact_gradient(theta_e)
for k in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
    print(f"iteration {k:3d}: exact messages J = {exact_history[k]:.4f}")
