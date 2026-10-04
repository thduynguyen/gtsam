# %% [markdown]
# # Chapter 15 examples: value-based control
#
# This notebook runs the examples of
# [Chapter 15](https://thduynguyen.github.io/gtsam/chapter15) on the endless
# track of Chapter 3: SARSA, Q-learning, fitted Q iteration and a small
# version of DQN. The dynamics table is used only to *sample* transitions,
# and to compute the exact answers the samples are checked against.

# %%
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track and its exact values (Section 1)

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
coin_flip = np.full((3, 2), 0.5)


def exact_Q(policy):
    """Q of a stationary policy, from the linear system of Chapter 3."""
    P_pi = np.einsum("sa,sat->st", policy, dynamics)
    r_pi = (policy * reward).sum(axis=1)
    V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
    return reward + gamma * dynamics @ V


def value_iteration(sweeps):
    """Q after some sweeps of value iteration (Chapter 4), from Q = 0."""
    Q = np.zeros((3, 2))
    history = []
    for _ in range(sweeps):
        Q = reward + gamma * dynamics @ Q.max(axis=1)
        history.append(Q)
    return history


Q_coin = exact_Q(coin_flip)
Q_star = value_iteration(500)[-1]
print("Q of the coin flip =\n", Q_coin)
print("Q* =\n", Q_star)
assert np.allclose(Q_coin, [[-0.2023, -0.2472], [0.0365, 2.1669],
                            [3.5354, 4.7108]], atol=1e-4)
assert np.allclose(Q_star, [[4.8775, 5.4194], [5.2629, 7.5610],
                            [9.2439, 10.0]], atol=1e-4)

# %% [markdown]
# ## A simulator: the dynamics factor through samples (Section 1)
#
# From here on the algorithms never read the table `dynamics`. They call
# `step`, which returns one sampled next state. After each move the episode
# ends with probability $1 - \gamma$ (Chapter 3), and the robot is put back
# at the start.

# %%
def step(rng, s, a):
    """One sampled transition: the reward and the next state."""
    return reward[s, a], rng.choice(3, p=dynamics[s, a])


def restart(rng, s_next):
    """The state the robot acts in next: s' if the episode continues."""
    return s_next if rng.random() < gamma else rng.choice(3, p=prior)


def collect(rng, policy, count):
    """A stream of transitions (s, a, r, s') under a policy."""
    data = np.zeros((count, 4))
    s = rng.choice(3, p=prior)
    for i in range(count):
        a = rng.choice(2, p=policy[s])
        r, s_next = step(rng, s, a)
        data[i] = s, a, r, s_next
        s = restart(rng, s_next)
    return data


rng = np.random.default_rng(0)
data = collect(rng, coin_flip, 200_000)
counts = np.zeros((3, 2))
np.add.at(counts, (data[:, 0].astype(int), data[:, 1].astype(int)), 1)
print("first transitions (s, a, r, s'):\n", data[:5])
print("share of each (s, a) in the data =\n", counts / counts.sum())
P_coin = np.einsum("sa,sat->st", coin_flip, dynamics)
d = np.linalg.solve((np.eye(3) - gamma * P_coin).T, prior)
print("discounted visitation d / 10, split over the two moves =",
      d / d.sum() / 2)
assert np.allclose(counts / counts.sum(), (d / d.sum() / 2)[:, None],
                   atol=0.001)

# %% [markdown]
# ## SARSA and Q-learning (Section 2)
#
# Both replay the same stream of coin-flip transitions. They differ in one
# line: the value used for the next state.

# %%
def td_control(data, rule, rng, exponent=0.6):
    """Tabular TD on a stream of transitions, with a decaying step size."""
    Q = np.zeros((3, 2))
    visits = np.zeros((3, 2))
    for s, a, r, s_next in data:
        s, a, s_next = int(s), int(a), int(s_next)
        if rule == "sarsa":  # the next action of the coin flip, sampled
            future = Q[s_next, rng.choice(2)]
        elif rule == "expected sarsa":  # the average under the coin flip
            future = coin_flip[s_next] @ Q[s_next]
        else:  # Q-learning: the maximum
            future = Q[s_next].max()
        visits[s, a] += 1
        alpha = visits[s, a] ** -exponent
        Q[s, a] += alpha * (r + gamma * future - Q[s, a])
    return Q


rng = np.random.default_rng(1)
Q_sarsa = td_control(data, "sarsa", rng)
Q_expected = td_control(data, "expected sarsa", rng)
Q_learning = td_control(data, "q-learning", rng)
print("SARSA =\n", Q_sarsa)
print("largest error against Q of the coin flip:",
      np.abs(Q_sarsa - Q_coin).max())
print("expected SARSA: largest error",
      np.abs(Q_expected - Q_coin).max())
print("Q-learning =\n", Q_learning)
print("largest error against Q*:", np.abs(Q_learning - Q_star).max())
print("greedy policy of Q-learning (0 = L, 1 = R):", Q_learning.argmax(axis=1))
assert np.abs(Q_sarsa - Q_coin).max() < 0.15
assert np.abs(Q_expected - Q_coin).max() < 0.1
assert np.abs(Q_learning - Q_star).max() < 0.1
assert list(Q_learning.argmax(axis=1)) == [R, R, R]

# %% [markdown]
# ## SARSA as control (Section 3)
#
# Stage 2 is greedy: after every update the policy becomes "the best move of
# the current table, except for a random move 10% of the time", and the next
# sample is drawn from that policy.

# %%
def sarsa_control(rng, count, explore=0.1, exponent=0.6):
    Q, visits = np.zeros((3, 2)), np.zeros((3, 2))

    def act(s):
        if rng.random() < explore:
            return rng.choice(2)
        return int(Q[s].argmax())

    s = rng.choice(3, p=prior)
    a = act(s)
    for _ in range(count):
        r, s_next = step(rng, s, a)
        a_next = act(s_next)
        visits[s, a] += 1
        Q[s, a] += visits[s, a] ** -exponent * (
            r + gamma * Q[s_next, a_next] - Q[s, a])
        if rng.random() < gamma:  # the episode continues
            s, a = s_next, a_next
        else:  # it ends: back to the start
            s = rng.choice(3, p=prior)
            a = act(s)
    return Q


def epsilon_greedy(Q, explore=0.1):
    policy = np.full((3, 2), explore / 2)
    policy[np.arange(3), Q.argmax(axis=1)] += 1 - explore
    return policy


Q_control = sarsa_control(np.random.default_rng(2), 200_000)
Q_target = exact_Q(epsilon_greedy(Q_control))
print("SARSA with a greedy Stage 2 =\n", Q_control)
print("exact Q of its own policy =\n", Q_target)
print("greedy moves:", Q_control.argmax(axis=1))
assert list(Q_control.argmax(axis=1)) == [R, R, R]
assert np.abs(Q_control - Q_target).max() < 0.25

# %% [markdown]
# ## Importance weights (Section 4)
#
# The data come from the coin flip $\pi_D$; the policy to evaluate, $\pi$,
# moves Right with probability 0.9. All averages below are computed exactly
# from the tables, so the differences are not sampling noise.

# %%
target_policy = np.tile([0.1, 0.9], (3, 1))
Q_pi = exact_Q(target_policy)
V_pi = (target_policy * Q_pi).sum(axis=1)

# One step: r + gamma * (average of Q under pi at s'), with s' from the data.
one_step = reward + gamma * dynamics @ V_pi

# Two steps: r + gamma r' + gamma^2 V(s''), with a' taken by the data policy.
inner = reward + gamma * dynamics @ V_pi  # value of (s', a')
two_step_plain = reward + gamma * dynamics @ (coin_flip * inner).sum(axis=1)
rho = target_policy / coin_flip  # pi(a' | s') / pi_D(a' | s')
two_step_weighted = reward + gamma * dynamics @ (
    coin_flip * rho * inner).sum(axis=1)
print("Q of pi =\n", Q_pi)
print("one-step target, averaged =\n", one_step)
print("two-step target, averaged, no weights =\n", two_step_plain)
print("two-step target, averaged, with weights =\n", two_step_weighted)
print("importance weights rho =\n", rho)
assert np.allclose(one_step, Q_pi)
assert np.allclose(two_step_weighted, Q_pi)
assert np.abs(two_step_plain - Q_pi).max() > 0.5

# %% [markdown]
# ## Fitted Q iteration (Section 5)
#
# A batch with five transitions per pair $(s, a)$, whose next states occur in
# exactly the proportions of the dynamics table. With a table as the
# regressor, each fit is one sweep of value iteration.

# %%
batch = []
for s in range(3):
    for a in range(2):
        for s_next in range(3):
            for _ in range(int(round(5 * dynamics[s, a, s_next]))):
                batch.append((s, a, reward[s, a], s_next))
batch = np.array(batch)
print("batch size:", len(batch))


def fit_table(batch, targets):
    """Least squares with one parameter per (s, a): the mean target."""
    total, count = np.zeros((3, 2)), np.zeros((3, 2))
    index = (batch[:, 0].astype(int), batch[:, 1].astype(int))
    np.add.at(total, index, targets)
    np.add.at(count, index, 1)
    return total / count


def fitted_q_iteration(batch, sweeps):
    Q = np.zeros((3, 2))
    history = []
    for _ in range(sweeps):
        targets = batch[:, 2] + gamma * Q[batch[:, 3].astype(int)].max(axis=1)
        Q = fit_table(batch, targets)  # the regression
        history.append(Q)
    return history


fitted = fitted_q_iteration(batch, 200)
exact = value_iteration(200)
for k in [1, 2, 5, 100]:
    print(f"sweep {k:3d}: V = {fitted[k - 1].max(axis=1)}")
    assert np.allclose(fitted[k - 1], exact[k - 1])
assert np.allclose(fitted[-1], Q_star, atol=1e-4)

# The same on 2000 sampled transitions: value iteration on the empirical model.
sample = data[:2000]
Q_fitted = fitted_q_iteration(sample, 200)[-1]
print("fitted Q iteration on 2000 samples =\n", Q_fitted)
print("largest error against Q*:", np.abs(Q_fitted - Q_star).max())
assert np.abs(Q_fitted - Q_star).max() < 0.5
assert list(Q_fitted.argmax(axis=1)) == [R, R, R]

# %% [markdown]
# ## A small DQN (Section 6)
#
# The three ingredients of DQN, with a table in place of the network: a
# replay buffer, minibatch gradient steps on the squared residual, and a
# frozen copy of the table for the targets, refreshed every 100 steps.

# %%
def small_dqn(rng, steps, batch_size=32, step_size=0.1, refresh=100,
              explore=0.2, capacity=5000):
    Q = np.zeros((3, 2))  # theta_Q: here one parameter per (s, a)
    Q_frozen = Q.copy()  # theta_Q^-: the frozen copy
    buffer = np.zeros((capacity, 4))  # the replay buffer D
    s = rng.choice(3, p=prior)
    for k in range(steps):
        # Act: mostly greedy, sometimes random; store the transition.
        a = rng.choice(2) if rng.random() < explore else int(Q[s].argmax())
        r, s_next = step(rng, s, a)
        buffer[k % capacity] = s, a, r, s_next  # overwrite the oldest
        s = restart(rng, s_next)
        # Learn: a minibatch of old transitions, targets from the frozen copy.
        chosen = buffer[rng.integers(min(k + 1, capacity), size=batch_size)]
        b_s, b_a = chosen[:, 0].astype(int), chosen[:, 1].astype(int)
        targets = chosen[:, 2] + gamma * Q_frozen[
            chosen[:, 3].astype(int)].max(axis=1)
        gradient = np.zeros((3, 2))
        np.add.at(gradient, (b_s, b_a), targets - Q[b_s, b_a])
        Q += step_size * gradient / batch_size
        if (k + 1) % refresh == 0:
            Q_frozen = Q.copy()
    return Q


Q_dqn = small_dqn(np.random.default_rng(3), 30_000)
print("small DQN =\n", Q_dqn)
print("largest error against Q*:", np.abs(Q_dqn - Q_star).max())
print("greedy moves:", Q_dqn.argmax(axis=1))
assert np.abs(Q_dqn - Q_star).max() < 0.5
assert list(Q_dqn.argmax(axis=1)) == [R, R, R]
