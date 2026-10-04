# %% [markdown]
# # Chapter 4 examples: decision nodes and elimination order
#
# This notebook runs the examples of
# [Chapter 4](https://thduynguyen.github.io/gtsam/chapter04): the best policy
# of the track of Chapter 1 in one backward pass, then value iteration and
# policy iteration on the endless track of Chapter 3.

# %%
import itertools

import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import SemiringDiscreteFactor
from gtsam.symbol_shorthand import A, S

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track of Chapter 1, without a policy

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
move_reward = np.array([[0.0, -1.0]] * 3)  # r(s, a)
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)

# %% [markdown]
# ## One backward pass: average over the next state, maximize over the action (Sections 2 and 3)
#
# First in numpy, where each line is one elimination.

# %%
V = final_reward  # V*_2
best = {}
for t in [1, 0]:
    Q = move_reward + dynamics @ V  # eliminate s_{t+1} by average
    best[t] = Q.argmax(axis=1)  # the conditional on a_t: the best move
    V = Q.max(axis=1)  # eliminate a_t by max
    print(f"Q*_{t} =\n{Q}\nV*_{t} = {V}, best move = {best[t]} (0 = L, 1 = R)")
J_star = prior @ V  # eliminate s_0 by average
print("J* =", J_star)
assert np.isclose(J_star, 6.1)

# %% [markdown]
# The same pass with the module. There is no built-in maximum, so the best
# move of each cell is read from the table of $Q$ and put back as a policy
# that always takes it. Summing out that policy substitutes the best move,
# which is the maximum.

# %%
state = lambda t: (S(t), 3)  # (key, cardinality)
action = lambda t: (A(t), 2)


def probability(keys, table):
    """Lift a probability table to (p, 0)."""
    return SemiringDiscreteFactor(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def value(keys, table):
    """Lift a reward table to (1, r)."""
    return SemiringDiscreteFactor.Reward(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def ordering(key):
    """An ordering holding one key."""
    result = Ordering()
    result.push_back(key)
    return result


def table(factor, keys):
    """Read a DecisionTreeFactor into an array indexed in the order of keys."""
    result = np.zeros([cardinality for _, cardinality in keys])
    for index in np.ndindex(*result.shape):
        values = DiscreteValues()
        for (key, _), index_of_key in zip(keys, index):
            values[key] = index_of_key
        result[index] = factor(values)
    return result


future = value([state(2)], final_reward)  # (1, V*_2)
for t in [1, 0]:
    keys = [state(t), action(t)]
    # Eliminate the next state by average. There is no policy factor.
    step = probability(keys + [state(t + 1)], dynamics)
    bucket = value(keys, move_reward) * (step * future).sum(ordering(S(t + 1)))
    Q = table(bucket.value(), keys)
    # Eliminate the action by max: the best move, as a one-hot policy factor.
    greedy = probability(keys, np.eye(2)[Q.argmax(axis=1)])
    future = (greedy * bucket).sum(ordering(A(t)))  # (1, V*_t)
    print(f"V*_{t} =", table(future.value(), [state(t)]))
J_star = (probability([state(0)], prior) * future).expectation()
print("J* =", J_star)
assert np.isclose(J_star, 6.1)

# %% [markdown]
# ## The order matters (Section 2)
#
# At the last move, compare the correct order, average over the next state and
# then maximize over the action, with the maximum over both, which assumes the
# robot can also choose how the move turns out.

# %%
outcome = move_reward[:, :, None] + final_reward[None, None, :]  # r + r(s2)
possible = np.where(dynamics > 0, outcome, -np.inf)
max_then_average = (move_reward + dynamics @ final_reward).max(axis=1)
max_of_both = possible.max(axis=(1, 2))
print("average over luck, max over action:", max_then_average)
print("max over luck and action:          ", max_of_both)
assert np.all(max_of_both >= max_then_average)

# %% [markdown]
# ## A shared decision does not decompose (Section 4)
#
# If the same table must be used at both moves, a *stationary* policy, the
# best move of a cell can no longer be chosen separately for each move. All
# eight deterministic stationary policies:


# %%
def evaluate(policies):
    """Expected return of a policy given as one table pi(a | s) per move."""
    V = final_reward
    for policy in reversed(policies):
        V = (policy * (move_reward + dynamics @ V)).sum(axis=1)
    return prior @ V


results = {}
for moves in itertools.product([L, R], repeat=3):
    policy = np.eye(2)[list(moves)]
    results[moves] = evaluate([policy, policy])
    name = "".join("LR"[m] for m in moves)
    print(f"cells 0,1,2 -> {name}:  J = {results[moves]:.3f}")
print("best stationary policy:", max(results.values()))
time_varying = evaluate([np.eye(2)[best[0]], np.eye(2)[best[1]]])
print("best policy with a table per move:", time_varying)
assert np.isclose(time_varying, 6.1)
assert max(results.values()) < 6.1

# %% [markdown]
# ## The endless track: value iteration (Section 5)

# %%
gamma = 0.9
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)

V = np.zeros(3)
for k in range(1, 201):
    Q = reward + gamma * dynamics @ V  # eliminate s' by average
    V = Q.max(axis=1)  # eliminate a by max
    if k in [1, 2, 5, 10, 50, 100, 200]:
        print(f"{k:3d} moves: V = {V}, greedy = {Q.argmax(axis=1)}")
V_star, Q_star = V, Q
print("Q* =\n", Q_star)
print("J* =", prior @ V_star)
assert np.allclose(V_star, [5.4194, 7.5610, 10.0], atol=1e-3)
assert np.isclose(prior @ V_star, 6.4902, atol=1e-3)

# %% [markdown]
# ## The endless track: policy iteration (Section 5)
#
# Stage 1 evaluates the current policy exactly, by solving the linear system
# of Chapter 3. Stage 2 takes, in each cell, the move with the largest
# advantage.


# %%
def stage1(policy):
    """Evaluate a stationary policy: V, Q and the advantage."""
    P_pi = np.einsum("sa,sat->st", policy, dynamics)
    r_pi = (policy * reward).sum(axis=1)
    V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
    Q = reward + gamma * dynamics @ V
    return V, Q, Q - V[:, None]


policy = np.full((3, 2), 0.5)  # start from the coin flip
for iteration in range(4):
    V, Q, advantage = stage1(policy)
    greedy = advantage.argmax(axis=1)
    print(f"iteration {iteration}: V = {V}, J = {prior @ V:.4f}, "
          f"greedy = {greedy}")
    policy = np.eye(2)[greedy]  # stage 2
assert np.allclose(V, V_star, atol=1e-3)
