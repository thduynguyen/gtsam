# %% [markdown]
# # Chapter 4 examples: decision nodes and elimination order
#
# This notebook runs the examples of
# [Chapter 4](https://thduynguyen.github.io/gtsam/chapter04) with the
# `gtsam/semiring` module: the best policy of the track of Chapter 1 in one
# backward pass, then value iteration and policy iteration on the endless
# track of Chapter 3. The states are summed out by the average and the actions
# by the maximum.

# %%
import itertools

import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam import SemiringRules, SemiringSum
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


def ordering(*keys):
    """An ordering of the given keys."""
    result = Ordering()
    for key in keys:
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


def track(policies=None):
    """The factor graph of the track, with or without policy factors."""
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    for t in range(2):
        keys = [state(t), action(t)]
        if policies is not None:
            graph.push_back(probability(keys, policies[t]))
        graph.push_back(probability(keys + [state(t + 1)], dynamics))
        graph.push_back(value(keys, move_reward))
    graph.push_back(value([state(2)], final_reward))
    return graph


backward = ordering(S(2), A(1), S(1), A(0), S(0))

# %% [markdown]
# ## One backward pass: average over the next state, maximize over the action (Sections 2 and 3)
#
# The actions get the maximum rule and the states keep the default, the
# average. The elimination order is backward in time.

# %%
rules = SemiringRules()
rules.setAll([A(0), A(1)], SemiringSum.Maximum())

graph = track()
J_star = graph.expectation(backward, rules)
print("J* =", J_star)
assert np.isclose(J_star, 6.1)

# The Bayes net holds, for each action, the best move and the regrets.
bayes_net = graph.eliminateSequential(backward, rules)
best = {}
for t, position in [(1, 1), (0, 3)]:
    keys = [state(t), action(t)]
    conditional = bayes_net.at(position)
    best[t] = table(conditional.greedy(), keys).argmax(axis=1)
    print(f"best move at step {t}: {best[t]} (0 = L, 1 = R)")
    print(f"regret Q*_{t} - V*_{t}:\n{table(conditional.surprise(), keys)}")
assert list(best[1]) == [L, R, R] and list(best[0]) == [R, R, R]

# %% [markdown]
# The same pass one elimination at a time, to read the tables $Q^*_t$ and
# $V^*_t$ of the chapter.

# %%
maximum = SemiringSum.Maximum()
future = value([state(2)], final_reward)  # (1, V*_2)
expected_Q = {1: [[0, -1], [0, 7], [2, 9]], 0: [[0, 4.6], [1.4, 7.6], [7.4, 8]]}
expected_V = {1: [0, 7, 9], 0: [4.6, 7.6, 8]}
for t in [1, 0]:
    keys = [state(t), action(t)]
    # Eliminate the next state by average. There is no policy factor.
    step = probability(keys + [state(t + 1)], dynamics)
    bucket = value(keys, move_reward) * (step * future).sum(ordering(S(t + 1)))
    Q = table(bucket.value(), keys)
    # Eliminate the action by maximum.
    conditional, future = bucket.eliminate(ordering(A(t)), maximum)
    V = table(future.value(), [state(t)])
    print(f"Q*_{t} =\n{Q}\nV*_{t} = {V}")
    assert np.allclose(Q, expected_Q[t]) and np.allclose(V, expected_V[t])
J_star = (probability([state(0)], prior) * future).expectation()
assert np.isclose(J_star, 6.1)

# %% [markdown]
# ## The order matters (Section 2)
#
# At the last move, compare the correct treatment, the average over the next
# state and the maximum over the action, with the maximum over both, which
# assumes the robot can also choose how the move turns out.

# %%
keys = [state(1), action(1)]
bucket_s2 = probability(keys + [state(2)], dynamics) * value(
    [state(2)], final_reward)
results = {}
for name, rule in [("average", SemiringSum.Average()), ("maximum", maximum)]:
    after_s2 = value(keys, move_reward) * bucket_s2.sum(ordering(S(2)), rule)
    results[name] = table(
        after_s2.sum(ordering(A(1)), maximum).value(), [state(1)])
print("average over luck, max over action:", results["average"])
print("max over luck and action:          ", results["maximum"])
assert np.allclose(results["average"], [0, 7, 9])
assert np.allclose(results["maximum"], [0, 9, 10])

# %% [markdown]
# ## A shared decision does not decompose (Section 4)
#
# If the same table must be used at both moves, a *stationary* policy, the
# best move of a cell can no longer be chosen separately for each move. All
# eight deterministic stationary policies, each evaluated by one elimination:

# %%
results = {}
for moves in itertools.product([L, R], repeat=3):
    policy = np.eye(2)[list(moves)]
    results[moves] = track([policy, policy]).expectation()
    name = "".join("LR"[m] for m in moves)
    print(f"cells 0,1,2 -> {name}:  J = {results[moves]:.3f}")
print("best stationary policy:", max(results.values()))
time_varying = track([np.eye(2)[best[0]], np.eye(2)[best[1]]]).expectation()
print("best policy with a table per move:", time_varying)
assert np.isclose(time_varying, 6.1)
assert np.isclose(max(results.values()), 6.0)

# %% [markdown]
# ## The endless track (Section 5)
#
# The discount is a termination outcome, as in Chapter 3: a fourth state,
# "ended", reached with probability $1 - \gamma$ after every move.

# %%
gamma = 0.9
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)

ENDED = 3
ended_dynamics = np.zeros((4, 2, 4))
ended_dynamics[:3, :, :3] = gamma * dynamics  # continue
ended_dynamics[:3, :, ENDED] = 1 - gamma  # end
ended_dynamics[ENDED, :, ENDED] = 1  # stay ended
ended_reward = np.vstack([reward, [0.0, 0.0]])
ended_prior = np.append(prior, 0.0)

now, move, later = (S(0), 4), (A(0), 2), (S(1), 4)
step = probability([now, move, later], ended_dynamics)
step_reward = value([now, move], ended_reward)


def backup(V, rule, policy=None):
    """One step of elimination: the next state by average, the action by `rule`.

    V is the value of the next state, as an array over the four states.
    Returns the conditional on the action and the new value, as an array.
    """
    bucket = step_reward * (step * value([later], V)).sum(ordering(S(1)))
    if policy is not None:
        bucket = probability([now, move], policy) * bucket
    conditional, new_factor = bucket.eliminate(ordering(A(0)), rule)
    return conditional, table(new_factor.value(), [now])


# %% [markdown]
# ### Value iteration: the one-pass method, unrolled

# %%
V = np.zeros(4)
for k in range(1, 201):
    conditional, V = backup(V, maximum)
    if k in [1, 2, 5, 10, 50, 100, 200]:
        greedy = table(conditional.greedy(), [now, move]).argmax(axis=1)[:3]
        print(f"{k:3d} moves: V = {V[:3]}, greedy = {greedy}")
V_star = V
print("J* =", ended_prior @ V_star)
assert np.allclose(V_star[:3], [5.4194, 7.5610, 10.0], atol=1e-3)
assert np.isclose(ended_prior @ V_star, 6.4902, atol=1e-3)

# The same number from one graph of 60 moves, eliminated with the rules.
moves = 60
graph = SemiringFactorGraph()
graph.push_back(probability([(S(0), 4)], ended_prior))
keys_backward = []
rules = SemiringRules()
for t in range(moves):
    keys = [(S(t), 4), (A(t), 2)]
    graph.push_back(probability(keys + [(S(t + 1), 4)], ended_dynamics))
    graph.push_back(value(keys, ended_reward))
    rules.set(A(t), maximum)
    keys_backward = [S(t + 1), A(t)] + keys_backward
unrolled = graph.expectation(ordering(*keys_backward, S(0)), rules)
print(f"one graph of {moves} moves: {unrolled:.4f}")
assert abs(unrolled - 6.4902) < 0.02

# %% [markdown]
# ### Policy iteration: two stages, with a greedy outer step
#
# Stage 1 evaluates the current policy by eliminating with the average rule
# until the value stops changing. Stage 2 is one more elimination of the
# action, by maximum and without the policy factor: its conditional holds the
# best move in each cell for the values of Stage 1.


# %%
def stage1(policy):
    """Evaluate a stationary policy: its value, as an array."""
    V = np.zeros(4)
    while True:
        _, new_V = backup(V, SemiringSum.Average(), policy)
        if np.abs(new_V - V).max() < 1e-12:
            return new_V
        V = new_V


def stage2(V):
    """The greedy policy for the values V, as a table."""
    conditional, _ = backup(V, maximum)
    return table(conditional.greedy(), [now, move])


policy = np.full((4, 2), 0.5)  # start from the coin flip
expected_J = [0.4385, 3.7805, 6.4902, 6.4902]
for iteration in range(4):
    V = stage1(policy)
    greedy = stage2(V)
    print(f"iteration {iteration}: V = {V[:3]}, J = {ended_prior @ V:.4f}, "
          f"greedy = {greedy.argmax(axis=1)[:3]}")
    assert np.isclose(ended_prior @ V, expected_J[iteration], atol=1e-4)
    policy = greedy
assert np.allclose(V[:3], V_star[:3], atol=1e-3)

# The advantages of the coin flip, from the conditional of its evaluation.
coin_flip = np.full((4, 2), 0.5)
conditional, _ = backup(stage1(coin_flip), SemiringSum.Average(), coin_flip)
advantage = table(conditional.surprise(), [now, move])[:3]
print("advantage of the coin flip:\n", advantage)
assert np.allclose(advantage, [[0.022, -0.022], [-1.065, 1.065],
                               [-0.588, 0.588]], atol=1e-3)
