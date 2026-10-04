# %% [markdown]
# # Chapter 3 examples: infinite horizon and discounting
#
# This notebook runs the examples of
# [Chapter 3](https://thduynguyen.github.io/gtsam/chapter03): the endless
# track, a robot that keeps moving until its episode ends at random.

# %%
import numpy as np
from gtsam import DecisionTreeFactor
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam.symbol_shorthand import A, S

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track (Section 1)
#
# The track and its slippery moves are those of Chapter 1. The robot now keeps
# moving. Moving Right costs 1, and every step spent in cell 2, at the charger,
# pays 2. After each move the episode continues with probability
# $\gamma = 0.9$.

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
policy = np.full((3, 2), 0.5)  # pi(a | s): a coin flip
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)

# %% [markdown]
# ## The unrolled chain (Section 3)
#
# The state-to-state table $P_\pi$ and the expected reward $r_\pi$ of the
# policy, then one Bellman backup per extra move.

# %%
P_pi = np.einsum("sa,sat->st", policy, dynamics)  # P_pi(s, s')
r_pi = (policy * reward).sum(axis=1)  # r_pi(s)
print("P_pi =\n", P_pi)
print("r_pi =", r_pi)

unrolled = {}
V = np.zeros(3)  # no moves left: nothing to collect
for k in range(1, 101):
    V = r_pi + gamma * P_pi @ V  # one more move
    unrolled[k] = V
for k in [1, 2, 5, 10, 20, 50, 100]:
    print(f"{k:3d} moves: V = {unrolled[k]},  J = {prior @ unrolled[k]:.4f}")

# %% [markdown]
# ## The fixed point (Section 4)
#
# With infinitely many moves the value satisfies $V = r_\pi + \gamma P_\pi V$,
# a linear system.

# %%
V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
Q = reward + gamma * dynamics @ V
J = prior @ V
print("V =", V)
print("Q =\n", Q)
print("J =", J)
assert np.allclose(V, [-0.2248, 1.1017, 4.1231], atol=1e-4)
assert np.isclose(J, 0.4385, atol=1e-4)
assert np.allclose((policy * Q).sum(axis=1), V)  # V is the average of Q

# The unrolled chain approaches it, within the bound gamma^k r_max / (1-gamma).
for k in [10, 20, 50, 100]:
    error = np.abs(unrolled[k] - V).max()
    bound = gamma ** k * np.abs(reward).max() / (1 - gamma)
    print(f"{k:3d} moves: error {error:.5f}, bound {bound:.5f}")
    assert error <= bound

# %% [markdown]
# ## Forward messages: the discounted visitation (Section 5)
#
# $d(s)$ is the expected number of steps the robot spends in cell $s$ before
# the episode ends. It gives the same $J$ from the other side.

# %%
d = np.linalg.solve((np.eye(3) - gamma * P_pi).T, prior)
print("d =", d, " sum =", d.sum())
print("d . r_pi =", d @ r_pi)
assert np.isclose(d.sum(), 1 / (1 - gamma))
assert np.isclose(d @ r_pi, J)

# %% [markdown]
# ## Discount as termination, on a factor graph (Section 2)
#
# Add a fourth state, "ended". After each move the robot goes there with
# probability $1 - \gamma$, and stays there, collecting nothing. No discount
# appears anywhere else: this is an ordinary semiring factor graph, and its
# expectation is the discounted return.

# %%
ENDED = 3
ended_dynamics = np.zeros((4, 2, 4))
ended_dynamics[:3, :, :3] = gamma * dynamics  # continue
ended_dynamics[:3, :, ENDED] = 1 - gamma  # end
ended_dynamics[ENDED, :, ENDED] = 1  # stay ended
ended_reward = np.vstack([reward, [0.0, 0.0]])
ended_policy = np.full((4, 2), 0.5)
ended_prior = np.append(prior, 0.0)

state = lambda t: (S(t), 4)  # (key, cardinality): cell 0, 1, 2 or ended
action = lambda t: (A(t), 2)  # 0 = Left, 1 = Right


def probability(keys, table):
    """Lift a probability table to (p, 0)."""
    return SemiringDiscreteFactor(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def value(keys, table):
    """Lift a reward table to (1, r)."""
    return SemiringDiscreteFactor.Reward(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def chain(moves):
    """The factor graph of the endless track, cut after some moves."""
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], ended_prior))
    for t in range(moves):
        graph.push_back(probability([state(t), action(t)], ended_policy))
        graph.push_back(probability(
            [state(t), action(t), state(t + 1)], ended_dynamics))
        graph.push_back(value([state(t), action(t)], ended_reward))
    return graph


for moves in [1, 2, 5, 10, 20]:
    expectation = chain(moves).expectation()
    print(f"{moves:3d} moves: graph.expectation() = {expectation:.4f}")
    assert np.isclose(expectation, prior @ unrolled[moves])
