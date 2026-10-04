# %% [markdown]
# # Chapter 3 examples: infinite horizon and discounting
#
# This notebook runs the examples of
# [Chapter 3](https://thduynguyen.github.io/gtsam/chapter03) with the
# `gtsam/semiring` module: the endless track, a robot that keeps moving until
# its episode ends at random. The backward message, the forward message and
# the fixed point are all eliminations on semiring factors; numpy is used only
# to hold the tables and to check the results with a direct linear solve.

# %%
import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
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


unrolled_J = {}
for moves in [1, 2, 5, 10, 20, 50]:
    unrolled_J[moves] = chain(moves).expectation()
    print(f"{moves:3d} moves: graph.expectation() = {unrolled_J[moves]:.4f}")
assert np.allclose([unrolled_J[k] for k in [1, 2, 5, 10, 20, 50]],
                   [-0.5, -0.59, -0.4958, -0.1404, 0.2358, 0.4299], atol=1e-4)

# %% [markdown]
# ## Identical steps: the unrolled chain (Section 3)
#
# Every step of the chain has the same three factors, so one step is enough.
# First the two tables that summarize it: averaging the action out of the
# dynamics gives $P_\pi$, and averaging it out of the reward gives $r_\pi$.

# %%
now, move, later = state(0), action(0), state(1)
policy_factor = probability([now, move], ended_policy)
step = probability([now, move, later], ended_dynamics)
step_reward = value([now, move], ended_reward)

P_pi = table((policy_factor * step).sum(ordering(A(0))).probability(),
             [now, later])[:3, :3] / gamma  # the part that continues
r_pi = table((policy_factor * step_reward).sum(ordering(A(0))).value(),
             [now])[:3]
print("P_pi =\n", P_pi)
print("r_pi =", r_pi)
assert np.allclose(P_pi, [[0.6, 0.4, 0], [0.4, 0.2, 0.4], [0, 0.4, 0.6]])
assert np.allclose(r_pi, [-0.5, -0.5, 1.5])

# %% [markdown]
# One more move is two eliminations on those factors: the next state, then the
# action. The value of the new factor is fed back in as the value of the next
# state.


# %%
def backup(V):
    """One step of elimination, backward: the next state, then the action.

    V is the value of the next state, as an array over the four states.
    Returns the bucket of the action, whose value is Q, the conditional on the
    action, whose value is the advantage, and the new value as an array.
    """
    bucket = policy_factor * step_reward * (step * value([later], V)).sum(
        ordering(S(1)))
    conditional, new_factor = bucket.eliminate(ordering(A(0)))
    return bucket, conditional, table(new_factor.value(), [now])


unrolled = {}
V = np.zeros(4)  # no moves left: nothing to collect
for k in range(1, 101):
    _, _, V = backup(V)  # one more move
    unrolled[k] = V[:3]
for k in [1, 2, 5, 10, 20, 50, 100]:
    print(f"{k:3d} moves: V = {unrolled[k]},  J = {prior @ unrolled[k]:.4f}")
    if k in unrolled_J:  # the same number from the graph of k moves
        assert np.isclose(prior @ unrolled[k], unrolled_J[k])
assert np.allclose(unrolled[2], [-0.95, -0.23, 2.13])
assert np.allclose(unrolled[100], [-0.2248, 1.1017, 4.1230], atol=1e-4)

# %% [markdown]
# ## The fixed point (Section 4)
#
# With infinitely many moves the value that enters a step and the value that
# leaves it are the same function. Repeat the step until the value stops
# changing; the conditional of the last step holds the stationary advantage.

# %%
V = np.zeros(4)
sweeps = 0
while True:
    bucket, conditional, new_V = backup(V)
    sweeps += 1
    if np.abs(new_V - V).max() < 1e-13:
        break
    V = new_V
V = new_V[:3]
Q = table(bucket.value(), [now, move])[:3]
advantage = table(conditional.surprise(), [now, move])[:3]
J = prior @ V
print("sweeps until the value stops changing:", sweeps)
print("V =", V)
print("Q =\n", Q)
print("advantage =\n", advantage)
print("J =", J)
assert np.allclose(V, [-0.2248, 1.1017, 4.1231], atol=1e-4)
assert np.isclose(J, 0.4385, atol=1e-4)
assert np.allclose(Q, [[-0.2023, -0.2472], [0.0365, 2.1669],
                       [3.5354, 4.7108]], atol=1e-4)
assert np.allclose(advantage, [[0.022, -0.022], [-1.065, 1.065],
                               [-0.588, 0.588]], atol=1e-3)
assert np.allclose((policy * Q).sum(axis=1), V)  # V is the average of Q

# An independent check: the same V from a direct solve of the linear system.
direct = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
print("direct solve of (I - gamma P_pi) V = r_pi:", direct)
assert np.allclose(V, direct)

# The unrolled chain approaches it, within the bound gamma^k r_max / (1-gamma).
for k in [10, 20, 50, 100]:
    error = np.abs(unrolled[k] - V).max()
    bound = gamma ** k * np.abs(reward).max() / (1 - gamma)
    print(f"{k:3d} moves: error {error:.5f}, bound {bound:.5f}")
    assert error <= bound

# %% [markdown]
# ## Forward messages: the discounted visitation (Section 5)
#
# The forward message is the marginal of the state. In the chain with the
# state "ended", the marginal of a cell at step $t$ is the probability that
# the robot is in that cell *and* the episode is still running, which is
# $\gamma^t d_t(s)$. One forward step is one elimination: multiply the
# marginal of the current state with the policy and the dynamics, and sum out
# the current state and the action.


# %%
def forward(marginal):
    """One step of elimination, forward: the marginal of the next state."""
    joint = probability([now], marginal) * policy_factor * step
    return table(joint.sum(ordering(S(0), A(0))).probability(), [later])


# The first forward messages agree with the marginals of an unrolled graph.
bayes_tree = chain(3).eliminateMultifrontal()
marginal = ended_prior
for t in range(4):
    from_tree = table(bayes_tree.marginalFactor(S(t)).probability(), [state(t)])
    print(f"t = {t}: gamma^t d_t = {marginal[:3]}")
    assert np.allclose(marginal, from_tree)
    marginal = forward(marginal)
assert np.allclose(forward(ended_prior)[:3], gamma * np.array([0.5, 0.3, 0.2]))

# d(s): add the forward messages of all steps, until no probability of
# running is left. It is the expected number of steps spent in each cell.
d = np.zeros(3)
marginal = ended_prior
while marginal[:3].sum() > 1e-14:
    d += marginal[:3]
    marginal = forward(marginal)
print("d =", d, " sum =", d.sum())
print("d . r_pi =", d @ r_pi)
assert np.allclose(d, [3.8062, 3.4746, 2.7192], atol=1e-4)
assert np.isclose(d.sum(), 1 / (1 - gamma))
assert np.isclose(d @ r_pi, J)

# An independent check: the mirror-image linear system.
assert np.allclose(d, np.linalg.solve((np.eye(3) - gamma * P_pi).T, prior))
