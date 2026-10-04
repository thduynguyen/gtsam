# %% [markdown]
# # Chapter 22 examples: partial observability
#
# This notebook runs the examples of
# [Chapter 22](https://thduynguyen.github.io/gtsam/chapter22) with the
# `gtsam/semiring` module: a robot on the track that cannot see its cell and
# only has a noisy sensor. The readings are variables of the factor graph.
# The belief is a conditional that forward elimination leaves, and the best
# policy that uses only the sensor comes from one elimination in the order
# that Chapter 4's rule forces. The same is done for the line.

# %%
import itertools

import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, HessianFactor
from gtsam import JacobianFactor, Ordering, noiseModel
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam import SemiringGaussianFactor, SemiringRules, SemiringSum
from gtsam.symbol_shorthand import A, S, U, X, Y

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track and a noisy cell sensor (Section 1)
#
# The dynamics are those of Chapter 1. The sensor reports the true cell with
# probability 0.7 and a neighbouring cell otherwise.

# %%
L, R = 0, 1
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
sensor = np.array([[0.7, 0.3, 0.0],  # p(y | s): rows s, columns y
                   [0.15, 0.7, 0.15],
                   [0.0, 0.3, 0.7]])
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)

# The track of Chapter 1, and the "lost robot" variant of this chapter.
track = dict(prior=np.array([0.5, 0.5, 0.0]),
             move_reward=np.array([[0.0, -1.0]] * 3))
lost = dict(prior=np.array([0.5, 0.0, 0.5]),
            move_reward=np.array([[0.0, -3.0]] * 3))

state = lambda t: (S(t), 3)  # (key, cardinality)
action = lambda t: (A(t), 2)
reading = lambda t: (Y(t), 3)


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


def robot(prior, move_reward, sensors=True, first_policy=None):
    """The factor graph of the two-move problem, without policy factors.

    With `sensors`, each state has a reading y_t joined to it by a sensor
    factor. `first_policy` is an optional table pi(a0 | y0).
    """
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    for t in range(2):
        if sensors:
            graph.push_back(probability([state(t), reading(t)], sensor))
        keys = [state(t), action(t)]
        graph.push_back(probability(keys + [state(t + 1)], dynamics))
        graph.push_back(value(keys, move_reward))
    graph.push_back(value([state(2)], final_reward))
    if first_policy is not None:
        graph.push_back(probability([reading(0), action(0)], first_policy))
    return graph


# %% [markdown]
# ## Three levels of knowledge, three elimination orders (Sections 1, 3, 4)
#
# The actions are eliminated by maximum and everything else by average. What
# the robot knows when it acts decides the order (Chapter 4, Section 2: a
# variable is eliminated before every decision made without knowing it).
#
# - Fully observed: each state after the action taken in it.
# - Blind: all states before any action.
# - With the sensor: all states first, then each action before the reading it
#   is chosen with.

# %%
maximum = SemiringSum.Maximum()
both_actions = SemiringRules()
both_actions.setAll([A(0), A(1)], maximum)

seen = ordering(S(2), A(1), S(1), A(0), S(0))
unseen = ordering(S(2), S(1), S(0), A(1), A(0))
sensed = ordering(S(2), S(1), S(0), A(1), Y(1), A(0), Y(0))


def fully_observed(prior, move_reward):
    return robot(prior, move_reward, sensors=False).expectation(
        seen, both_actions)


def blind(prior, move_reward):
    """The best pair of moves chosen without any observation, and its J."""
    graph = robot(prior, move_reward, sensors=False)
    bayes_net = graph.eliminateSequential(unseen, both_actions)
    first = table(bayes_net.at(4).greedy(), [action(0)]).argmax()
    last = table(bayes_net.at(3).greedy(), [action(0), action(1)])[first]
    return graph.expectation(unseen, both_actions), (int(first),
                                                     int(last.argmax()))


def with_sensor(prior, move_reward):
    return robot(prior, move_reward).expectation(sensed, both_actions)


for name, problem in [("track", track), ("lost robot", lost)]:
    J_blind, moves_blind = blind(**problem)
    print(f"{name:10s}: blind {J_blind:.4f} {moves_blind}, "
          f"sensor {with_sensor(**problem):.4f}, "
          f"fully observed {fully_observed(**problem):.4f}")

assert np.isclose(fully_observed(**track), 6.1)
assert np.isclose(blind(**track)[0], 6.0)
assert np.isclose(with_sensor(**track), 6.0)
assert np.isclose(fully_observed(**lost), 3.2)
assert np.isclose(blind(**lost)[0], 2.2)
assert np.isclose(with_sensor(**lost), 2.69)

# %% [markdown]
# ## The belief: a forward message given the readings (Section 2)
#
# Eliminating a hidden state forward in time leaves its conditional given the
# readings: the belief, for every reading at once. For the lost robot, the
# belief after the first reading:

# %%
prior = lost["prior"]
graph = SemiringFactorGraph()
graph.push_back(probability([state(0)], prior))
graph.push_back(probability([state(0), reading(0)], sensor))
bayes_net = graph.eliminateSequential(ordering(S(0), Y(0)))
beliefs0 = table(bayes_net.at(0).probability(), [reading(0), state(0)])
p_y0 = table(bayes_net.at(1).probability(), [reading(0)])
for y0 in range(3):
    print(f"y0 = {y0}: probability {p_y0[y0]:.3f}, belief {beliefs0[y0]}")
assert np.allclose(p_y0, [0.35, 0.30, 0.35])
assert np.allclose(beliefs0, [[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]])

# %% [markdown]
# Then the first reading says "cell 1" and the robot moves Right. The reading
# and the action are fixed, so the sensor factor is a table on $s_0$ alone and
# the dynamics factor a table on $(s_0, s_1)$. Eliminating $s_0$ predicts, and
# the second sensor factor updates.

# %%
graph = SemiringFactorGraph()
graph.push_back(probability([state(0)], prior))
graph.push_back(probability([state(0)], sensor[:, 1]))  # y0 = 1
graph.push_back(probability([state(0), state(1)], dynamics[:, R, :]))
predicted = table(
    graph.eliminateSequential(ordering(S(0), S(1))).at(1).probability(),
    [state(1)])
print("predicted after moving Right:", predicted)
assert np.allclose(predicted, [0.1, 0.4, 0.5])

graph.push_back(probability([state(1), reading(1)], sensor))
bayes_net = graph.eliminateSequential(ordering(S(0), S(1), Y(1)))
beliefs1 = table(bayes_net.at(1).probability(), [reading(1), state(1)])
p_y1 = table(bayes_net.at(2).probability(), [reading(1)])
for y1 in range(3):
    print(f"  y1 = {y1}: probability {p_y1[y1]:.4f}, belief {beliefs1[y1]}")
for y1 in range(3):  # a check: multiply by the sensor column and normalize
    joint = predicted * sensor[:, y1]
    assert np.isclose(p_y1[y1], joint.sum())
    assert np.allclose(beliefs1[y1], joint / joint.sum())

# %% [markdown]
# With every reading fixed, the belief is also the marginal of the Bayes tree,
# as in any GTSAM graph.


# %%
def module_belief(prior, y0, a0, y1):
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    graph.push_back(probability([state(0)], sensor[:, y0]))
    graph.push_back(probability([state(0), state(1)], dynamics[:, a0, :]))
    graph.push_back(probability([state(1)], sensor[:, y1]))
    marginal = graph.eliminateMultifrontal().marginalFactor(S(1))
    belief = table(marginal.probability(), [state(1)])
    return belief / belief.sum()


for y1 in range(3):
    assert np.allclose(module_belief(prior, 1, R, y1), beliefs1[y1])

# %% [markdown]
# ## The best policy that sees only the sensor (Sections 3 and 4)
#
# One elimination in the order states, last action, last reading, first
# action, first reading. The conditional of each action is the policy, as a
# table over the readings and actions that came before it.

# %%
graph = robot(**lost)
bayes_net = graph.eliminateSequential(sensed, both_actions)
first_policy = table(bayes_net.at(5).greedy(), [reading(0), action(0)])
last_policy = table(bayes_net.at(3).greedy(),
                    [reading(0), action(0), reading(1), action(1)])

# The value of each first move for each first reading: what is left on
# (y0, a0) once the states, the last action and the last reading are gone.
_, remaining = graph.eliminatePartialSequential(
    ordering(S(2), S(1), S(0), A(1), Y(1)), both_actions)
first_values = table(remaining.product().value(), [reading(0), action(0)])
for y0 in range(3):
    print(f"y0 = {y0}: belief {beliefs0[y0]}, value of L, R = "
          f"{first_values[y0]}, best first move "
          f"{'LR'[first_policy[y0].argmax()]}")
print("best last move for (y0, a0, y1), 0 = L, 1 = R:\n",
      last_policy.argmax(axis=-1))
assert np.allclose(first_values, [[0, 0.4], [1.95, 2.2], [5.4, 4.0]])
assert list(first_policy.argmax(axis=1)) == [R, R, L]
J_sensor = with_sensor(**lost)
assert np.isclose(p_y0 @ first_values.max(axis=1), J_sensor)

# %% [markdown]
# The action values of the fully observed problem, by the backward pass of
# Chapter 4, one elimination at a time. At the last move they are also the
# right ones for the hidden-state problem, averaged over the belief.


# %%
def action_values(move_reward):
    """Q*_0 and Q*_1 of the fully observed problem, as tables."""
    future = value([state(2)], final_reward)
    Q = {}
    for t in [1, 0]:
        keys = [state(t), action(t)]
        step = probability(keys + [state(t + 1)], dynamics)
        bucket = value(keys, move_reward) * (step * future).sum(
            ordering(S(t + 1)))
        Q[t] = table(bucket.value(), keys)
        future = bucket.sum(ordering(A(t)), maximum)
    return Q[0], Q[1]


Q0, Q1 = action_values(lost["move_reward"])
print("Q*_1 =\n", Q1)
print("Q*_0 =\n", Q0)
# Right is the better last move when the belief b satisfies b @ Q1[:, R] >
# b @ Q1[:, L].
print("difference Q*_1(s, R) - Q*_1(s, L):", Q1[:, R] - Q1[:, L])
assert np.allclose(Q1, [[0, -3], [0, 5], [2, 7]])
# The last move kept by the elimination is the best one for the belief.
for y1 in range(3):
    assert last_policy[1, R, y1].argmax() == (beliefs1[y1] @ Q1).argmax()

# %% [markdown]
# A check by brute force, in plain numpy: all deterministic policies that map
# the readings to moves. The first move depends on $y_0$ (8 choices), the last
# move on $(y_0, y_1)$ (512 choices).


# %%
def evaluate(prior, move_reward, first, last):
    """Expected return of a policy given as tables first[y0], last[y0][y1]."""
    J = 0.0
    for s0, y0 in itertools.product(range(3), repeat=2):
        p0 = prior[s0] * sensor[s0, y0]
        if p0 == 0:
            continue
        a0 = first[y0]
        for s1, y1 in itertools.product(range(3), repeat=2):
            p1 = p0 * dynamics[s0, a0, s1] * sensor[s1, y1]
            if p1 == 0:
                continue
            a1 = last[y0][y1]
            J += p1 * (move_reward[s0, a0] + move_reward[s1, a1] +
                       dynamics[s1, a1] @ final_reward)
    return J


best = -np.inf
for first in itertools.product([L, R], repeat=3):
    for flat in itertools.product([L, R], repeat=9):
        last = np.reshape(flat, (3, 3))
        best = max(best, evaluate(**lost, first=first, last=last))
print("best of 4096 policies:", best)
assert np.isclose(best, J_sensor)

# %% [markdown]
# ## QMDP and where it errs (Section 5)
#
# QMDP weights the fully observed action values by the belief. As an
# elimination: put $Q^*_0$ on $(s_0, a_0)$ as a value factor, sum out $s_0$
# given the reading, and take the maximum over $a_0$. That assumes the cell
# will be known exactly after this move.

# %%
qmdp_bucket = (probability([state(0)], prior) *
               probability([state(0), reading(0)], sensor) *
               value([state(0), action(0)], Q0)).sum(ordering(S(0)))
qmdp_values = table(qmdp_bucket.value(), [reading(0), action(0)])
conditional, _ = qmdp_bucket.eliminate(ordering(A(0)), maximum)
qmdp_policy = table(conditional.greedy(), [reading(0), action(0)])
for y0 in range(3):
    print(f"y0 = {y0}: QMDP values of L, R = {qmdp_values[y0]}, true values "
          f"= {first_values[y0]}, QMDP move "
          f"{'LR'[qmdp_policy[y0].argmax()]}, best move "
          f"{'LR'[first_policy[y0].argmax()]}")
assert np.allclose(qmdp_values, [[0, 1], [2.7, 2.5], [5.4, 4.0]])
assert list(qmdp_policy.argmax(axis=1)) == [R, L, L]

# The return of QMDP: its first move as a policy factor on (y0, a0), and the
# last move still the best one for the belief.
last_action = SemiringRules()
last_action.set(A(1), maximum)
J_qmdp = robot(**lost, first_policy=qmdp_policy).expectation(
    sensed, last_action)
J_again = robot(**lost, first_policy=first_policy).expectation(
    sensed, last_action)
print("QMDP:", J_qmdp, " best:", J_sensor)
assert np.isclose(J_qmdp, 2.615) and np.isclose(J_again, J_sensor)
assert J_qmdp < J_sensor

# %% [markdown]
# ## The value of the sensor on the original track (Section 1)
#
# On the track of Chapter 1 the sensor changes nothing: no reading makes the
# belief in cell 0 large enough to prefer Left at the last move.

# %%
_, Q1_track = action_values(track["move_reward"])
gap = Q1_track[:, R] - Q1_track[:, L]
print("Q*_1(s, R) - Q*_1(s, L) on the track:", gap)
# Left is better only if b0 * 1 > b1 * 7 + b2 * 7, i.e. b0 > 7/8.
# The belief after a move Right, for every pair of readings at once: the
# conditional of s1 given (y0, y1).
graph = SemiringFactorGraph()
graph.push_back(probability([state(0)], track["prior"]))
graph.push_back(probability([state(0), reading(0)], sensor))
graph.push_back(probability([state(0), state(1)], dynamics[:, R, :]))
graph.push_back(probability([state(1), reading(1)], sensor))
bayes_net = graph.eliminateSequential(ordering(S(0), S(1), Y(1), Y(0)))
beliefs = table(bayes_net.at(1).probability(),
                [reading(0), reading(1), state(1)])
largest = beliefs[:, :, 0].max()
print("largest belief in cell 0 after a move Right:", largest)
assert largest < 7 / 8

# %% [markdown]
# ## The separation principle on the line (Section 6)
#
# The line of Chapter 1 with a noisy position sensor,
# $y_t = x_t + n_t$, $n_t \sim N(0, 0.5)$. The same order as for the track:
# the states by average, then each action by maximum before the reading it is
# chosen with.

# %%
I, zero = np.eye(1), np.zeros(1)
sigma_w, sigma_y = 0.5, 0.5


def variance(v):
    """A scalar Gaussian noise model with the given variance."""
    return noiseModel.Isotropic.Variance(1, v)


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def line(policies=()):
    """The line with a sensor, and optional policy factors."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([2.0]), variance(1.0)))
    for t in range(2):
        graph.push_back(gaussian(Y(t), I, X(t), -I, zero, variance(sigma_y)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 variance(sigma_w)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    for policy in policies:
        graph.push_back(policy)
    return graph


controls = SemiringRules()
controls.setAll([U(0), U(1)], maximum)

# The states, the last action and the last reading, by their rules.
bayes_net, remaining = line().eliminatePartialSequential(
    ordering(X(2), X(1), X(0), U(1), Y(1)), controls)
last_law = bayes_net.at(3).conditional()  # u1 given (u0, y0, y1)
print("the last action: u1 +", last_law.S().ravel(), "(u0, y0, y1) =",
      last_law.d())

# The Kalman filter in closed form, to compare: x_hat_1 = (1 - L1)(x_hat_0 +
# u0) + L1 y1, with x_hat_0 = (1 - L0) 2 + L0 y0.
L0, L1 = 2 / 3, 0.625
K0, K1 = 0.6, 0.5  # the Riccati gains of the fully observed problem
assert np.allclose(last_law.S().ravel(),
                   [K1 * (1 - L1), K1 * (1 - L1) * L0, K1 * L1])
assert np.isclose(last_law.d()[0], -K1 * (1 - L1) * (1 - L0) * 2.0)

# %% [markdown]
# The elimination left the law $u_1 = -K_1 \hat x_1$: the Riccati gain applied
# to the Kalman estimate, without either being computed separately.
#
# For the first action the module needs a hand. What is left holds a Gaussian
# factor, $p(y_0)$, that still lists $u_0$ among its keys, with zero
# information, and the maximum rule refuses a variable that a Gaussian factor
# mentions. So the maximum over $u_0$ is taken on the value channel alone, and
# $p(y_0)$ is multiplied back in for the last average.

# %%
left = remaining.product()
value_of_first = SemiringGaussianFactor.Reward(left.value())
first_law, value_of_reading = value_of_first.eliminate(ordering(U(0)), maximum)
first_conditional = first_law.conditional()  # u0 given y0
print("the first action: u0 +", first_conditional.S().ravel(), "y0 =",
      first_conditional.d())
assert np.isclose(first_conditional.S()[0, 0], K0 * L0)
assert np.isclose(first_conditional.d()[0], -K0 * (1 - L0) * 2.0)

# p(y0): eliminate x0 from the prior and the first sensor factor.
first_reading = (gaussian(X(0), I, np.array([2.0]), variance(1.0)).multiply(
    gaussian(Y(0), I, X(0), -I, zero, variance(sigma_y)))).sum(ordering(X(0)))
J_lqg = first_reading.multiply(value_of_reading).expectation()
print("J with the sensor, by elimination:", J_lqg)
assert np.isclose(J_lqg, -9.70625)

# %% [markdown]
# The same number by evaluating the two laws as policy factors: the
# conditionals that the elimination returned, put back into the graph as hard
# linear constraints, with the average everywhere. And the formula of the
# chapter: the fully observed optimum minus a price for the variance of the
# belief.

# %%
laws = [SemiringGaussianFactor(first_conditional),
        SemiringGaussianFactor(last_law)]
J_closed_loop = line(laws).expectation(
    ordering(X(2), U(1), Y(1), X(1), U(0), Y(0), X(0)))
print("J of the closed loop:", J_closed_loop)
assert np.isclose(J_closed_loop, J_lqg)

gains = [K0, K1]
H_uu = [1 + 1.5, 1 + 1.0]  # C_u + B' P_{t+1} B
J_full = -9.25
# Variance of the belief after each reading (the Kalman filter).
variances = []
predicted_variance = 1.0  # the prior variance of x0
for t in range(2):
    posterior = predicted_variance * sigma_y / (predicted_variance + sigma_y)
    variances.append(posterior)
    predicted_variance = posterior + sigma_w
print("variance of the belief after y0 and y1:", variances)
price = sum(H_uu[t] * gains[t] ** 2 * variances[t] for t in range(2))
print("price of not seeing the state:", price, " J =", J_full - price)
assert np.allclose(variances, [1 / 3, 0.3125])
assert np.isclose(J_full - price, J_lqg)

# A check by simulation, with a fixed seed.
rng = np.random.default_rng(0)
samples = 400_000
x = 2 + rng.normal(size=samples)
mean, var, total = np.full(samples, 2.0), 1.0, np.zeros(samples)
for t in range(2):
    y = x + np.sqrt(sigma_y) * rng.normal(size=samples)
    kalman = var / (var + sigma_y)
    mean, var = mean + kalman * (y - mean), (1 - kalman) * var
    u = -gains[t] * mean
    total -= x ** 2 + u ** 2
    x = x + u + np.sqrt(sigma_w) * rng.normal(size=samples)
    mean, var = mean + u, var + sigma_w
total -= x ** 2
print("simulated J:", total.mean(), "+-", total.std() / np.sqrt(samples))
assert abs(total.mean() - J_lqg) < 0.05
