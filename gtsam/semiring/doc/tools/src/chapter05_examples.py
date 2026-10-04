# %% [markdown]
# # Chapter 5 examples: gradients by elimination
#
# This notebook runs the examples of
# [Chapter 5](https://thduynguyen.github.io/gtsam/chapter05) with the
# `gtsam/semiring` module: the gradient of the expected return with respect to
# the parameters of a policy, computed in three ways that must agree, and the
# two-stage loop that uses it. The second-order semiring of Section 4 is the
# one computation the module does not implement; it runs in numpy.

# %%
import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam.symbol_shorthand import A, S

np.set_printoptions(precision=4, suppress=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


# %% [markdown]
# ## One decision (Section 2)
#
# The robot picks Left with probability $\sigma(\theta)$. Left costs 1, the
# outcome is good with probability 0.9 after Left and 0.2 after Right, and a
# good outcome pays 10. Elimination gives $J$, and the conditional of the
# action holds the policy and the advantage.

# %%
ACTION, OUTCOME = (A(0), 2), (S(0), 2)  # Left / Right, good / bad


def one_decision(theta):
    """The factor graph of the one-decision problem."""
    left = sigmoid(theta)
    graph = SemiringFactorGraph()
    graph.push_back(probability([ACTION], [left, 1 - left]))
    graph.push_back(value([ACTION], [-1.0, 0.0]))
    graph.push_back(probability([ACTION, OUTCOME], [[0.9, 0.1], [0.2, 0.8]]))
    graph.push_back(value([OUTCOME], [10.0, 0.0]))
    return graph


theta_one = np.log(0.6 / 0.4)  # so that pi(L) = 0.6
graph = one_decision(theta_one)
J_one = graph.expectation()
conditional = graph.eliminateSequential(ordering(S(0), A(0))).at(1)
pi = table(conditional.probability(), [ACTION])  # the policy
advantage_one = table(conditional.surprise(), [ACTION])  # A(a) = Q(a) - J
Q_one = advantage_one + J_one
dpi = pi[0] * pi[1] * np.array([1.0, -1.0])  # derivative of pi(L), pi(R)
print("pi =", pi, " J =", J_one)
print("Q =", Q_one, " advantage =", advantage_one)
print("sum of dpi * Q         :", dpi @ Q_one)
print("sum of dpi * advantage :", dpi @ advantage_one)
print("E[dlog pi * advantage] :", pi @ (dpi / pi * advantage_one))
h = 1e-6
numeric = (one_decision(theta_one + h).expectation() -
           one_decision(theta_one - h).expectation()) / (2 * h)
print("finite difference      :", numeric)
assert np.isclose(J_one, 5.6) and np.allclose(Q_one, [8, 2])
assert np.allclose(advantage_one, [2.4, -3.6])
assert np.isclose(dpi @ Q_one, 1.44) and np.isclose(numeric, 1.44)
assert np.isclose(dpi @ advantage_one, 1.44)
assert np.isclose(pi @ (dpi / pi * advantage_one), 1.44)

# %% [markdown]
# ## The track with a parametric policy (Section 1)
#
# One parameter per cell, shared by both moves:
# $\pi_\theta(R \mid s) = \sigma(\theta_s)$. With $\theta = 0$ it is the coin
# flip of Chapter 1.

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
move_reward = np.array([[0.0, -1.0]] * 3)  # r(s, a)
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)


def policy_table(theta):
    """pi(a | s) for the logistic policy: columns Left, Right."""
    right = sigmoid(np.asarray(theta, float))
    return np.stack([1 - right, right], axis=1)


def policy_derivative(theta):
    """d pi(a | s) / d theta_s: columns Left, Right."""
    right = sigmoid(np.asarray(theta, float))
    slope = right * (1 - right)
    return np.stack([-slope, slope], axis=1)


# %% [markdown]
# ## Stage 1 with the module: backward and forward messages (Section 3)
#
# Elimination gives the expected return and, in the conditionals of the
# actions, the advantages. The marginals of the states are the forward
# messages.

# %%
state = lambda t: (S(t), 3)  # (key, cardinality)
action = lambda t: (A(t), 2)


def build(theta):
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    for t in range(2):
        keys = [state(t), action(t)]
        graph.push_back(probability(keys, policy_table(theta)))
        graph.push_back(probability(keys + [state(t + 1)], dynamics))
        graph.push_back(value(keys, move_reward))
    graph.push_back(value([state(2)], final_reward))
    return graph


def stage1(theta):
    """Evaluate the policy: J, the advantages A_t and the visitations d_t."""
    graph = build(theta)
    backward = ordering(S(2), A(1), S(1), A(0), S(0))
    bayes_net = graph.eliminateSequential(backward)
    # The conditionals of a_1 and a_0 are at positions 1 and 3.
    advantage = {1: table(bayes_net.at(1).surprise(), [state(1), action(1)]),
                 0: table(bayes_net.at(3).surprise(), [state(0), action(0)])}
    bayes_tree = graph.eliminateMultifrontal()
    visitation = {t: table(bayes_tree.marginalFactor(S(t)).probability(),
                           [state(t)]) for t in range(2)}
    return graph.expectation(), advantage, visitation


theta = np.zeros(3)
J, advantage, visitation = stage1(theta)
print("J =", J)
for t in range(2):
    print(f"d_{t} = {visitation[t]}")
    print(f"A_{t} =\n{advantage[t]}")
assert np.isclose(J, 1.4)

# %% [markdown]
# ## The gradient: forward message times local derivative times backward message (Section 3)


# %%
def gradient(theta, advantage, visitation):
    """dJ/dtheta_s = sum_t d_t(s) sum_a dpi(a|s) A_t(s, a)."""
    dpi = policy_derivative(theta)
    return sum(visitation[t] * (dpi * advantage[t]).sum(axis=1)
               for t in range(2))


g = gradient(theta, advantage, visitation)
print("gradient by forward-backward:", g)

h = 1e-6
numeric = np.array([
    (build(theta + h * e).expectation() - build(theta - h * e).expectation())
    / (2 * h) for e in np.eye(3)])
print("gradient by finite differences:", numeric)
assert np.allclose(g, [0.15, 1.0, 0.35]) and np.allclose(numeric, g)

# %% [markdown]
# ## The same gradient in one backward pass: the second-order semiring (Section 4)
#
# Each entry carries four numbers $(p, w, \dot p, \dot w)$: the pair of
# Chapter 1 and its derivative with respect to one parameter. The module does
# not implement entries of this kind, so this section, and only this one, runs
# a short elimination routine in numpy: multiply the factors of a bucket, sum
# over the variable, put the new factor back. Its result is checked against
# the gradient that the module gave above.


# %%
class SecondOrder:
    """Entries (p, w, dp, dw): the expectation semiring and its derivative."""

    @staticmethod
    def probability(p):
        zero = np.zeros_like(p)
        return (p, zero, zero, zero)

    @staticmethod
    def policy(stacked):
        p, dp = stacked  # the table and its derivative
        zero = np.zeros_like(p)
        return (p, zero, dp, zero)

    @staticmethod
    def reward(r):
        zero = np.zeros_like(r)
        return (np.ones_like(r), r, zero, zero)

    @staticmethod
    def times(a, b):
        return (a[0] * b[0],
                a[0] * b[1] + a[1] * b[0],
                a[0] * b[2] + a[2] * b[0],
                a[0] * b[3] + a[3] * b[0] + a[1] * b[2] + a[2] * b[1])

    @staticmethod
    def plus(a, axis):
        return tuple(channel.sum(axis) for channel in a)


def expand(factor, variables):
    """Reshape the channels of a factor so its axes follow `variables`."""
    names, channels = factor
    index = [names.index(v) for v in variables if v in names]
    return tuple(np.transpose(c, index).reshape(
        [c.shape[names.index(v)] if v in names else 1 for v in variables])
        for c in channels)


def multiply(semiring, factors):
    """The product of several factors, on the union of their variables."""
    variables = tuple(dict.fromkeys(v for names, _ in factors for v in names))
    product = expand(factors[0], variables)
    for factor in factors[1:]:
        product = semiring.times(product, expand(factor, variables))
    shape = np.broadcast_shapes(*(c.shape for c in product))
    return variables, tuple(np.broadcast_to(c, shape) for c in product)


def eliminate(semiring, terms, order):
    """Variable elimination. Returns what is left when no variable remains."""
    factors = [(names, getattr(semiring, kind)(np.asarray(table, float)))
               for kind, names, table in terms]
    for variable in order:
        bucket = [f for f in factors if variable in f[0]]
        factors = [f for f in factors if variable not in f[0]]
        names, product = multiply(semiring, bucket)
        separator = tuple(v for v in names if v != variable)
        factors.append(
            (separator, semiring.plus(product, names.index(variable))))
    return multiply(semiring, factors)[1]


def second_order_pass(theta, k):
    """(Z, J, dZ/dtheta_k, dJ/dtheta_k) from one elimination."""
    pi = policy_table(theta)
    dpi = np.zeros_like(pi)
    dpi[k] = policy_derivative(theta)[k]  # only row k depends on theta_k
    terms = [("probability", ("s0",), prior),
             ("reward", ("s2",), final_reward)]
    for t in range(2):
        s, a, n = f"s{t}", f"a{t}", f"s{t + 1}"
        terms += [("policy", (s, a), np.stack([pi, dpi])),
                  ("probability", (s, a, n), dynamics),
                  ("reward", (s, a), move_reward)]
    result = eliminate(SecondOrder, terms, ["s2", "a1", "s1", "a0", "s0"])
    return [float(channel) for channel in result]


for k in range(3):
    Z, J_k, dZ, dJ = second_order_pass(theta, k)
    print(f"theta_{k}: (Z, J, dZ, dJ) = ({Z:.4f}, {J_k:.4f}, {dZ:.4f}, {dJ:.4f})")
    assert np.isclose(J_k, 1.4) and np.isclose(dJ, g[k]) and abs(dZ) < 1e-12

# %% [markdown]
# ## The Fisher matrix (Section 5)
#
# It is built from the same forward messages. For this policy it is diagonal,
# because each parameter belongs to one cell.


# %%
def fisher(theta, visitation):
    """sum_t sum_s d_t(s) sum_a pi (dlog pi)(dlog pi)^T, for one theta per cell."""
    right = sigmoid(np.asarray(theta, float))
    # sum_a pi (dlog pi)^2 = dpi^2 / pi(L) + dpi^2 / pi(R) = pi(L) pi(R).
    per_cell = right * (1 - right)
    return np.diag(sum(visitation[t] for t in range(2)) * per_cell)


F = fisher(theta, visitation)
print("Fisher matrix =\n", F)
print("natural gradient =", np.linalg.solve(F, g))
assert np.allclose(np.diag(F), [0.25, 0.2, 0.05])

# A check over all trajectories. The product of all factors is one table with
# the probability and the return of every trajectory; the score of a
# trajectory is the sum of the scores of its two policy factors.
trajectory_keys = [state(0), action(0), state(1), action(1), state(2)]
joint = build(theta).product()
p_tau = table(joint.probability(), trajectory_keys)
R_tau = table(joint.value(), trajectory_keys)
pi, dpi = policy_table(theta), policy_derivative(theta)
brute_F, brute_g = np.zeros((3, 3)), np.zeros(3)
for s0, a0, s1, a1, s2 in np.argwhere(p_tau > 0):
    score = np.zeros(3)
    score[s0] += dpi[s0, a0] / pi[s0, a0]
    score[s1] += dpi[s1, a1] / pi[s1, a1]
    p = p_tau[s0, a0, s1, a1, s2]
    brute_F += p * np.outer(score, score)
    brute_g += p * score * R_tau[s0, a0, s1, a1, s2]
print("E[score score^T] over all trajectories =\n", brute_F)
print("E[score * R] over all trajectories =", brute_g)
assert np.allclose(brute_F, F) and np.allclose(brute_g, g)

# %% [markdown]
# ## Stage 2: improving the policy (Section 6)
#
# Stage 1 evaluates the current policy; Stage 2 moves the parameters along the
# gradient, or along the natural gradient. Each iteration is one elimination
# and one update. The natural gradient is damped, as in Levenberg-Marquardt:
# the Fisher matrix shrinks to zero as the policy becomes deterministic.


# %%
def run(natural, step, iterations, damping=0.01):
    theta = np.zeros(3)
    history = []
    for k in range(iterations + 1):
        J, advantage, visitation = stage1(theta)  # stage 1
        history.append(J)
        g = gradient(theta, advantage, visitation)
        if natural:
            damped = fisher(theta, visitation) + damping * np.eye(3)
            g = np.linalg.solve(damped, g)
        theta = theta + step * g  # stage 2
    return history, theta


plain, theta_plain = run(natural=False, step=1.0, iterations=50)
natural, theta_natural = run(natural=True, step=0.2, iterations=50)
for k in [0, 1, 2, 3, 5, 10, 20, 50]:
    print(f"iteration {k:2d}:  gradient J = {plain[k]:.4f}   "
          f"natural gradient J = {natural[k]:.4f}")
print("pi(R | s) after gradient steps:        ", sigmoid(theta_plain))
print("pi(R | s) after natural gradient steps:", sigmoid(theta_natural))
assert plain[-1] > plain[0] and natural[-1] > 5.9
