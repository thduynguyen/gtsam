# %% [markdown]
# # Chapter 1 examples: evaluating a policy
#
# This notebook runs the examples of [Chapter 1](https://thduynguyen.github.io/gtsam/chapter01). The
# chapter has the explanations and the derivations; here is only the code, in the
# same order, with a pointer to the section each part belongs to.

# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gtsam
from gtsam import (
    DecisionTreeFactor,
    DiscreteValues,
    HessianFactor,
    JacobianFactor,
    Ordering,
    SemiringDiscreteFactor,
    SemiringFactorGraph,
    SemiringGaussianFactor,
    VectorValues,
    noiseModel,
)
from gtsam.symbol_shorthand import A, S, U, X

np.set_printoptions(precision=4, suppress=True)


def ordering(*keys):
    """An elimination ordering holding the given keys."""
    result = Ordering()
    for key in keys:
        result.push_back(key)
    return result


def table(factor, keys):
    """Read a discrete table into an array indexed in the order of keys."""
    result = np.zeros([cardinality for _, cardinality in keys])
    for index in np.ndindex(*result.shape):
        values = DiscreteValues()
        for (key, _), value in zip(keys, index):
            values[key] = value
        result[index] = factor(values)
    return np.round(result, 9) + 0.0  # rounds off noise, avoids printing -0

# %% [markdown]
# ## The operators (Section 5)
#
# A semiring factor holds a probability and a value for every outcome. Two small
# tables, a transition and a reward, show the three operators: multiply,
# marginalize and condition.

# %%
a, s = (A(0), 2), (S(1), 2)  # an action and a next state, two values each

# A probability table lifted to (p, 0), and a reward table lifted to (1, r).
transition = SemiringDiscreteFactor(
    DecisionTreeFactor([a, s], "0.9 0.1 0.2 0.8"))
payoff = SemiringDiscreteFactor.Reward(DecisionTreeFactor(s, "10 0"))

# Multiply: probabilities multiply, values add.
product = transition * payoff
print("product, rows a, columns s")
print("p =", table(product.probability(), [a, s]).tolist())
print("v =", table(product.value(), [a, s]).tolist())
print("w =", table(product.weightedValue(), [a, s]).tolist(), " (stored: p * v)")

# Marginalize the next state: both stored tables are summed over it.
marginal = product.sum(ordering(S(1)))
print("\nafter marginalizing s")
print("p =", table(marginal.probability(), [a]).tolist())
print("v =", table(marginal.value(), [a]).tolist())

# Condition: divide the product by its marginal.
conditional = product / marginal
print("\nconditional of s given a")
print("p =", table(conditional.probability(), [a, s]).tolist())
print("v =", table(conditional.value(), [a, s]).tolist(), " (the surprise)")

# %% [markdown]
# ## Discrete example: a robot on a track (Section 7)
#
# A robot on three cells makes two moves, Left or Right, each with probability
# 0.5. A move succeeds with probability 0.8, otherwise the robot stays. Moving
# right costs 1, and ending in cell 2 pays 10.
#
# ![A robot on a three-cell track](https://raw.githubusercontent.com/thduynguyen/gtsam/feature/semiringfactor/gtsam/semiring/doc/figures/TrackWorld.svg)
#
# Each table of the problem is lifted to a semiring factor and added to a graph.

# %%
def state(t):
    """Discrete key of the cell after t moves: (key, number of values)."""
    return (S(t), 3)


def action(t):
    """Discrete key of move t, where 0 is Left and 1 is Right."""
    return (A(t), 2)


def probability(keys, values):
    """Lift a probability table to the semiring factor (p, 0)."""
    return SemiringDiscreteFactor(
        DecisionTreeFactor(keys, np.ravel(values).tolist()))


def reward(keys, values):
    """Lift a reward table to the semiring factor (1, r)."""
    return SemiringDiscreteFactor.Reward(
        DecisionTreeFactor(keys, np.ravel(values).tolist()))


# The tables of the problem. The first key of a table varies slowest.
start = [0.5, 0.5, 0.0]                          # p(s0)
coin_flip = [[0.5, 0.5]] * 3                     # pi(a | s)
slippery = [[[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]],  # p(s' | s, a)
            [[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]],
            [[0.0, 0.8, 0.2], [0.0, 0.0, 1.0]]]
move_reward = [[0.0, -1.0]] * 3                  # r(s, a)
final_reward = [0.0, 0.0, 10.0]                  # r(s2)

prior = probability([state(0)], start)
policy = [probability([state(t), action(t)], coin_flip) for t in range(2)]
dynamics = [
    probability([state(t), action(t), state(t + 1)], slippery)
    for t in range(2)
]
rewards = [reward([state(t), action(t)], move_reward) for t in range(2)]
final = reward([state(2)], final_reward)

graph = SemiringFactorGraph()
for factor in [prior, *policy, *dynamics, *rewards, final]:
    graph.push_back(factor)
print(graph.size(), "factors on", graph.keys().size(), "variables")

# %% [markdown]
# ### Step 1: eliminate the last state
#
# The new factor holds the expected final reward for each cell and move. The
# conditional holds the dynamics with the surprise of each outcome.

# %%
# The bucket of s2: the last dynamics factor and the final reward.
bucket = dynamics[1] * final
conditional_s2, phi_s1_a1 = bucket.eliminate(ordering(S(2)))

keys = [state(1), action(1)]
print("value of phi(s1, a1): rows s1, columns L and R")
print(table(phi_s1_a1.value(), keys))

print("\nc(s2 | s1 = 1, a1 = R), over s2 = 0, 1, 2")
print("probability:", table(conditional_s2.probability(), keys + [state(2)])[1, 1])
print("surprise:   ", table(conditional_s2.surprise(), keys + [state(2)])[1, 1])

# %% [markdown]
# ### Step 2: eliminate the last action
#
# The bucket's value is the action value Q1, the new factor's value is the state
# value V1, and the conditional's surprise is the advantage A1.

# %%
# The bucket of a1: the policy, the reward, and phi(s1, a1).
bucket = policy[1] * rewards[1] * phi_s1_a1
conditional_a1, phi_s1 = bucket.eliminate(ordering(A(1)))

print("Q1: rows s1, columns L and R")
print(table(bucket.value(), keys))
print("V1:", table(phi_s1.value(), [state(1)]))
print("A1: rows s1, columns L and R")
print(table(conditional_a1.surprise(), keys))

# %% [markdown]
# ### Step 3: eliminate the middle state

# %%
# The bucket of s1: the first dynamics factor and phi(s1) = (1, V1).
bucket = dynamics[0] * phi_s1
conditional_s1, phi_s0_a0 = bucket.eliminate(ordering(S(1)))

keys = [state(0), action(0)]
print("value of phi(s0, a0): rows s0, columns L and R")
print(table(phi_s0_a0.value(), keys))

print("\nc(s1 | s0 = 1, a0 = R), over s1 = 0, 1, 2")
print("probability:", table(conditional_s1.probability(), keys + [state(1)])[1, 1])
print("surprise:   ", table(conditional_s1.surprise(), keys + [state(1)])[1, 1])

# %% [markdown]
# ### Step 4: eliminate the first action

# %%
# The bucket of a0: the policy, the reward, and phi(s0, a0).
bucket = policy[0] * rewards[0] * phi_s0_a0
conditional_a0, phi_s0 = bucket.eliminate(ordering(A(0)))

print("Q0: rows s0, columns L and R")
print(table(bucket.value(), keys))
print("V0:", table(phi_s0.value(), [state(0)]))
print("A0: rows s0, columns L and R")
print(table(conditional_a0.surprise(), keys))

# %% [markdown]
# ### Step 5: eliminate the first state
#
# What is left is a constant whose value is the expected return J.

# %%
# The bucket of s0: the prior and phi(s0) = (1, V0).
bucket = prior * phi_s0
conditional_s0, constant = bucket.eliminate(ordering(S(0)))

print("J =", round(constant.expectation(), 6))
print("surprise of the start, V0 - J:", table(conditional_s0.surprise(), [state(0)]))

# %% [markdown]
# ### The result
#
# The graph does all of this in one call. The surprises of the conditionals along
# one trajectory add up to its return relative to the average.

# %%
# The same, in one call each: the expected return, and the Bayes net.
print("J =", round(graph.expectation(), 6))
bayes_net = graph.eliminateSequential(ordering(S(2), A(1), S(1), A(0), S(0)))

# Start in cell 1, move Right to cell 2, move Right and stay in cell 2.
trajectory = DiscreteValues()
for key, value in [(S(0), 1), (A(0), 1), (S(1), 2), (A(1), 1), (S(2), 2)]:
    trajectory[key] = value

total = 0.0
for i, name in enumerate(["s2", "a1", "s1", "a0", "s0"]):  # elimination order
    surprise = bayes_net.at(i).surprise()(trajectory)
    total += surprise
    print(f"surprise of {name}: {surprise:+.2f}")
print(f"sum: {total:+.2f}, and R - J = {(-1 - 1 + 10) - graph.expectation():+.2f}")

# %% [markdown]
# ## Continuous example: a robot on a line (Section 8)
#
# A robot on a line makes two moves toward the origin. Each move adds the action
# and Gaussian slip to its position, the policy goes half way back with some
# jitter, and the rewards penalize distance and effort.
#
# ![A robot on a line](https://raw.githubusercontent.com/thduynguyen/gtsam/feature/semiringfactor/gtsam/semiring/doc/figures/LineWorld.svg)
#
# A density is a `JacobianFactor` and a quadratic is a `HessianFactor`. The
# helper `describe` prints a value quadratic as z' M z + m' z + c, to compare
# with the formulas of the chapter.

# %%
I = np.eye(1)
zero = np.zeros(1)


def variance(v):
    """A scalar Gaussian noise model with the given variance."""
    return noiseModel.Isotropic.Variance(1, v)


def gaussian(*args):
    """Lift a Gaussian factor to the semiring factor (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def scalars(**values):
    """Values for scalar variables, e.g. scalars(x0=2.0, u0=-1.0)."""
    result = VectorValues()
    for name, value in values.items():
        key = (X if name[0] == "x" else U)(int(name[1:]))
        result.insert(key, np.array([value]))
    return result


def describe(quadratic):
    """Print a value quadratic as z' M z + m' z + c."""
    if quadratic is None:
        print("  zero")
        return
    names = [gtsam.DefaultKeyFormatter(key) for key in quadratic.keys()]
    if names:
        print("  variables z =", names)
        print("  M =", (np.round(0.5 * quadratic.information(), 9) + 0.0).tolist())
        print("  m =", (np.round(-np.ravel(quadratic.linearTerm()), 9) + 0.0).tolist())
    print("  c =", round(0.5 * quadratic.constantTerm(), 9) + 0.0)


# x0 = 2 + noise of variance 1.
line_prior = gaussian(X(0), I, np.array([2.0]), variance(1.0))
# Policy: u + 0.5 x = e, with e of variance 0.1.
line_policy = [
    gaussian(U(t), I, X(t), 0.5 * I, zero, variance(0.1)) for t in range(2)
]
# Dynamics: x' - x - u = w, with w of variance 0.5.
line_dynamics = [
    gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero, variance(0.5))
    for t in range(2)
]

line_graph = SemiringFactorGraph()
line_graph.push_back(line_prior)
for t in range(2):
    line_graph.push_back(line_policy[t])
    line_graph.push_back(line_dynamics[t])
    line_graph.push_back(penalty(X(t)))  # -x^2
    line_graph.push_back(penalty(U(t)))  # -u^2
line_graph.push_back(penalty(X(2)))      # -x2^2
print(line_graph.size(), "factors on", line_graph.keys().size(), "variables")

# %% [markdown]
# ### Step 1: eliminate the last state

# %%
# The bucket of x2: the last dynamics factor and the final reward.
bucket = line_dynamics[1].multiply(penalty(X(2)))
conditional_x2, phi_x1_u1 = bucket.eliminate(ordering(X(2)))

print("value of phi(x1, u1), expected -((x1 + u1)^2 + 0.5):")
describe(phi_x1_u1.value())

# %% [markdown]
# ### Step 2: eliminate the last action

# %%
# The bucket of u1: the policy, the two penalties, and phi(x1, u1).
bucket = (line_policy[1].multiply(penalty(X(1))).multiply(penalty(U(1)))
          .multiply(phi_x1_u1))
conditional_u1, phi_x1 = bucket.eliminate(ordering(U(1)))

print("V1, expected -(1.5 x^2 + 0.7):")
describe(phi_x1.value())
print("A1, expected 0.2 - 2 (u + 0.5 x)^2 = -2 u^2 - 2 u x - 0.5 x^2 + 0.2:")
describe(conditional_u1.surprise())

# %% [markdown]
# ### Step 3: eliminate the middle state

# %%
# The bucket of x1: the first dynamics factor and phi(x1) = (1, V1).
bucket = line_dynamics[0].multiply(phi_x1)
conditional_x1, phi_x0_u0 = bucket.eliminate(ordering(X(1)))

print("value of phi(x0, u0), expected -(1.5 (x0 + u0)^2 + 1.45):")
describe(phi_x0_u0.value())

# %% [markdown]
# ### Step 4: eliminate the first action

# %%
# The bucket of u0: the policy, the two penalties, and phi(x0, u0).
bucket = (line_policy[0].multiply(penalty(X(0))).multiply(penalty(U(0)))
          .multiply(phi_x0_u0))
conditional_u0, phi_x0 = bucket.eliminate(ordering(U(0)))

print("V0, expected -(1.625 x^2 + 1.7):")
describe(phi_x0.value())
print("A0, expected 0.25 + 0.025 x^2 - 2.5 (u + 0.6 x)^2"
      " = -2.5 u^2 - 3 u x - 0.875 x^2 + 0.25:")
describe(conditional_u0.surprise())

# %% [markdown]
# ### Step 5: eliminate the first state

# %%
# The bucket of x0: the prior and phi(x0) = (1, V0).
bucket = line_prior.multiply(phi_x0)
conditional_x0, constant = bucket.eliminate(ordering(X(0)))

print("J =", round(constant.expectation(), 6))
print("surprise of the start, expected 8.125 - 1.625 x^2:")
describe(conditional_x0.surprise())

# %% [markdown]
# ### The result
#
# The value functions are parabolas in the position. The advantage of the first
# move is a parabola in the move, highest at the best move, which is close to
# the policy's average move but not on it.

# %%
positions = np.linspace(-3, 3, 61)
moves = np.linspace(-2.6, 0.2, 57)

figure = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Value functions",
                    "Advantage of the first move, from x0 = 2"))
figure.add_trace(go.Scatter(x=positions, y=-positions**2, name="V2",
                            line=dict(dash="dash")), row=1, col=1)
figure.add_trace(go.Scatter(
    x=positions, y=[phi_x1.value(scalars(x1=x)) for x in positions],
    name="V1"), row=1, col=1)
figure.add_trace(go.Scatter(
    x=positions, y=[phi_x0.value(scalars(x0=x)) for x in positions],
    name="V0"), row=1, col=1)
figure.add_trace(go.Scatter(
    x=moves, y=[conditional_u0.surprise(scalars(x0=2.0, u0=u)) for u in moves],
    name="A0(2, u)"), row=1, col=2)
figure.add_trace(go.Scatter(
    x=[-1.2, -1.0],
    y=[conditional_u0.surprise(scalars(x0=2.0, u0=u)) for u in (-1.2, -1.0)],
    mode="markers+text", text=["best move", "policy's mean"],
    textposition=["top left", "top right"], name="moves",
    marker=dict(size=10)), row=1, col=2)
figure.update_xaxes(title_text="position x", row=1, col=1)
figure.update_xaxes(title_text="first move u0", row=1, col=2)
figure.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
figure.show()

# %%
# The same, in one call each: the expected return, and the Bayes net.
print("J =", round(line_graph.expectation(), 6))
line_bayes_net = line_graph.eliminateSequential(
    ordering(X(2), U(1), X(1), U(0), X(0)))

trajectory = scalars(x0=2.0, u0=-1.0, x1=1.0, u1=-0.5, x2=0.5)
total = 0.0
for i, name in enumerate(["x2", "u1", "x1", "u0", "x0"]):  # elimination order
    surprise = line_bayes_net.at(i).surprise(trajectory)
    total += surprise
    print(f"surprise of {name}: {surprise:+.3f}")
print(f"sum: {total:+.3f}, and R - J = {-6.5 - line_graph.expectation():+.3f}")
