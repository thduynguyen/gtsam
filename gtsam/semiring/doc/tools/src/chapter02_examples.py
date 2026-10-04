# %% [markdown]
# # Chapter 2 examples: the semiring family
#
# This notebook runs the examples of
# [Chapter 2](https://thduynguyen.github.io/gtsam/chapter02). One short
# variable-elimination routine is run on the track of Chapter 1 with five
# different semirings, and each answers a different question.

# %%
import numpy as np
from scipy.special import logsumexp, softmax

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track of Chapter 1
#
# Three cells, two moves. A table has one axis per variable.

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
policy = np.full((3, 2), 0.5)  # pi(a | s): a coin flip
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0]] * 3)  # r(s, a): moving right costs 1
final = np.array([0.0, 0.0, 10.0])  # r(s2): the charger is in cell 2

# The eight terms of the two-move problem: (kind, variables, table).
terms = [
    ("probability", ("s0",), prior),
    ("probability", ("s0", "a0"), policy),
    ("reward", ("s0", "a0"), reward),
    ("probability", ("s0", "a0", "s1"), dynamics),
    ("probability", ("s1", "a1"), policy),
    ("reward", ("s1", "a1"), reward),
    ("probability", ("s1", "a1", "s2"), dynamics),
    ("reward", ("s2",), final),
]
order = ["s2", "a1", "s1", "a0", "s0"]  # backward in time

# %% [markdown]
# ## Five semirings (Section 2)
#
# An entry of a factor is a tuple of numbers, its *channels*. A semiring says
# how to lift a probability and a reward to an entry, how to multiply two
# entries, and how to add entries over the values of a variable.


# %%
class SumProduct:
    """One channel, a probability: ordinary elimination."""
    name = "sum-product"
    probability = staticmethod(lambda p: (p,))
    reward = staticmethod(lambda r: (np.ones_like(r),))  # rewards are ignored
    times = staticmethod(lambda a, b: (a[0] * b[0],))
    plus = staticmethod(lambda a, axis: (a[0].sum(axis),))
    read = staticmethod(lambda a: a[0])


class Expectation:
    """Probability and weighted value (p, w), w = p v: Chapter 1."""
    name = "expectation"
    probability = staticmethod(lambda p: (p, np.zeros_like(p)))
    reward = staticmethod(lambda r: (np.ones_like(r), r))
    times = staticmethod(lambda a, b: (a[0] * b[0], a[0] * b[1] + a[1] * b[0]))
    plus = staticmethod(lambda a, axis: (a[0].sum(axis), a[1].sum(axis)))
    read = staticmethod(lambda a: a[1] / a[0])


class MaxSum:
    """One channel, a value: values add, and the best outcome is kept."""
    name = "max-sum"
    # A possible outcome contributes 0, an impossible one minus infinity.
    probability = staticmethod(
        lambda p: (np.where(p > 0, 0.0, -np.inf),))
    reward = staticmethod(lambda r: (r,))
    times = staticmethod(lambda a, b: (a[0] + b[0],))
    plus = staticmethod(lambda a, axis: (a[0].max(axis),))
    read = staticmethod(lambda a: a[0])


class Tilted:
    """Probability and tilted value (p, m), m = p exp(kappa v)."""

    def __init__(self, kappa):
        self.kappa, self.name = kappa, f"tilted, kappa = {kappa:g}"

    def probability(self, p):
        return (p, p)

    def reward(self, r):
        return (np.ones_like(r), np.exp(self.kappa * r))

    def times(self, a, b):
        return (a[0] * b[0], a[1] * b[1])

    def plus(self, a, axis):
        return (a[0].sum(axis), a[1].sum(axis))

    def read(self, a):
        return np.log(a[1] / a[0]) / self.kappa


# %% [markdown]
# ## One elimination routine for all of them (Section 1)
#
# This is bucket elimination: for each variable, multiply the factors that
# contain it, add over its values, and put the new factor back.


# %%
def expand(factor, variables):
    """Reshape the channels of a factor so its axes follow `variables`."""
    names, channels = factor
    index = [names.index(v) for v in variables if v in names]
    shape = [-1 if v in names else 1 for v in variables]
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
        axis = names.index(variable)
        separator = tuple(v for v in names if v != variable)
        factors.append((separator, semiring.plus(product, axis)))
    return multiply(semiring, factors)[1]


for semiring in [SumProduct(), Expectation(), MaxSum()]:
    result = eliminate(semiring, terms, order)
    print(f"{semiring.name:12s} {float(semiring.read(result)):8.4f}")

# %% [markdown]
# Sum-product gives 1: the probabilities of all trajectories sum to one.
# The expectation semiring gives the expected return $J = 1.4$ of Chapter 1.
# Max-sum gives 9, the return of the single best trajectory: start in cell 1,
# move Right into cell 2, then move Left and slip, which keeps the robot at
# the charger for free. The best trajectory counts on luck.

# %%
assert np.isclose(SumProduct.read(eliminate(SumProduct(), terms, order)), 1.0)
assert np.isclose(Expectation.read(eliminate(Expectation(), terms, order)), 1.4)
assert np.isclose(MaxSum.read(eliminate(MaxSum(), terms, order)), 9.0)

# %% [markdown]
# ## The tilted family (Section 3)
#
# The tilted mean $\frac{1}{\kappa} \log \mathbb{E}[e^{\kappa R}]$ moves from
# the worst return through the average to the best one as $\kappa$ grows.

# %%
returns = {}
for kappa in [-20, -2, -0.5, -0.01, 0.01, 0.5, 2, 20]:
    semiring = Tilted(kappa)
    returns[kappa] = float(semiring.read(eliminate(semiring, terms, order)))
    print(f"kappa = {kappa:6.2f}   tilted mean of the return = "
          f"{returns[kappa]:7.4f}")

# %%
assert abs(returns[0.01] - 1.4) < 0.1 and abs(returns[-0.01] - 1.4) < 0.1
assert abs(returns[20] - 9.0) < 0.3  # approaches the best return
assert abs(returns[-20] + 2.0) < 0.3  # approaches the worst return

# %% [markdown]
# The soft maximum with temperature $\eta$ is the same rule with
# $\kappa = 1 / \eta$: for $\eta = 2$ it gives the number of $\kappa = 0.5$.

# %%
eta = 2.0
soft = Tilted(1 / eta)
soft_maximum = float(soft.read(eliminate(soft, terms, order)))
print(f"soft maximum, eta = {eta:g}: {soft_maximum:.4f}")
assert np.isclose(soft_maximum, returns[0.5])

# %% [markdown]
# A smaller example of the tilted mean: a fair coin that pays 0 or 10.

# %%
coin_p, coin_v = np.array([0.5, 0.5]), np.array([0.0, 10.0])
for kappa in [-5, -0.5, -0.01, 0.01, 0.5, 5]:
    tilted = np.log(coin_p @ np.exp(kappa * coin_v)) / kappa
    print(f"kappa = {kappa:5.2f}   tilted mean = {tilted:.3f}")
assert np.isclose(np.log(coin_p @ np.exp(0.5 * coin_v)) / 0.5, 8.627, atol=1e-3)
assert np.isclose(np.log(coin_p @ np.exp(-0.5 * coin_v)) / -0.5, 1.373, atol=1e-3)

# %% [markdown]
# The best return is 9 and the worst is $-2$ (move right twice from cell 0 and
# slip both times). A brute-force check over all trajectories:

# %%
trajectories = []
for s0 in range(3):
    for a0 in range(2):
        for s1 in range(3):
            for a1 in range(2):
                for s2 in range(3):
                    p = (prior[s0] * policy[s0, a0] * dynamics[s0, a0, s1] *
                         policy[s1, a1] * dynamics[s1, a1, s2])
                    if p > 0:
                        trajectories.append(
                            (p, reward[s0, a0] + reward[s1, a1] + final[s2]))
p, ret = np.array(trajectories).T
print("number of possible trajectories:", len(p))
print("sum of probabilities:", p.sum())
print("expected return:", p @ ret)
print("best and worst return:", ret.max(), ret.min())
for kappa in [-2, 0.5, 2]:
    brute = np.log(p @ np.exp(kappa * ret)) / kappa
    assert np.isclose(brute, returns[kappa])
    print(f"kappa = {kappa:4.1f}: brute force {brute:.4f}")

# %% [markdown]
# ## Conditionals and the normalization invariant (Section 5)
#
# Take the bucket of the last action $a_1$ under the coin-flip policy. Its
# product is $(\pi(a \mid s), Q_1(s, a))$. Dividing by the new factor gives
# the conditional, and adding the conditional over the action gives the *one*
# of the semiring, in every row.

# %%
Q1 = reward + dynamics @ final  # Q_1(s, a) = r(s, a) + sum_s' p(s'|s,a) r(s')
print("Q1 =\n", Q1)

# Expectation semiring: the surprise is the advantage, with mean zero.
V1 = (policy * Q1).sum(1, keepdims=True)
advantage = Q1 - V1
print("V1 =", V1.ravel())
print("advantage =\n", advantage)
print("sum of pi * advantage per cell:", (policy * advantage).sum(1))
assert np.allclose((policy * advantage).sum(1), 0)

# Max-sum: the surprise is the regret, with maximum zero.
regret = Q1 - Q1.max(1, keepdims=True)
print("regret =\n", regret)
assert np.allclose(regret.max(1), 0)

# Tilted, kappa = 1: the surprise is the soft advantage, with tilted mean zero.
kappa = 1.0
soft_V1 = np.log((policy * np.exp(kappa * Q1)).sum(1, keepdims=True)) / kappa
soft_advantage = Q1 - soft_V1
tilted_policy = policy * np.exp(kappa * soft_advantage)
print("soft V1 =", soft_V1.ravel())
print("tilted policy pi * exp(kappa * soft advantage) =\n", tilted_policy)
print("its rows sum to", tilted_policy.sum(1))
assert np.allclose(tilted_policy.sum(1), 1)

# With a uniform policy, the tilted policy is the softmax of kappa * Q, and
# the soft value is the log-sum-exp of kappa * Q (up to the constant log 2).
assert np.allclose(tilted_policy, softmax(kappa * Q1, axis=1))
assert np.allclose(soft_V1.ravel(),
                   (logsumexp(kappa * Q1, axis=1) - np.log(2)) / kappa)

# %% [markdown]
# ## The log-dual form (Section 6)
#
# In the form $(\ell, v)$, with $\ell = \log p$, multiplication adds both
# channels. A product of many small probabilities then stays finite.

# %%
def log_dual_plus(l, v, axis):
    """Add (l, v) entries over an axis: log-sum-exp and a weighted mean."""
    total = logsumexp(l, axis=axis)
    weights = np.exp(l - np.expand_dims(total, axis))
    return total, (weights * v).sum(axis)


# 400 unnormalized factors on one binary variable, each with a small
# probability and a reward of 1 for the second outcome.
p = np.array([1e-3, 2e-3])
r = np.array([0.0, 1.0])
n = 400

# Stored form (p, w): the product underflows to zero.
p_product = p ** n
w_product = p_product * (n * r)
with np.errstate(invalid="ignore", divide="ignore"):
    print("(p, w):", w_product.sum() / p_product.sum())

# Log-dual form: both channels add.
l_product, v_product = n * np.log(p), n * r
total, value = log_dual_plus(l_product, v_product, axis=0)
print("(l, v): log p =", total, " value =", value)
assert np.isclose(value, 400.0)
