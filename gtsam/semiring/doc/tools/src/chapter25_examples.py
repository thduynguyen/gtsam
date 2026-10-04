# %% [markdown]
# # Chapter 25 examples: distributional RL
#
# This notebook runs the examples of
# [Chapter 25](https://thduynguyen.github.io/gtsam/chapter25). The elimination
# routine of Chapter 2 is run with a semiring whose entries are whole
# distributions of the return, on the track of Chapter 1. Then the
# distributional Bellman equation is iterated on the endless track of
# Chapter 3, on a fixed grid of returns as in C51.

# %%
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track of Chapter 1 and the elimination routine of Chapter 2

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
policy = np.full((3, 2), 0.5)  # pi(a | s): a coin flip
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
move_reward = np.array([[0.0, -1.0]] * 3)  # r(s, a)
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)

terms = [
    ("probability", ("s0",), prior),
    ("probability", ("s0", "a0"), policy),
    ("reward", ("s0", "a0"), move_reward),
    ("probability", ("s0", "a0", "s1"), dynamics),
    ("probability", ("s1", "a1"), policy),
    ("reward", ("s1", "a1"), move_reward),
    ("probability", ("s1", "a1", "s2"), dynamics),
    ("reward", ("s2",), final_reward),
]
order = ["s2", "a1", "s1", "a0", "s0"]  # backward in time


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


def eliminate(semiring, terms, order, keep=()):
    """Variable elimination; stops before the variables in `keep`."""
    factors = [(names, getattr(semiring, kind)(np.asarray(table, float)))
               for kind, names, table in terms]
    for variable in order:
        if variable in keep:
            break
        bucket = [f for f in factors if variable in f[0]]
        factors = [f for f in factors if variable not in f[0]]
        names, product = multiply(semiring, bucket)
        separator = tuple(v for v in names if v != variable)
        factors.append(
            (separator, semiring.plus(product, names.index(variable))))
    return factors if keep else multiply(semiring, factors)[1]


# %% [markdown]
# ## The convolution semiring (Section 2)
#
# An entry holds one number for every possible value $z$ of the accumulated
# reward: the probability of the outcome times the probability that the
# reward so far is $z$. On the track the reward is an integer from $-2$ to
# $10$, so an entry is 13 numbers.

# %%
Z = np.arange(-2, 11)  # the possible accumulated rewards


class Convolution:
    """Entries h(z): probability mass at every accumulated reward z."""

    @staticmethod
    def probability(p):
        # Probability p, and no reward: all the mass at z = 0.
        return tuple(p if z == 0 else np.zeros_like(p) for z in Z)

    @staticmethod
    def reward(r):
        # Probability one, and the reward r: all the mass at z = r.
        return tuple((r == z).astype(float) for z in Z)

    @staticmethod
    def times(a, b):
        # Rewards add: the masses are convolved.
        result = []
        for z in Z:
            total = 0.0
            for i, z1 in enumerate(Z):
                z2 = z - z1
                if Z[0] <= z2 <= Z[-1]:
                    total = total + a[i] * b[z2 - Z[0]]
            result.append(total)
        return tuple(result)

    @staticmethod
    def plus(a, axis):
        # Exclusive outcomes: the masses are added.
        return tuple(channel.sum(axis) for channel in a)


distribution = np.array(
    [float(h) for h in eliminate(Convolution, terms, order)])
for z, h in zip(Z, distribution):
    if h > 0:
        print(f"P(R = {z:2d}) = {h:.4f}")
print("total probability:", distribution.sum())
print("mean:", distribution @ Z)
assert np.isclose(distribution.sum(), 1) and np.isclose(distribution @ Z, 1.4)

# %% [markdown]
# A check by enumerating the 24 possible trajectories.

# %%
brute = np.zeros(len(Z))
count = 0
for s0 in range(3):
    for a0 in range(2):
        for s1 in range(3):
            for a1 in range(2):
                for s2 in range(3):
                    p = (prior[s0] * policy[s0, a0] * dynamics[s0, a0, s1] *
                         policy[s1, a1] * dynamics[s1, a1, s2])
                    if p > 0:
                        count += 1
                        ret = int(move_reward[s0, a0] + move_reward[s1, a1] +
                                  final_reward[s2])
                        brute[ret - Z[0]] += p
print("possible trajectories:", count)
assert count == 24 and np.allclose(brute, distribution)

# %% [markdown]
# ## The other semirings are summaries of this one (Section 3)
#
# The expectation semiring keeps the total mass and the first moment. The
# tilted semiring keeps the total mass and one exponential moment. A third
# number, the second moment, gives the variance.

# %%
p, w = distribution.sum(), distribution @ Z
print("(p, w) =", (p, w), " value w / p =", w / p)
for kappa, expected in [(-0.5, -0.2768), (0.5, 5.4251), (2.0, 7.6490)]:
    m = distribution @ np.exp(kappa * Z)
    tilted = np.log(m / p) / kappa
    print(f"kappa = {kappa:4.1f}: tilted mean {tilted:.4f}")
    assert np.isclose(tilted, expected, atol=1e-4)

variance = distribution @ Z ** 2 - (distribution @ Z) ** 2
print("variance:", variance, " standard deviation:", np.sqrt(variance))


class SecondMoment:
    """Entries (p, w, w2): mass, first and second moment of the reward."""

    @staticmethod
    def probability(p):
        return (p, np.zeros_like(p), np.zeros_like(p))

    @staticmethod
    def reward(r):
        return (np.ones_like(r), r, r ** 2)

    @staticmethod
    def times(a, b):
        return (a[0] * b[0],
                a[0] * b[1] + a[1] * b[0],
                a[0] * b[2] + 2 * a[1] * b[1] + a[2] * b[0])

    @staticmethod
    def plus(a, axis):
        return tuple(channel.sum(axis) for channel in a)


p, w, w2 = [float(c) for c in eliminate(SecondMoment, terms, order)]
print("second-moment semiring: (p, w, w2) =", (p, w, w2),
      " variance =", w2 - w ** 2)
assert np.isclose(w2 - w ** 2, variance)

# The map from a distribution to its moments turns a convolution into the
# product of the expectation semiring.
rng = np.random.default_rng(0)
h1, h2 = rng.random(5), rng.random(5)
z5 = np.arange(5)
product = np.convolve(h1, h2)
p1, w1, p2, w2 = h1.sum(), h1 @ z5, h2.sum(), h2 @ z5
assert np.isclose(product.sum(), p1 * p2)
assert np.isclose(product @ np.arange(9), p1 * w2 + p2 * w1)

# %% [markdown]
# ## Risk measures read from the distribution (Section 3)


# %%
def cvar(values, probabilities, level):
    """Mean of the worst `level` fraction of the outcomes."""
    remaining, total = level, 0.0
    for value, probability in sorted(zip(values, probabilities)):
        take = min(probability, remaining)
        total += take * value
        remaining -= take
        if remaining <= 0:
            break
    return total / level


for level in [1.0, 0.5, 0.25, 0.1]:
    print(f"CVaR at level {level:4.2f}: {cvar(Z, distribution, level):.4f}")
print("P(R >= 8):", distribution[Z >= 8].sum())
assert np.isclose(cvar(Z, distribution, 1.0), 1.4)

# %% [markdown]
# ## The value factor now holds a distribution per state (Section 4)
#
# Stopping the elimination before $s_1$ leaves a factor on $s_1$: for each
# cell, the distribution of the reward still to come with one move left.

# %%
factors = eliminate(Convolution, terms, order, keep=("s1",))
(names, channels), = [f for f in factors if f[0] == ("s1",)]
to_go = np.stack(channels, axis=1)  # rows: cell s1, columns: z
for cell in range(3):
    support = {int(z): round(float(h), 3)
               for z, h in zip(Z, to_go[cell]) if h > 0}
    print(f"cell {cell}: {support}, mean {to_go[cell] @ Z:.2f}")
assert np.allclose(to_go @ Z, [-0.5, 3.5, 5.5])  # V_1 of Chapter 1

# %% [markdown]
# ## The distributional Bellman equation on a grid (Section 5)
#
# The endless track of Chapter 3, with discount 0.9 and the coin-flip policy.
# The distribution of the discounted return of each cell is stored as 51
# probabilities on a fixed grid, as in C51. One backup shifts and shrinks the
# grid of the next state, $r + \gamma z$, and projects the result back onto
# the grid.

# %%
gamma = 0.9
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
atoms = np.linspace(-10, 20, 51)  # returns lie between -1/0.1 and 2/0.1
spacing = atoms[1] - atoms[0]


def project(values, probabilities):
    """Put the masses at `values` onto the grid, splitting between neighbours."""
    position = (np.clip(values, atoms[0], atoms[-1]) - atoms[0]) / spacing
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(atoms) - 1)
    weight_upper = position - lower
    result = np.zeros(len(atoms))
    np.add.at(result, lower, probabilities * (1 - weight_upper))
    np.add.at(result, upper, probabilities * weight_upper)
    return result


categorical = np.zeros((3, len(atoms)))
categorical[:, np.argmin(np.abs(atoms))] = 1  # start: the return is 0
for sweep in range(300):
    updated = np.zeros_like(categorical)
    for s in range(3):
        for a in range(2):
            for n in range(3):
                weight = policy[s, a] * dynamics[s, a, n]
                if weight > 0:
                    updated[s] += weight * project(
                        reward[s, a] + gamma * atoms, categorical[n])
    categorical = updated
means = categorical @ atoms
deviations = np.sqrt(categorical @ atoms ** 2 - means ** 2)
print("means of the grid distributions:", means)
print("V of Chapter 3:                  [-0.2248  1.1017  4.1231]")
print("standard deviations:", deviations)
assert np.allclose(means, [-0.2248, 1.1017, 4.1231], atol=1e-3)

# %% [markdown]
# A check by simulation with a fixed seed, from cell 1. The discounted return
# of an endless episode, and the *undiscounted* return of an episode that
# ends at random with probability $1 - \gamma$ per step (Chapter 3). The two
# have the same mean and different distributions.

# %%
rng = np.random.default_rng(0)
episodes = 20_000
discounted, terminated = np.zeros(episodes), np.zeros(episodes)
for i in range(episodes):
    s, weight = 1, 1.0
    for t in range(150):
        a = rng.integers(2)
        discounted[i] += weight * reward[s, a]
        weight *= gamma
        s = rng.choice(3, p=dynamics[s, a])
    s = 1
    while True:
        a = rng.integers(2)
        terminated[i] += reward[s, a]
        if rng.random() > gamma:
            break
        s = rng.choice(3, p=dynamics[s, a])
print(f"discounted return:   mean {discounted.mean():.3f}, "
      f"standard deviation {discounted.std():.3f}")
print(f"random termination:  mean {terminated.mean():.3f}, "
      f"standard deviation {terminated.std():.3f}")
print(f"grid, cell 1:        mean {means[1]:.3f}, "
      f"standard deviation {deviations[1]:.3f}")
assert abs(discounted.mean() - 1.1017) < 0.1
assert abs(terminated.mean() - 1.1017) < 0.2
assert terminated.std() > 1.5 * discounted.std()

# Quantiles of the discounted return from cell 1: grid against simulation.
cumulative = np.cumsum(categorical[1])
for level in [0.1, 0.5, 0.9]:
    on_grid = atoms[np.searchsorted(cumulative, level)]
    print(f"quantile {level}: grid {on_grid:.2f}, simulation "
          f"{np.quantile(discounted, level):.2f}")

# %% [markdown]
# ## The sampled version: categorical TD (Section 5)
#
# C51 without the network: the same backup from one sampled transition at a
# time, mixed into the stored distribution with a step size.

# %%
rng = np.random.default_rng(1)
learned = np.zeros((3, len(atoms)))
learned[:, np.argmin(np.abs(atoms))] = 1
s, step = 1, 0.02
for _ in range(60_000):
    a = rng.integers(2)
    n = rng.choice(3, p=dynamics[s, a])
    target = project(reward[s, a] + gamma * atoms, learned[n])
    learned[s] = (1 - step) * learned[s] + step * target
    s = n
print("means of the learned distributions:", learned @ atoms)
assert np.allclose(learned @ atoms, [-0.2248, 1.1017, 4.1231], atol=0.6)
