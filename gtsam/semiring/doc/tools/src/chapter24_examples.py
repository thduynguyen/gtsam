# %% [markdown]
# # Chapter 24 examples: exploration and dual control
#
# This notebook runs the examples of
# [Chapter 24](https://thduynguyen.github.io/gtsam/chapter24) with the
# `gtsam/semiring` module: a bandit with two arms whose success probabilities
# are unknown. The best way to explore is computed exactly, by eliminating a
# factor graph over beliefs with the maximum at the pulls and the average at
# the outcomes, and compared with the usual heuristics, each evaluated exactly
# on the same graph with its policy factors.

# %%
from functools import lru_cache
from math import comb, exp, lgamma

import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import GaussianFactorGraph, JacobianFactor, noiseModel
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam import SemiringRules, SemiringSum
from gtsam.symbol_shorthand import A, B, G, Y
from scipy.stats import beta as beta_distribution

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The problem and the belief (Section 1)
#
# A belief is four counts: successes and failures of arm U (unknown) and of
# arm K (known fairly well). A Beta distribution with these counts is what is
# known about each arm's success probability.

# %%
U, K = 0, 1  # the two arms
start = (1, 1, 12, 8)  # (successes, failures) of U, then of K, as Beta counts


def mean(belief, arm):
    """Probability that the next pull of an arm succeeds, given the belief."""
    successes, failures = belief[2 * arm], belief[2 * arm + 1]
    return successes / (successes + failures)


def after(belief, arm, success):
    """The belief after one more pull: one count goes up by one."""
    counts = list(belief)
    counts[2 * arm + (0 if success else 1)] += 1
    return tuple(counts)


print("probability of success, arm U:", mean(start, U))
print("probability of success, arm K:", mean(start, K))
print("belief after a success of U:", after(start, U, True))
print("belief after a failure of U:", after(start, U, False))

# %% [markdown]
# The helpers of the other notebooks. The pull $a_t$ and its outcome $y_t$
# (0 = failure, 1 = success) are binary variables.


# %%
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


pull = lambda t: (A(t), 2)  # (key, cardinality)
outcome = lambda t: (Y(t), 2)
maximum = SemiringSum.Maximum()
at_pulls = SemiringRules()  # the maximum at every pull, the average elsewhere
at_pulls.setAll([A(t) for t in range(20)], maximum)

# %% [markdown]
# ## The parameters are eliminated first (Section 2)
#
# The success probabilities are never known, so they are eliminated before
# every pull and outcome. They are continuous with a Beta density, which is
# neither of the module's two families, so this one elimination is done in
# closed form: it leaves a single factor that joins all pulls and outcomes,
#
# $$p(y_0, \ldots, y_{T-1} \mid a_0, \ldots, a_{T-1}) =
# \frac{\mathrm{B}(\text{counts of U after})}
# {\mathrm{B}(\text{counts of U before})} \cdot
# \frac{\mathrm{B}(\text{counts of K after})}
# {\mathrm{B}(\text{counts of K before})},$$
#
# with $\mathrm{B}$ the Beta function of the two counts of an arm. The rest is
# the module: the outcome of a pull is eliminated by average, then the pull by
# maximum, from the last pull to the first.


# %%
def log_beta(successes, failures):
    return lgamma(successes) + lgamma(failures) - lgamma(successes + failures)


def history_factor(start, pulls):
    """p(all outcomes | all pulls), with the parameters summed out."""
    result = np.zeros((2, 2) * pulls)  # indexed by a_0, y_0, a_1, y_1, ...
    for history in np.ndindex(*result.shape):
        belief = start
        for pulled, success in zip(history[0::2], history[1::2]):
            belief = after(belief, pulled, success)
        result[history] = exp(
            log_beta(*belief[:2]) - log_beta(*start[:2]) +
            log_beta(*belief[2:]) - log_beta(*start[2:]))
    return result


def history_graph(start, pulls):
    """One factor over the whole history, and a reward factor per outcome."""
    keys = [key for t in range(pulls) for key in (pull(t), outcome(t))]
    graph = SemiringFactorGraph()
    graph.push_back(probability(keys, history_factor(start, pulls)))
    for t in range(pulls):
        graph.push_back(value([outcome(t)], [0, 1]))
    return graph


def history_order(pulls):
    """Backward in time: each outcome before the pull that produced it."""
    return [key for t in reversed(range(pulls)) for key in (Y(t), A(t))]


by_history = {
    pulls: history_graph(start, pulls).expectation(
        ordering(*history_order(pulls)), at_pulls)
    for pulls in [1, 2, 5]}
print("best expected number of successes, by history:", by_history)
print("entries of the factor over the history, 5 pulls:", 4 ** 5)
assert np.isclose(by_history[1], 0.6) and np.isclose(by_history[2], 1.2)
assert np.isclose(by_history[5], 3.0867, atol=1e-4)

# %% [markdown]
# ## The best policy, by elimination over beliefs (Section 2)
#
# The factor over the history has $4^T$ entries, but it depends on the history
# only through the four counts. So the graph is rewritten with one variable
# $\nu_t$ per pull, the belief before pull $t$, which takes one value for each
# set of counts that $t$ pulls can lead to. Its factors:
#
# - the outcome factor $p(y_t = 1 \mid \nu_t, a_t) = \bar\theta_{a_t}$,
# - the reward factor $(1, y_t)$,
# - the factor that says which belief comes next: 1 if $\nu_{t+1}$ is $\nu_t$
#   with one count increased, as $a_t$ and $y_t$ say, and 0 otherwise.
#
# The first pull has a single belief, so it needs no belief variable.


# %%
@lru_cache(maxsize=None)
def beliefs_after(start, pulls):
    """The beliefs that a given number of pulls can lead to."""
    return [(start[0] + i, start[1] + j, start[2] + k,
             start[3] + pulls - i - j - k)
            for i in range(pulls + 1) for j in range(pulls + 1 - i)
            for k in range(pulls + 1 - i - j)]


@lru_cache(maxsize=None)
def successors(start, t):
    """Index of the next belief, for each belief, arm and outcome of pull t."""
    index = {belief: i for i, belief in enumerate(beliefs_after(start, t + 1))}
    return np.array([[[index[after(belief, pulled, success)]
                       for success in (0, 1)] for pulled in (U, K)]
                     for belief in beliefs_after(start, t)])


def pull_factors(start, pulls, t, policy=None):
    """Keys and factors of pull t: outcome, reward and, if given, policy."""
    beliefs = beliefs_after(start, t)
    keys = ([(B(t), len(beliefs))] if t > 0 else []) + [pull(t), outcome(t)]
    success = [[mean(belief, pulled) for pulled in (U, K)]
               for belief in beliefs]
    success = np.array(success)
    factors = [probability(keys, np.stack([1 - success, success], axis=2)),
               value([outcome(t)], [0, 1])]
    if policy is not None:
        factors.append(probability(
            keys[:-1], [policy(belief, pulls - t) for belief in beliefs]))
    return keys, factors


def bandit(start, pulls, policy=None):
    """The factor graph over beliefs, with or without policy factors."""
    graph = SemiringFactorGraph()
    for t in range(pulls):
        keys, factors = pull_factors(start, pulls, t, policy)
        for factor in factors:
            graph.push_back(factor)
        if t + 1 < pulls:  # which belief comes next
            count = len(beliefs_after(start, t + 1))
            graph.push_back(probability(keys + [(B(t + 1), count)],
                                        np.eye(count)[successors(start, t)]))
    return graph


def belief_order(pulls):
    """Backward in time: the outcome, then the pull, then the belief before."""
    return [key for t in reversed(range(pulls))
            for key in (Y(t), A(t), B(t))][:-1]  # there is no B(0)


def best(start, pulls):
    """Expected number of successes of the best policy."""
    return bandit(start, pulls).expectation(
        ordering(*belief_order(pulls)), at_pulls)


def first_pull(start, pulls):
    """Value of each arm at the first pull, if the rest is played best."""
    _, remaining = bandit(start, pulls).eliminatePartialSequential(
        ordering(*belief_order(pulls)[:-1]), at_pulls)
    return table(remaining.product().value(), [pull(0)])


first = {pulls: first_pull(start, pulls) for pulls in [1, 2, 5, 10]}
for pulls, values in first.items():
    print(f"{pulls:2d} pulls: value of U first = {values[U]:.4f}, "
          f"of K first = {values[K]:.4f}, best first pull: "
          f"{'UK'[int(np.argmax(values))]}")
assert np.allclose(first[1], [0.5, 0.6])
assert np.allclose(first[2], [1.1333, 1.2], atol=1e-4)
assert np.allclose(first[5], [3.0867, 3.0381], atol=1e-4)
assert np.allclose(first[10], [6.3693, 6.3138], atol=1e-4)
assert np.isclose(best(start, 10), 6.3693, atol=1e-4)
# The graph over beliefs and the graph over histories agree.
for pulls in [1, 2, 5]:
    assert np.isclose(best(start, pulls), by_history[pulls])

# %% [markdown]
# With one pull left, the known arm is better: 0.6 against 0.5. With 5 or 10
# pulls, the best first pull is the arm with the *lower* expected payoff. The
# value of the information it buys, for 10 pulls:

# %%
print("value of U first minus value of K first:", first[10][U] - first[10][K])
print("expected payoff of the first pull itself:", mean(start, U), "against",
      mean(start, K))
assert np.isclose(first[10][U] - first[10][K], 0.056, atol=5e-4)

# The two beliefs that the first pull of U can lead to.
later = {}
for success in [True, False]:
    belief = after(start, U, success)
    later[success] = best(belief, 9)
    print(f"after {'success' if success else 'failure'}: belief {belief}, "
          f"P(success of U) = {mean(belief, U):.3f}, value of 9 more pulls = "
          f"{later[success]:.4f}")
assert np.isclose(later[True], 6.339, atol=5e-4)
assert np.isclose(later[False], 5.4)
assert np.isclose(0.5 * (1 + later[True]) + 0.5 * later[False], first[10][U])

# %% [markdown]
# ## The size of the graph (Section 2)
#
# The number of distinct beliefs after at most $T$ pulls, against the number
# of distinct histories.

# %%
for pulls in [5, 10, 20]:
    beliefs = sum(len(beliefs_after(start, t)) for t in range(pulls + 1))
    histories = sum(4 ** k for k in range(pulls + 1))
    print(f"{pulls:2d} pulls: {beliefs:6d} beliefs, {histories:.3g} histories")
    assert beliefs == comb(pulls + 4, 4)
assert comb(10 + 4, 4) == 1001

# %% [markdown]
# ## Twenty pulls, one pull at a time (Section 2)
#
# The factor that says which belief comes next is a table with one entry per
# (belief, arm, outcome, next belief). For 20 pulls the last of these tables
# would have 11 million entries, nearly all zero. Eliminating $\nu_{t+1}$ from
# its bucket reads the value factor at the one next belief that the table
# allows, so that step is done here by indexing an array. The outcome and the
# pull are eliminated by the module as before.


# %%
def one_pull_at_a_time(start, pulls, policy=None):
    """Value of each arm at the first pull, and the value of the start."""
    rule = maximum if policy is None else SemiringSum.Average()
    V = np.zeros(len(beliefs_after(start, pulls)))  # no pulls left
    for t in reversed(range(pulls)):
        keys, factors = pull_factors(start, pulls, t, policy)
        # Eliminate the next belief, by indexing, and the outcome, by average.
        bucket = factors[0] * factors[1] * value(keys, V[successors(start, t)])
        Q = bucket.sum(ordering(Y(t)))
        # Eliminate the pull, by maximum or by average under the policy.
        choice = Q if policy is None else factors[2] * Q
        V = table(choice.sum(ordering(A(t)), rule).value(), keys[:-2])
    return table(Q.value(), [pull(0)]), float(V)


for pulls in [5, 10]:  # the same numbers as the single elimination
    values, value_of_start = one_pull_at_a_time(start, pulls)
    assert np.allclose(values, first[pulls])
    assert np.isclose(value_of_start, best(start, pulls))

first[20], best_of_20 = one_pull_at_a_time(start, 20)
print(f"20 pulls: value of U first = {first[20][U]:.4f}, "
      f"of K first = {first[20][K]:.4f}")
assert np.allclose(first[20], [12.998, 12.937], atol=5e-4)

# %% [markdown]
# ## Heuristics, each evaluated exactly (Section 3)
#
# A heuristic is a policy: a probability of pulling each arm, as a function of
# the belief. With its policy factors in the graph, its expected number of
# successes is the plain policy evaluation of Chapter 1: every variable is
# eliminated by average.


# %%
def evaluate(policy, start, pulls):
    """Exact expected number of successes of a policy(belief, pulls_left)."""
    if pulls > 10:
        return one_pull_at_a_time(start, pulls, policy)[1]
    return bandit(start, pulls, policy).expectation(
        ordering(*belief_order(pulls)))


def greedy(belief, pulls_left):
    """Pull the arm with the higher expected payoff now."""
    return (1.0, 0.0) if mean(belief, U) >= mean(belief, K) else (0.0, 1.0)


def epsilon_greedy(belief, pulls_left, epsilon=0.1):
    """Greedy, but a random arm with probability epsilon."""
    return tuple((1 - epsilon) * p + epsilon / 2
                 for p in greedy(belief, pulls_left))


def ucb(belief, pulls_left):
    """Pull the arm with the higher optimistic estimate (UCB1)."""
    counts = [belief[0] + belief[1], belief[2] + belief[3]]
    index = [mean(belief, arm) + np.sqrt(2 * np.log(sum(counts)) / counts[arm])
             for arm in (U, K)]
    return (1.0, 0.0) if index[U] >= index[K] else (0.0, 1.0)


grid = np.linspace(0, 1, 4001)[1:-1]


@lru_cache(maxsize=None)
def probability_u_is_better(belief):
    """P(success probability of U > that of K) under the two Betas."""
    density_u = beta_distribution.pdf(grid, belief[0], belief[1])
    below_k = beta_distribution.cdf(grid, belief[2], belief[3])
    return float(np.trapezoid(density_u * below_k, grid))


def thompson(belief, pulls_left):
    """Sample both success probabilities from the belief; pull the larger."""
    p = probability_u_is_better(belief)
    return (p, 1 - p)


policies = {"greedy": greedy, "epsilon-greedy": epsilon_greedy,
            "UCB": ucb, "Thompson sampling": thompson}
results = {}
for pulls in [5, 10, 20]:
    results[pulls] = {name: evaluate(policy, start, pulls)
                      for name, policy in policies.items()}
    results[pulls]["best"] = best(start, pulls) if pulls <= 10 else best_of_20
    print(f"{pulls:2d} pulls: " + ", ".join(
        f"{name} {value:.4f}" for name, value in results[pulls].items()))

for pulls in [5, 10, 20]:
    assert all(value <= results[pulls]["best"] + 1e-12
               for value in results[pulls].values())
assert np.allclose(list(results[5].values()),
                   [3.0, 2.994, 2.5, 2.930, 3.087], atol=5e-4)
assert np.allclose(list(results[10].values()),
                   [6.028, 6.054, 5.523, 6.046, 6.369], atol=5e-4)
assert np.allclose(list(results[20].values()),
                   [12.172, 12.327, 11.919, 12.474, 12.998], atol=5e-4)
assert results[20]["Thompson sampling"] > results[20]["greedy"]
assert results[5]["greedy"] > results[5]["Thompson sampling"]
# The pass of one pull at a time agrees with the single elimination.
assert np.isclose(one_pull_at_a_time(start, 10, thompson)[1],
                  results[10]["Thompson sampling"])

# %% [markdown]
# Thompson sampling at the start: how often it pulls the unknown arm.

# %%
print("P(U is the better arm) at the start:", probability_u_is_better(start))
assert np.isclose(probability_u_is_better(start), 0.4, atol=1e-3)

# %% [markdown]
# ## An exact special case: nothing to learn (Section 4)
#
# If both arms are known almost exactly, exploring cannot pay, and the best
# policy must be the greedy one, with the value of pulling the better arm
# every time.

# %%
known = (5000, 5000, 6000, 4000)  # success probabilities 0.5 and 0.6
print("best:", best(known, 10), " greedy:", evaluate(greedy, known, 10))
assert np.isclose(best(known, 10), 6.0, atol=1e-3)
assert np.isclose(best(known, 10), evaluate(greedy, known, 10))

# A check of the exact evaluation by simulation, with a fixed seed.
rng = np.random.default_rng(0)
episodes, total = 40_000, 0
for _ in range(episodes):
    truth = [rng.beta(1, 1), rng.beta(12, 8)]  # the true success probabilities
    belief = start
    for pulls_left in range(10, 0, -1):
        pulled = U if rng.random() < thompson(belief, pulls_left)[U] else K
        success = rng.random() < truth[pulled]
        total += success
        belief = after(belief, pulled, success)
print("Thompson sampling, simulated:", total / episodes, " exact:",
      results[10]["Thompson sampling"])
assert abs(total / episodes - results[10]["Thompson sampling"]) < 0.05

# %% [markdown]
# ## The upper bound of knowing the truth (Section 2)
#
# An agent that is told both success probabilities would pull the better arm
# every time. This is an integral over the two Beta densities, done on a grid.

# %%
density_u = beta_distribution.pdf(grid, 1, 1)
density_k = beta_distribution.pdf(grid, 12, 8)
cdf_u = beta_distribution.cdf(grid, 1, 1)
cdf_k = beta_distribution.cdf(grid, 12, 8)
# E[max] = integral of x times the density of the maximum.
expected_max = np.trapezoid(grid * (density_u * cdf_k + density_k * cdf_u),
                            grid)
print("expected success probability of the better arm:", expected_max)
print("for 10 pulls:", 10 * expected_max, " best achievable:",
      results[10]["best"])
assert np.isclose(10 * expected_max, 6.85, atol=5e-3)
assert 10 * expected_max > results[10]["best"]

# %% [markdown]
# ## Dual control: an action that buys information (Section 5)
#
# The line with an unknown actuator gain, $x' = x + \theta_p u + w$. The belief
# about the gain is a Gaussian prior factor, and the observed displacement
# $x' - x$ is a measurement factor on the gain with Jacobian $u$. The variance
# of the gain after the move is the inverse of the information of the two
# factors, and it depends on how large the move was.

# %%
prior_variance, sigma_w = 1.0, 0.5


def variance_after(u, displacement=0.0):
    """Variance of the gain after observing the displacement of a move u."""
    graph = GaussianFactorGraph()
    graph.add(JacobianFactor(G(0), np.eye(1), np.zeros(1),
                             noiseModel.Isotropic.Variance(1, prior_variance)))
    graph.add(JacobianFactor(G(0), np.array([[u]]), np.array([displacement]),
                             noiseModel.Isotropic.Variance(1, sigma_w)))
    information, _ = graph.hessian()
    return 1 / information[0, 0]


for u in [0.0, 0.5, 1.0, 2.0]:
    print(f"u = {u:3.1f}: variance of the gain after the move = "
          f"{variance_after(u):.4f}")
assert np.allclose([variance_after(u) for u in [0.0, 0.5, 1.0, 2.0]],
                   [1, 2 / 3, 1 / 3, 1 / 9])
# The variance does not depend on what was observed, only on the move.
assert np.isclose(variance_after(1.0, displacement=3.0), 1 / 3)
