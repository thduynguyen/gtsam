# %% [markdown]
# # Chapter 24 examples: exploration and dual control
#
# This notebook runs the examples of
# [Chapter 24](https://thduynguyen.github.io/gtsam/chapter24): a bandit with
# two arms whose success probabilities are unknown. The best way to explore is
# computed exactly, by elimination over beliefs, and compared with the usual
# heuristics, each evaluated exactly as well.

# %%
from functools import lru_cache
from math import comb

import numpy as np
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
# ## The best policy, by elimination over beliefs (Section 2)
#
# Backward in time: the outcome of a pull is eliminated by average, with the
# predicted probability of success, and the choice of arm by maximum.


# %%
def pull(belief, pulls_left, arm, value):
    """Value of pulling an arm now, then continuing with `value`."""
    p = mean(belief, arm)
    return (p * (1 + value(after(belief, arm, True), pulls_left - 1)) +
            (1 - p) * value(after(belief, arm, False), pulls_left - 1))


@lru_cache(maxsize=None)
def best(belief, pulls_left):
    """Expected number of successes of the best policy."""
    if pulls_left == 0:
        return 0.0
    return max(pull(belief, pulls_left, arm, best) for arm in (U, K))


for pulls in [1, 2, 5, 10, 20]:
    values = [pull(start, pulls, arm, best) for arm in (U, K)]
    print(f"{pulls:2d} pulls: value of U first = {values[U]:.4f}, "
          f"of K first = {values[K]:.4f}, best first pull: "
          f"{'UK'[int(np.argmax(values))]}")
assert np.isclose(pull(start, 1, U, best), 0.5)
assert np.isclose(pull(start, 1, K, best), 0.6)
assert np.isclose(best(start, 10), 6.3693, atol=1e-4)

# %% [markdown]
# With one pull left, the known arm is better: 0.6 against 0.5. With two
# pulls or more, the best first pull is the arm with the *lower* expected
# payoff. The value of the information it buys, for 10 pulls:

# %%
first = [pull(start, 10, arm, best) for arm in (U, K)]
print("value of U first minus value of K first:", first[U] - first[K])
print("expected payoff of the first pull itself:", mean(start, U), "against",
      mean(start, K))

# The two beliefs that the first pull of U can lead to.
for success in [True, False]:
    belief = after(start, U, success)
    print(f"after {'success' if success else 'failure'}: belief {belief}, "
          f"P(success of U) = {mean(belief, U):.3f}, value of 9 more pulls = "
          f"{best(belief, 9):.4f}")

# %% [markdown]
# ## The size of the graph (Section 2)
#
# The number of distinct beliefs after at most $T$ pulls, against the number
# of distinct histories.

# %%
for pulls in [5, 10, 20]:
    beliefs = comb(pulls + 4, 4)
    histories = sum(4 ** k for k in range(pulls + 1))
    print(f"{pulls:2d} pulls: {beliefs:6d} beliefs, {histories:.3g} histories")
assert comb(10 + 4, 4) == 1001

# %% [markdown]
# ## Heuristics, each evaluated exactly (Section 3)
#
# A heuristic is a policy: a probability of pulling each arm, as a function of
# the belief. With the policy fixed, its expected number of successes is the
# plain policy evaluation of Chapter 1 on the same tree of beliefs.


# %%
def evaluate(policy):
    """Exact expected number of successes of a policy(belief, pulls_left)."""

    @lru_cache(maxsize=None)
    def value(belief, pulls_left):
        if pulls_left == 0:
            return 0.0
        probabilities = policy(belief, pulls_left)
        return sum(probabilities[arm] * pull(belief, pulls_left, arm, value)
                   for arm in (U, K) if probabilities[arm] > 0)

    return value


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
    results[pulls] = {name: evaluate(policy)(start, pulls)
                      for name, policy in policies.items()}
    results[pulls]["best"] = best(start, pulls)
    print(f"{pulls:2d} pulls: " + ", ".join(
        f"{name} {value:.4f}" for name, value in results[pulls].items()))

for pulls in [5, 10, 20]:
    assert all(value <= results[pulls]["best"] + 1e-12
               for value in results[pulls].values())
assert np.isclose(results[5]["greedy"], 3.0)
assert results[20]["Thompson sampling"] > results[20]["greedy"]
assert results[5]["greedy"] > results[5]["Thompson sampling"]

# %% [markdown]
# Thompson sampling at the start: how often it pulls the unknown arm.

# %%
print("P(U is the better arm) at the start:", probability_u_is_better(start))

# %% [markdown]
# ## An exact special case: nothing to learn (Section 4)
#
# If both arms are known almost exactly, exploring cannot pay, and the best
# policy must be the greedy one, with the value of pulling the better arm
# every time.

# %%
known = (5000, 5000, 6000, 4000)  # success probabilities 0.5 and 0.6
print("best:", best(known, 10), " greedy:", evaluate(greedy)(known, 10))
assert np.isclose(best(known, 10), 6.0, atol=1e-3)
assert np.isclose(best(known, 10), evaluate(greedy)(known, 10))

# A check of the exact evaluation by simulation, with a fixed seed.
rng = np.random.default_rng(0)
episodes, total = 40_000, 0
for _ in range(episodes):
    truth = [rng.beta(1, 1), rng.beta(12, 8)]  # the true success probabilities
    belief = start
    for pulls_left in range(10, 0, -1):
        arm = U if rng.random() < thompson(belief, pulls_left)[U] else K
        success = rng.random() < truth[arm]
        total += success
        belief = after(belief, arm, success)
print("Thompson sampling, simulated:", total / episodes, " exact:",
      results[10]["Thompson sampling"])
assert abs(total / episodes - results[10]["Thompson sampling"]) < 0.05

# %% [markdown]
# ## The upper bound of knowing the truth (Section 2)
#
# An agent that is told both success probabilities would pull the better arm
# every time.

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
assert 10 * expected_max > results[10]["best"]

# %% [markdown]
# ## Dual control: an action that buys information (Section 5)
#
# The line with an unknown actuator gain, $x' = x + \theta_p u + w$. After one
# move, the variance of the belief about the gain depends on how large the
# move was.

# %%
prior_variance, sigma_w = 1.0, 0.5
for u in [0.0, 0.5, 1.0, 2.0]:
    posterior_variance = 1 / (1 / prior_variance + u ** 2 / sigma_w)
    print(f"u = {u:3.1f}: variance of the gain after the move = "
          f"{posterior_variance:.4f}")
assert np.isclose(1 / (1 / prior_variance + 1.0 / sigma_w), 1 / 3)
