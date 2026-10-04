# %% [markdown]
# # Chapter 2 examples: the semiring family
#
# This notebook runs the examples of
# [Chapter 2](https://thduynguyen.github.io/gtsam/chapter02) with the
# `gtsam/semiring` module. One factor graph, the track of Chapter 1, is
# eliminated with different rules for summing out a variable, and each rule
# answers a different question.

# %%
import numpy as np
from scipy.special import softmax

from gtsam import DecisionTreeFactor, DiscreteFactorGraph, DiscreteValues
from gtsam import Ordering, SemiringDiscreteFactor, SemiringFactorGraph
from gtsam import SemiringRules, SemiringSum
from gtsam.symbol_shorthand import A, S

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track of Chapter 1, as a semiring factor graph
#
# Three cells, two moves. A probability table is lifted to $(p, 0)$ and a
# reward table to $(1, r)$.

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


def track(policies=(policy, policy)):
    """The factor graph of the track; pass policies=None to leave them out."""
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    for t in range(2):
        keys = [state(t), action(t)]
        if policies is not None:
            graph.push_back(probability(keys, policies[t]))
        graph.push_back(probability(keys + [state(t + 1)], dynamics))
        graph.push_back(value(keys, reward))
    graph.push_back(value([state(2)], final))
    return graph


variables = [S(2), A(1), S(1), A(0), S(0)]
backward = ordering(*variables)  # eliminate backward in time


def everywhere(rule):
    """The same rule for every variable."""
    rules = SemiringRules()
    rules.setAll(variables, rule)
    return rules


# %% [markdown]
# ## One graph, five questions (Section 1)
#
# The graph and the elimination order stay the same. Only the rule changes.

# %%
graph = track()

# Sum-product and expectation: the average rule. The root holds the pair
# (total probability, expected return).
root = graph.product().sum(backward)
total_probability, expected_return = root.evaluate(DiscreteValues())
print(f"sum-product   {total_probability:8.4f}")
print(f"expectation   {expected_return:8.4f}")

# Max-sum: the maximum at every variable.
best_return = graph.expectation(backward, everywhere(SemiringSum.Maximum()))
print(f"max-sum       {best_return:8.4f}")

# Tilted: a tilted mean at every variable.
tilted = {}
for kappa in [-20, -2, -0.5, -0.01, 0.01, 0.5, 2, 20]:
    tilted[kappa] = graph.expectation(
        backward, everywhere(SemiringSum.Tilted(kappa)))
    print(f"tilted, kappa = {kappa:6.2f}   {tilted[kappa]:8.4f}")

# Soft maximum: the same rule, written with a temperature eta = 1 / kappa.
soft = graph.expectation(backward, everywhere(SemiringSum.SoftMaximum(2.0)))
print(f"soft maximum, eta = 2   {soft:8.4f}")

assert np.isclose(total_probability, 1.0) and np.isclose(expected_return, 1.4)
assert np.isclose(best_return, 9.0)
assert np.isclose(soft, tilted[0.5])
assert np.allclose([tilted[k] for k in [-2, -0.5, 0.5, 2]],
                   [-0.9265, -0.2768, 5.4251, 7.6490], atol=1e-4)
assert abs(tilted[0.01] - 1.4) < 0.1 and abs(tilted[-0.01] - 1.4) < 0.1
assert abs(tilted[20] - 9.0) < 0.3 and abs(tilted[-20] + 2.0) < 0.3

# %% [markdown]
# A check against all trajectories. The product of all factors is one table
# with the probability and the return of every trajectory.

# %%
joint = graph.product()
keys = [state(0), action(0), state(1), action(1), state(2)]
p = table(joint.probability(), keys).ravel()
returns = table(joint.value(), keys).ravel()[p > 0]
p = p[p > 0]
print("number of possible trajectories:", len(p))
print("sum of probabilities:", p.sum())
print("expected return:", p @ returns)
print("best and worst return:", returns.max(), returns.min())
assert len(p) == 24 and np.isclose(p @ returns, 1.4)
assert returns.max() == 9 and returns.min() == -2
for kappa in [-2, 0.5, 2]:
    assert np.isclose(np.log(p @ np.exp(kappa * returns)) / kappa,
                      tilted[kappa])

# %% [markdown]
# ## Max-sum, step by step (Section 2)
#
# The last move of the track, eliminated by maximum: the last state $s_2$,
# then the last action $a_1$. There is no policy factor.

# %%
maximum = SemiringSum.Maximum()
keys_1 = [state(1), action(1)]

# Step 1: the bucket of s2 holds the dynamics and the final reward.
bucket_s2 = probability(keys_1 + [state(2)], dynamics) * value(
    [state(2)], final)
_, phi_sa = bucket_s2.eliminate(ordering(S(2)), maximum)
best_final = table(phi_sa.value(), keys_1)
average_final = table(bucket_s2.sum(ordering(S(2))).value(), keys_1)
print("new factor on (s1, a1), the best reachable final reward:\n", best_final)
print("the average instead (expectation semiring):\n", average_final)
assert np.allclose(best_final, [[0, 0], [0, 10], [10, 10]])
assert np.allclose(average_final, [[0, 0], [0, 8], [2, 10]])

# Step 2: the bucket of a1 holds the reward of the move and the new factor.
bucket_a1 = value(keys_1, reward) * phi_sa
conditional_a1, phi_s = bucket_a1.eliminate(ordering(A(1)), maximum)
print("bucket of a1:\n", table(bucket_a1.value(), keys_1))
print("new factor on s1:", table(phi_s.value(), [state(1)]))
print("regret of each action:\n", table(conditional_a1.surprise(), keys_1))
kept = table(conditional_a1.greedy(), keys_1).argmax(axis=1)
print("the action kept:", kept, "(0 = Left, 1 = Right)")
assert np.allclose(table(bucket_a1.value(), keys_1), [[0, -1], [0, 9], [10, 9]])
assert np.allclose(table(phi_s.value(), [state(1)]), [0, 9, 10])
assert list(kept) == [L, R, L]

# The correct order for a decision: average over s2, then maximum over a1.
correct_bucket = value(keys_1, reward) * bucket_s2.sum(ordering(S(2)))
conditional, correct = correct_bucket.eliminate(ordering(A(1)), maximum)
correct_kept = table(conditional.greedy(), keys_1).argmax(axis=1)
print("average over s2, then max over a1:",
      table(correct.value(), [state(1)]), " action kept:", correct_kept)
assert np.allclose(table(correct.value(), [state(1)]), [0, 7, 9])
assert list(correct_kept) == [L, R, R]

# %% [markdown]
# The best trajectory has probability 0.08, and a robot that plays the actions
# max-sum keeps does poorly on average. (At the first move, cell 1 is a tie
# between Left and Right for max-sum; Right is the move of the trajectory
# described in the chapter.)


# %%
def one_hot(moves):
    """A policy table that always takes the given move in each cell."""
    return np.eye(2)[np.asarray(moves)]


always_right = one_hot([R, R, R])
print("probability of the best trajectory:",
      prior[1] * dynamics[1, R, 2] * dynamics[2, L, 2])
max_sum_return = track((always_right, one_hot(kept))).expectation()
best_policy_return = track((always_right, one_hot([L, R, R]))).expectation()
print("expected return of max-sum's actions:", max_sum_return)
print("expected return of the best policy:  ", best_policy_return)
assert np.isclose(prior[1] * dynamics[1, R, 2] * dynamics[2, L, 2], 0.08)
assert np.isclose(max_sum_return, 3.3)
assert np.isclose(best_policy_return, 6.1)

# %% [markdown]
# ## Joint optimization is a member of the family (Section 2)
#
# Trajectory optimization, as it is done with GTSAM, maximizes the return plus
# the log-probability over all variables at once. In GTSAM's own terms that is
# max-product on an ordinary factor graph, with each reward $r$ entered as the
# factor $e^{r}$. There is no policy: the actions are free variables.

# %%
joint_graph = DiscreteFactorGraph()
joint_graph.push_back(DecisionTreeFactor([state(0)], prior.tolist()))
for t in range(2):
    keys = [state(t), action(t)]
    joint_graph.push_back(DecisionTreeFactor(
        keys + [state(t + 1)], np.ravel(dynamics).tolist()))
    joint_graph.push_back(DecisionTreeFactor(
        keys, np.exp(np.ravel(reward)).tolist()))
joint_graph.push_back(DecisionTreeFactor([state(2)], np.exp(final).tolist()))

plan = joint_graph.optimize()  # the most probable assignment
score = np.log(joint_graph(plan))  # return plus log-probability
trajectory = [plan[key] for key in [S(0), A(0), S(1), A(1), S(2)]]
print("plan (s0, a0, s1, a1, s2):", trajectory)
print("score: return + log-probability =", round(score, 4))
assert trajectory == [1, R, 2, R, 2]
assert np.isclose(score, 8 + np.log(0.4))

# The trajectory that max-sum prefers scores less once its slip is charged.
slip_score = 9 + np.log(prior[1] * dynamics[1, R, 2] * dynamics[2, L, 2])
print("score of the trajectory with the slip:", round(slip_score, 2))
assert np.isclose(slip_score, 6.47, atol=0.005) and slip_score < score

# %% [markdown]
# ## The tilted mean of a coin (Section 2)
#
# Toss a coin: heads, the return is 10; tails, it is 0. The tilted mean of
# one variable is one elimination with the tilted rule.

# %%
coin = (S(0), 2)
coin_p, coin_v = np.array([0.5, 0.5]), np.array([0.0, 10.0])
coin_factor = SemiringDiscreteFactor(
    DecisionTreeFactor([coin], coin_p.tolist()),
    DecisionTreeFactor([coin], coin_v.tolist()))
for kappa in [-5, -0.5, -0.01, 0.01, 0.5, 5]:
    tilted_mean = coin_factor.sum(
        ordering(S(0)), SemiringSum.Tilted(kappa)).expectation()
    print(f"kappa = {kappa:5.2f}   tilted mean = {tilted_mean:.3f}")
    assert np.isclose(tilted_mean,
                      np.log(coin_p @ np.exp(kappa * coin_v)) / kappa)

# The three steps for kappa = 0.5: stretch, average, undo the stretch.
stretched = np.exp(0.5 * coin_v)
print("stretched:", stretched, " average:", coin_p @ stretched)
assert np.allclose(stretched, [1, 148.4], atol=0.05)
assert np.isclose(coin_p @ stretched, 74.7, atol=0.05)
half = coin_factor.sum(ordering(S(0)), SemiringSum.Tilted(0.5)).expectation()
assert np.isclose(half, 8.627, atol=1e-3)
# The same in the stored form (p, m): the sum is two additions.
coin_m = coin_p * stretched
print("stored (p, m):", [(float(a), round(float(b), 1))
                         for a, b in zip(coin_p, coin_m)],
      " sum:", (float(coin_p.sum()), round(float(coin_m.sum()), 1)))
assert np.allclose(coin_m, [0.5, 74.2], atol=0.05)
assert np.isclose(np.log(coin_m.sum() / coin_p.sum()) / 0.5, 8.63, atol=0.005)

# %% [markdown]
# ## What "smooth" means (Section 2)
#
# Two actions: Left with value 0, Right with value d, equal prior weights.
# The maximum switches abruptly at d = 0; the soft maximum, here with
# temperature 1, moves gradually.

# %%
choice = (A(0), 2)
soft_rule = SemiringSum.SoftMaximum(1.0)
print("    d   max P(R)  soft P(R)  max value  soft value")
smooth = {}
for d in [-1.0, -0.1, 0.0, 0.1, 1.0]:
    two_actions = SemiringDiscreteFactor(
        DecisionTreeFactor([choice], [0.5, 0.5]),
        DecisionTreeFactor([choice], [0.0, d]))
    conditional, soft_value = two_actions.eliminate(ordering(A(0)), soft_rule)
    soft_right = table(conditional.tilted(1.0), [choice])[R]
    hard_value = two_actions.sum(ordering(A(0)), maximum).expectation()
    hard_right = "tie" if d == 0 else str(int(d > 0))
    smooth[d] = (soft_right, soft_value.expectation())
    print(f"{d:5.1f}   {hard_right:>5}     {soft_right:.3f}     "
          f"{hard_value:5.1f}      {smooth[d][1]:.3f}")
    # The gap to the maximum is at most eta * log(number of actions).
    assert hard_value - np.log(2) <= smooth[d][1] <= hard_value
assert np.allclose([smooth[d][0] for d in [-1.0, -0.1, 0.0, 0.1, 1.0]],
                   [0.269, 0.475, 0.5, 0.525, 0.731], atol=5e-4)
assert np.allclose([smooth[d][1] for d in [-1.0, -0.1, 0.0, 0.1, 1.0]],
                   [-0.380, -0.049, 0.0, 0.051, 0.620], atol=5e-4)

# %% [markdown]
# ## Conditionals and the normalization invariant (Section 5)
#
# Take the bucket of the last action $a_1$ under the coin-flip policy. Its
# product is $(\pi(a \mid s), Q_1(s, a))$. Eliminating the action with a rule
# gives a conditional, and adding that conditional over the action by the same
# rule gives the *one* of the semiring, $(1, 0)$, in every row.

# %%
bucket = probability(keys_1, policy) * correct_bucket
print("Q1 =\n", table(bucket.value(), keys_1))
assert np.allclose(table(bucket.value(), keys_1), [[0, -1], [0, 7], [2, 9]])

names = {"average": SemiringSum.Average(), "maximum": SemiringSum.Maximum(),
         "tilted, kappa = 1": SemiringSum.Tilted(1.0)}
surprises = {}
for name, rule in names.items():
    conditional, new_factor = bucket.eliminate(ordering(A(1)), rule)
    surprises[name] = table(conditional.surprise(), keys_1)
    print(f"\n{name}: value of the new factor",
          table(new_factor.value(), [state(1)]))
    print("value channel of the conditional:\n", surprises[name])
    # The invariant: summing the conditional by its own rule gives (1, 0).
    one = conditional.sum(ordering(A(1)), rule)
    assert np.allclose(table(one.probability(), [state(1)]), 1.0)
    assert np.allclose(table(one.value(), [state(1)]), 0.0)

assert np.allclose(surprises["average"], [[0.5, -0.5], [-3.5, 3.5], [-3.5, 3.5]])
assert np.allclose(surprises["maximum"], [[0, -1], [-7, 0], [-7, 0]])
assert np.allclose(surprises["tilted, kappa = 1"],
                   [[0.38, -0.62], [-6.31, 0.69], [-6.31, 0.69]], atol=5e-3)

# The tilted conditional is a normalized policy: the softmax of Q.
conditional, _ = bucket.eliminate(ordering(A(1)), SemiringSum.Tilted(1.0))
tilted_policy = table(conditional.tilted(1.0), keys_1)
print("\ntilted policy pi * exp(kappa * soft advantage) =\n", tilted_policy)
assert np.allclose(tilted_policy.sum(axis=1), 1)
assert np.allclose(tilted_policy, softmax(table(bucket.value(), keys_1), axis=1))
assert np.allclose(tilted_policy,
                   [[0.7311, 0.2689], [0.0009, 0.9991], [0.0009, 0.9991]],
                   atol=1e-4)

# %% [markdown]
# ## The log-dual form (Section 6)
#
# This last example is about how numbers are stored, not about elimination,
# so it is plain arithmetic. In the form $(\ell, v)$, with $\ell = \log p$,
# multiplication adds both channels, and a product of many small probabilities
# stays finite.

# %%
from scipy.special import logsumexp


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
total, log_dual_value = log_dual_plus(l_product, v_product, axis=0)
print("(l, v): log p =", total, " value =", log_dual_value)
assert np.isclose(log_dual_value, 400.0)
