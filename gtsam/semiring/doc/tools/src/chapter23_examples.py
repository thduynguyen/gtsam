# %% [markdown]
# # Chapter 23 examples: inverse problems
#
# This notebook runs the examples of
# [Chapter 23](https://thduynguyen.github.io/gtsam/chapter23) with the
# `gtsam/semiring` module: maximum-entropy inverse reinforcement learning on
# the track. An expert is observed, and the reward factors are learned so that
# a soft-optimal policy reproduces what the expert does. Stage 1 is an
# elimination with the soft maximum at the actions; only the sampling of
# demonstrations and the arithmetic of the gradient step are plain numpy.

# %%
import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam import SemiringRules, SemiringSum
from gtsam.symbol_shorthand import A, S

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track, with reward factors that have parameters (Section 1)
#
# The reward of a move is $\theta_R$ for moving Right, and the final reward is
# $\theta_0$, $\theta_1$ or $\theta_2$ depending on the last cell. The true
# parameters are those of Chapter 1.

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
base_policy = np.full((3, 2), 0.5)  # the coin flip
eta = 3.0  # temperature of the soft maximum

theta_true = np.array([-1.0, 0.0, 0.0, 10.0])  # theta_R, theta_0..theta_2

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


def rewards(theta):
    """The reward tables r(s, a) and r(s2) for the parameters theta."""
    move_reward = np.zeros((3, 2))
    move_reward[:, R] = theta[0]
    return move_reward, np.asarray(theta[1:], float)


# The factors that do not depend on theta are built once.
prior_factor = probability([state(0)], prior)
dynamics_factors = [
    probability([state(t), action(t), state(t + 1)], dynamics)
    for t in range(2)]
base_factors = [probability([state(t), action(t)], base_policy)
                for t in range(2)]

# %% [markdown]
# ## Stage 1, backward: a soft backward pass (Section 2)
#
# The graph has the coin flip as its policy factors and reward factors that
# depend on $\theta$. The next state is eliminated by average and the action
# by soft maximum. The conditional on each action, tilted by its value
# channel, is the soft-optimal policy (Chapter 2, Section 5).

# %%
backward_order = ordering(S(2), A(1), S(1), A(0), S(0))
soft_actions = SemiringRules()
soft_actions.setAll([A(0), A(1)], SemiringSum.SoftMaximum(eta))


def soft_pass(move_reward, final_reward, rules=soft_actions):
    """The soft-optimal policy of each move, for given reward tables."""
    graph = SemiringFactorGraph()
    graph.push_back(prior_factor)
    for t in range(2):
        graph.push_back(base_factors[t])
        graph.push_back(dynamics_factors[t])
        graph.push_back(value([state(t), action(t)], move_reward))
    graph.push_back(value([state(2)], final_reward))
    bayes_net = graph.eliminateSequential(backward_order, rules)
    # The conditionals of a1 and a0 are at positions 1 and 3.
    return {1: table(bayes_net.at(1).tilted(1 / eta), [state(1), action(1)]),
            0: table(bayes_net.at(3).tilted(1 / eta), [state(0), action(0)])}


def backward(theta):
    """Soft-optimal policies for the reward parameters theta."""
    return soft_pass(*rewards(theta))


expert = backward(theta_true)
for t in range(2):
    print(f"expert policy at move {t}: pi(R | s) = {expert[t][:, R]}")
assert np.allclose(expert[0].sum(axis=1), 1)

# A check of the module against the formulas of the chapter, in numpy.
move_reward, V = rewards(theta_true)
for t in [1, 0]:
    Q = move_reward + dynamics @ V  # average over the next state
    V = eta * np.log((base_policy * np.exp(Q / eta)).sum(axis=1))
    assert np.allclose(expert[t], base_policy * np.exp((Q - V[:, None]) / eta))

# %% [markdown]
# ## Stage 1, forward: expected features (Section 2)
#
# The forward messages $d_t$, by factor operations: multiply the message with
# the policy factor, read the visitation of each state and action, multiply
# with the dynamics and sum out the state and the action. They give the
# expected number of moves Right and the distribution of the last cell.


# %%
def forward(policies):
    """Visitations d_t(s) pi_t(a | s) of each move, and d_2(s)."""
    message, pairs = prior_factor, {}
    for t in range(2):
        keys = [state(t), action(t)]
        pair = message * probability(keys, policies[t])
        pairs[t] = table(pair.probability(), keys)
        message = (pair * dynamics_factors[t]).sum(ordering(S(t), A(t)))
    return pairs, table(message.probability(), [state(2)])


def features(policies):
    """Expected features: [number of moves Right, last cell is 0, 1, 2]."""
    pairs, last = forward(policies)
    return np.concatenate([[pairs[0][:, R].sum() + pairs[1][:, R].sum()],
                           last])


f_expert = features(expert)
print("expected features of the expert:", f_expert)
assert np.allclose(f_expert, [1.591, 0.199, 0.190, 0.611], atol=5e-4)

# %% [markdown]
# ## The gradient is a difference of expected features (Section 3)
#
# The log-likelihood of the expert's choices, per demonstration, and its
# gradient: (expert features - model features) / eta.

# %%
expert_pairs, _ = forward(expert)


def log_likelihood(theta):
    policies = backward(theta)
    return sum((expert_pairs[t] * np.log(policies[t])).sum() for t in range(2))


def gradient(theta, f_data):
    return (f_data - features(backward(theta))) / eta


theta = np.array([0.5, 1.0, -1.0, 2.0])  # an arbitrary point
h = 1e-6
numeric = np.array([
    (log_likelihood(theta + h * e) - log_likelihood(theta - h * e)) / (2 * h)
    for e in np.eye(4)])
print("gradient by feature matching:  ", gradient(theta, f_expert))
print("gradient by finite differences:", numeric)
assert np.allclose(gradient(theta, f_expert), numeric, atol=1e-6)
# At the true parameters the model is the expert: the gradient vanishes.
assert np.allclose(gradient(theta_true, f_expert), 0)

# %% [markdown]
# ## Stage 2: gradient ascent on the reward parameters (Section 3)
#
# Each iteration is one soft elimination, one forward pass and one step.


# %%
def fit(f_data, iterations=20000, step=2.0):
    theta = np.zeros(4)
    start = log_likelihood(theta)
    for k in range(iterations):
        theta = theta + step * eta * gradient(theta, f_data)
    return theta, start


theta_fit, start = fit(f_expert)
fitted = backward(theta_fit)
print("log-likelihood: start", start, " end", log_likelihood(theta_fit),
      " best possible", log_likelihood(theta_true))
print("recovered theta:", theta_fit)
print("true theta:     ", theta_true)
print("features of the fitted model:", features(fitted))
print("features of the expert:      ", f_expert)
for t in range(2):
    print(f"move {t}: fitted pi(R | s) = {fitted[t][:, R]}, "
          f"expert = {expert[t][:, R]}")
assert np.allclose(features(fitted), f_expert, atol=1e-6)
assert np.allclose(theta_fit, [-1, -3.333, -3.333, 6.667], atol=1e-3)

# %% [markdown]
# ## What is identified, and what is not (Section 4)
#
# The recovered parameters differ from the true ones by a constant added to
# the three final rewards. Differences between final rewards are recovered.

# %%
shift = theta_fit[1:] - theta_true[1:]
print("recovered minus true, final rewards:", shift)
print("recovered theta_R:", theta_fit[0])
print("recovered theta_2 - theta_0:", theta_fit[3] - theta_fit[1])
print("recovered theta_1 - theta_0:", theta_fit[2] - theta_fit[1])
assert np.allclose(shift, shift[0], atol=1e-3)
assert np.isclose(theta_fit[0], -1.0, atol=1e-3)
assert np.isclose(theta_fit[3] - theta_fit[1], 10.0, atol=1e-2)
# The constant direction never moves: the three indicators sum to one.
assert np.isclose(theta_fit[1:].sum(), 0.0, atol=1e-9)

# %% [markdown]
# A richer ambiguity: *shaping*. Adding $\Phi(s') - \Phi(s)$ to the reward of
# every move, for any function $\Phi$ of the state, and subtracting $\Phi$
# from the final reward, leaves the soft-optimal policy unchanged.

# %%
potential = np.array([3.0, -2.0, 5.0])  # an arbitrary Phi(s)
move_reward, final_reward = rewards(theta_true)
shaped_move = move_reward + dynamics @ potential - potential[:, None]
shaped_final = final_reward - potential
shaped = soft_pass(shaped_move, shaped_final)
for t in range(2):
    assert np.allclose(shaped[t], expert[t])
print("the shaped rewards give the same policy at both moves")

# %% [markdown]
# ## From a finite set of demonstrations (Section 5)
#
# 500 sampled demonstrations, with a fixed seed. The features are counted, and
# the same ascent is run.

# %%
rng = np.random.default_rng(0)
demonstrations = 500
counts = np.zeros(4)
for _ in range(demonstrations):
    s = rng.choice(3, p=prior)
    for t in range(2):
        a = rng.choice(2, p=expert[t][s])
        counts[0] += a == R
        s = rng.choice(3, p=dynamics[s, a])
    counts[1 + s] += 1
f_counted = counts / demonstrations
print("counted features: ", f_counted)
print("expected features:", f_expert)

theta_sampled, _ = fit(f_counted)
sampled = backward(theta_sampled)
print("theta from 500 demonstrations:", theta_sampled)
for t in range(2):
    print(f"move {t}: pi(R | s) = {sampled[t][:, R]}, "
          f"expert = {expert[t][:, R]}")
assert np.allclose(features(sampled), f_counted, atol=1e-5)
assert np.allclose(theta_sampled, [-2.38, -4.49, -3.60, 8.08], atol=5e-3)

# %% [markdown]
# ## Keeping the dynamics fixed (Section 6)
#
# If the soft maximum is applied to the next state as well, the model believes
# the robot can count on lucky slips. The value of moving Left from the
# charger at the last move, from the bucket of the last state under each rule:

# %%
move_reward, final_reward = rewards(theta_true)
keys = [state(1), action(1)]
bucket = dynamics_factors[1] * value([state(2)], final_reward)
average = table(bucket.sum(ordering(S(2))).value(), keys)[2, L]
tilted = table(
    bucket.sum(ordering(S(2)), SemiringSum.SoftMaximum(eta)).value(),
    keys)[2, L]
print("average over the next state:     ", average)
print("soft maximum over the next state:", tilted)
assert np.isclose(average, 2.0) and np.isclose(tilted, 5.57, atol=5e-3)
assert np.isclose(tilted,
                  eta * np.log(dynamics[2, L] @ np.exp(final_reward / eta)))
