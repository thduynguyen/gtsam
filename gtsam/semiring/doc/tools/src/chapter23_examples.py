# %% [markdown]
# # Chapter 23 examples: inverse problems
#
# This notebook runs the examples of
# [Chapter 23](https://thduynguyen.github.io/gtsam/chapter23): maximum-entropy
# inverse reinforcement learning on the track. An expert is observed, and the
# reward factors are learned so that a soft-optimal policy reproduces what the
# expert does.

# %%
import numpy as np

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


def rewards(theta):
    """The reward tables r(s, a) and r(s2) for the parameters theta."""
    move_reward = np.zeros((3, 2))
    move_reward[:, R] = theta[0]
    return move_reward, np.asarray(theta[1:], float)


# %% [markdown]
# ## Stage 1, backward: a soft backward pass (Section 2)
#
# The next state is eliminated by average and the action by soft maximum,
# with the coin flip as the base policy. The conditional on each action is the
# tilted policy of Chapter 2, Section 5.


# %%
def backward(theta):
    """Soft action values and the soft-optimal policy of each move."""
    move_reward, V = rewards(theta)
    policies = {}
    for t in [1, 0]:
        Q = move_reward + dynamics @ V  # average over the next state
        V = eta * np.log((base_policy * np.exp(Q / eta)).sum(axis=1))
        policies[t] = base_policy * np.exp((Q - V[:, None]) / eta)
    return policies


expert = backward(theta_true)
for t in range(2):
    print(f"expert policy at move {t}: pi(R | s) = {expert[t][:, R]}")
assert np.allclose(expert[0].sum(axis=1), 1)

# %% [markdown]
# ## Stage 1, forward: expected features (Section 2)
#
# The forward messages $d_t$ give the expected number of moves Right and the
# distribution of the last cell.


# %%
def forward(policies):
    """Visitations d_t(s) and state-action visitations d_t(s) pi_t(a | s)."""
    d, pairs = prior, {}
    for t in range(2):
        pairs[t] = d[:, None] * policies[t]
        d = np.einsum("sa,sat->t", pairs[t], dynamics)
    return pairs, d  # d is the distribution of the last cell


def features(policies):
    """Expected features: [number of moves Right, last cell is 0, 1, 2]."""
    pairs, last = forward(policies)
    return np.concatenate([[pairs[0][:, R].sum() + pairs[1][:, R].sum()],
                           last])


f_expert = features(expert)
print("expected features of the expert:", f_expert)

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


# %%
def fit(f_data, iterations=20000, step=2.0):
    theta = np.zeros(4)
    history = []
    for k in range(iterations):
        history.append(log_likelihood(theta))
        theta = theta + step * eta * gradient(theta, f_data)
    return theta, history


theta_fit, history = fit(f_expert)
fitted = backward(theta_fit)
print("log-likelihood: start", history[0], " end", log_likelihood(theta_fit),
      " best possible", log_likelihood(theta_true))
print("recovered theta:", theta_fit)
print("true theta:     ", theta_true)
print("features of the fitted model:", features(fitted))
print("features of the expert:      ", f_expert)
for t in range(2):
    print(f"move {t}: fitted pi(R | s) = {fitted[t][:, R]}, "
          f"expert = {expert[t][:, R]}")
assert np.allclose(features(fitted), f_expert, atol=1e-6)

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

V = shaped_final
for t in [1, 0]:
    Q = shaped_move + dynamics @ V
    V = eta * np.log((base_policy * np.exp(Q / eta)).sum(axis=1))
    shaped_policy = base_policy * np.exp((Q - V[:, None]) / eta)
    assert np.allclose(shaped_policy, expert[t])
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

# %% [markdown]
# ## Keeping the dynamics fixed (Section 6)
#
# If the soft maximum is applied to the next state as well, the model believes
# the robot can count on lucky slips. The value of moving Left from the
# charger at the last move:

# %%
move_reward, final_reward = rewards(theta_true)
average = dynamics[2, L] @ final_reward
tilted = eta * np.log(dynamics[2, L] @ np.exp(final_reward / eta))
print("average over the next state:     ", average)
print("soft maximum over the next state:", tilted)
assert np.isclose(average, 2.0) and tilted > 5
