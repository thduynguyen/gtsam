# %% [markdown]
# # Chapter 13 examples: actor-critic
#
# This notebook runs the examples of
# [Chapter 13](https://thduynguyen.github.io/gtsam/chapter13). On the endless
# track of Chapter 3, a learned backward message (the critic) and sampled
# forward messages (visited states) drive a gradient step on the policy (the
# actor). Each piece is compared with the exact two-stage loop of Chapter 5.

# %%
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track with a parametric policy (Section 1)
#
# $\pi_\theta(R \mid s) = \sigma(\theta_s)$, one parameter per cell. The
# simulator is `episodes`; the tables are used for the exact reference only.

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
cumulative = dynamics.cumsum(axis=2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def policy_table(theta):
    right = sigmoid(theta)
    return np.stack([1 - right, right], axis=1)


# %% [markdown]
# ## The exact two stages (Section 1)
#
# Stage 1 solves the linear systems of Chapter 3 for the backward message $V$
# and the forward message $d$. The gradient is the formula of Chapter 5.


# %%
def exact_stage1(theta):
    """J, V, Q, A and the discounted visitation d of the policy."""
    policy = policy_table(theta)
    P_pi = np.einsum("sa,sat->st", policy, dynamics)
    r_pi = (policy * reward).sum(axis=1)
    V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
    Q = reward + gamma * dynamics @ V
    d = np.linalg.solve((np.eye(3) - gamma * P_pi).T, prior)
    return prior @ V, V, Q, Q - V[:, None], d


def exact_gradient(theta):
    """sum_s d(s) sum_a dpi(a | s) A(s, a), for one parameter per cell."""
    _, _, _, A, d = exact_stage1(theta)
    slope = sigmoid(theta) * (1 - sigmoid(theta))
    return d * slope * (A[:, R] - A[:, L])


theta = np.zeros(3)
J, V, Q, A, d = exact_stage1(theta)
gradient = exact_gradient(theta)
print("J =", J, " V =", V, " d =", d)
print("A(s, R) - A(s, L) =", A[:, R] - A[:, L])
print("exact gradient =", gradient)

h = 1e-6
numeric = np.array([(exact_stage1(theta + h * e)[0]
                     - exact_stage1(theta - h * e)[0]) / (2 * h)
                    for e in np.eye(3)])
print("finite differences =", numeric)
assert np.allclose(gradient, numeric)
assert np.allclose(gradient, [-0.0427, 1.8506, 0.7990], atol=1e-4)
assert np.isclose(J, 0.4385, atol=1e-4)

# %% [markdown]
# ## Forward message: states visited before the episode ends (Section 2)
#
# An episode starts from $p(s_0)$ and, after every move, continues with
# probability $\gamma$. The expected number of visits to a cell per episode is
# the discounted visitation $d(s)$ of Chapter 3.


# %%
def episodes(theta, M, rng):
    """Run M episodes. Returns the visited transitions s, a, r, s'."""
    right = sigmoid(theta)
    s = (rng.random(M)[:, None] > prior.cumsum()).sum(axis=1)
    batch = []
    while len(s):
        a = (rng.random(len(s)) < right[s]).astype(int)
        s_next = (rng.random(len(s))[:, None] > cumulative[s, a]).sum(axis=1)
        batch.append((s, a, reward[s, a], s_next))
        running = rng.random(len(s)) < gamma  # the episode continues
        s = s_next[running]
    return [np.concatenate(column) for column in zip(*batch)]


rng = np.random.default_rng(0)
M = 20000
s, a, r, s_next = episodes(theta, M, rng)
visits = np.bincount(s, minlength=3) / M
print("transitions per episode:", len(s) / M)
print("visits per episode:", visits)
print("exact d:           ", d)
assert np.allclose(visits, d, atol=0.1)

# %% [markdown]
# ## The gradient from samples, with the exact critic (Section 3)
#
# Each visited transition contributes its score term times its TD residual.
# With the exact $V$ as the critic the estimate is unbiased.


# %%
def sampled_gradient(theta, V_hat, s, a, r, s_next, M):
    """(1 / M) sum over visited transitions of g * TD residual."""
    residual = r + gamma * V_hat[s_next] - V_hat[s]
    g = a - sigmoid(theta[s])  # d log pi(a | s) / d theta_s
    return np.bincount(s, weights=g * residual, minlength=3) / M


estimate = sampled_gradient(theta, V, s, a, r, s_next, M)
print("sampled gradient, exact critic:", estimate)
print("exact gradient:                ", gradient)
assert np.allclose(estimate, gradient, atol=0.06)

# With a wrong critic the estimate is biased. A critic that is zero everywhere
# sees only the reward of the current step.
print("sampled gradient, critic = 0:  ",
      sampled_gradient(theta, np.zeros(3), s, a, r, s_next, M))
biased = d * 0.25 * (reward[:, R] - reward[:, L])
print("its exact mean:                ", biased)

# %% [markdown]
# ## The critic: TD(0) on the same transitions (Section 2)

# %%
def critic_update(V_hat, s, r, s_next, alpha_V, rng):
    """One pass of TD(0) over a batch of transitions, in random order."""
    for i in rng.permutation(len(s)):
        V_hat[s[i]] += alpha_V * (r[i] + gamma * V_hat[s_next[i]] - V_hat[s[i]])
    return V_hat


rng = np.random.default_rng(1)
V_hat = np.zeros(3)
for sweep in range(30):
    batch = episodes(theta, 50, rng)
    V_hat = critic_update(V_hat, batch[0], batch[2], batch[3], 0.02, rng)
print("critic after 1500 episodes:", V_hat, " exact V:", V)
assert np.allclose(V_hat, V, atol=0.3)

# %% [markdown]
# ## Actor-critic: both learned together (Section 5)
#
# Each iteration runs a few episodes with the current policy, updates the
# critic by TD(0), and moves the actor along the sampled gradient. The
# expected return of each iterate is evaluated exactly, for reporting only.

# %%
def actor_critic(iterations, M, alpha_theta, alpha_V, seed):
    rng = np.random.default_rng(seed)
    theta, V_hat = np.zeros(3), np.zeros(3)
    history = []
    for k in range(iterations + 1):
        history.append(exact_stage1(theta)[0])
        s, a, r, s_next = episodes(theta, M, rng)  # stage 1: sample
        V_hat = critic_update(V_hat, s, r, s_next, alpha_V, rng)  # the critic
        theta = theta + alpha_theta * sampled_gradient(  # stage 2: the actor
            theta, V_hat, s, a, r, s_next, M)
    return history, theta, V_hat


def exact_loop(iterations, alpha_theta):
    theta = np.zeros(3)
    history = []
    for k in range(iterations + 1):
        history.append(exact_stage1(theta)[0])
        theta = theta + alpha_theta * exact_gradient(theta)
    return history, theta


history, theta_ac, V_ac = actor_critic(300, 20, 0.2, 0.05, seed=0)
exact_history, theta_exact = exact_loop(300, 0.2)
for k in [0, 1, 5, 10, 20, 50, 100, 200, 300]:
    print(f"iteration {k:3d}: actor-critic J = {history[k]:.4f}   "
          f"exact messages J = {exact_history[k]:.4f}")
print("pi(R | s), actor-critic:  ", sigmoid(theta_ac))
print("pi(R | s), exact messages:", sigmoid(theta_exact))
print("critic:", V_ac, " exact V of the final policy:",
      exact_stage1(theta_ac)[1])
assert history[-1] > 6.2 and exact_history[-1] > 6.2

# Several seeds, to see the spread of the sampled loop.
finals = [actor_critic(300, 20, 0.2, 0.05, seed)[0][-1] for seed in range(10)]
print("final J over 10 seeds: min", min(finals).round(3), " mean",
      np.mean(finals).round(3), " max", max(finals).round(3))
assert min(finals) > 6.0

# %% [markdown]
# ## Compatible features: the critic that is enough (Section 4)
#
# Fit the advantage by a linear function of the score,
# $\hat A(s, a) = \theta_Q^\top g(s, a)$, by least squares with the weights
# $d(s)\, \pi_\theta(a \mid s)$. The fitted weights are the natural gradient,
# and the gradient computed from the fit is exact.

# %%
theta = np.zeros(3)
J, V, Q, A, d = exact_stage1(theta)
policy = policy_table(theta)
# The score of (s, a): a vector with one nonzero entry, in position s.
score = np.zeros((3, 2, 3))
for cell in range(3):
    score[cell, L, cell] = -sigmoid(theta[cell])
    score[cell, R, cell] = 1 - sigmoid(theta[cell])
weights = d[:, None] * policy
fisher = np.einsum("sa,sai,saj->ij", weights, score, score)
theta_Q = np.linalg.solve(fisher, np.einsum("sa,sai,sa->i", weights, score, A))
print("Fisher matrix =\n", fisher)
print("fitted weights theta_Q =", theta_Q)
print("natural gradient       =", np.linalg.solve(fisher, exact_gradient(theta)))
print("gradient from the fit  =", fisher @ theta_Q)
fit = np.einsum("sai,i->sa", score, theta_Q)
print("fitted advantage =\n", fit, "\nexact advantage =\n", A)
assert np.allclose(theta_Q, np.linalg.solve(fisher, exact_gradient(theta)))
assert np.allclose(fisher @ theta_Q, exact_gradient(theta))
assert np.allclose(theta_Q, A[:, R] - A[:, L])
