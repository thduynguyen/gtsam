# %% [markdown]
# # Chapter 17 examples: soft and EM methods
#
# This notebook runs the examples of
# [Chapter 17](https://thduynguyen.github.io/gtsam/chapter17) on the endless
# track of Chapter 3: the tilted conditional as an improved policy, the
# E-step and M-step loop of MPO, REPS and AWR, and the soft fixed point that
# SAC computes.

# %%
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track and exact policy evaluation (Section 1)

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
coin_flip = np.full((3, 2), 0.5)


def evaluate(policy):
    """Exact V, Q and discounted visitation d of a policy (Chapter 3)."""
    P_pi = np.einsum("sa,sat->st", policy, dynamics)
    r_pi = (policy * reward).sum(axis=1)
    V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
    Q = reward + gamma * dynamics @ V
    d = np.linalg.solve((np.eye(3) - gamma * P_pi).T, prior)
    return V, Q, d


V, Q, d = evaluate(coin_flip)
print("coin flip: V =", V, " J =", prior @ V)
print("Q =\n", Q)
assert np.isclose(prior @ V, 0.4385, atol=1e-4)

# %% [markdown]
# ## Stage 1, the E-step: the tilted conditional (Section 2)
#
# Eliminate the action from the bucket $(\pi(a \mid s), Q(s, a))$ with the
# soft maximum at temperature $\eta$. The conditional, reweighted by its own
# surprise, is a new policy $q$.


# %%
def soft_max(policy, Q, eta):
    """eta log sum_a pi(a|s) exp(Q(s,a) / eta), computed stably."""
    return eta * logsumexp(Q / eta, b=policy, axis=1)


def tilt(policy, Q, eta):
    """The tilted policy q = pi exp((Q - soft max) / eta)."""
    return policy * np.exp((Q - soft_max(policy, Q, eta)[:, None]) / eta)


eta = 1.0
soft_V = soft_max(coin_flip, Q, eta)
q = tilt(coin_flip, Q, eta)
print("soft maximum =", soft_V, " average V =", V)
print("soft advantage =\n", Q - soft_V[:, None])
print("tilted policy q =\n", q)
print("rows of q sum to", q.sum(axis=1))
assert np.allclose(q.sum(axis=1), 1)  # the normalization invariant
assert np.all(soft_V >= V)

# The same q from the ordinary advantage: q is proportional to pi exp(A / eta).
A = Q - V[:, None]
proportional = coin_flip * np.exp(A / eta)
assert np.allclose(proportional / proportional.sum(axis=1, keepdims=True), q)

# The tilted policy is the best trade-off between value and staying close.
rng = np.random.default_rng(0)


def kl(q, p):
    """KL(q || p) per state, with 0 log 0 = 0."""
    ratio = np.where(q > 0, q / p, 1.0)
    return (q * np.log(ratio)).sum(axis=1)


best = (q * Q).sum(axis=1) - eta * kl(q, coin_flip)
print("value minus eta * KL at q:", best)
assert np.allclose(best, soft_V)
for _ in range(1000):
    right = rng.random(3)
    other = np.stack([1 - right, right], axis=1)
    assert np.all((other * Q).sum(axis=1) - eta * kl(other, coin_flip)
                  <= soft_V + 1e-12)

# %% [markdown]
# ## The E-step and M-step loop with a free table (Sections 3 and 6)
#
# Stage 1 evaluates the current policy exactly and tilts it. With a table as
# the policy, the M-step fit is exact: the new policy is $q$.


# %%
def em(eta, iterations):
    policy = coin_flip.copy()
    history = []
    for _ in range(iterations):
        V, Q, d = evaluate(policy)  # stage 1: evaluate ...
        history.append(prior @ V)
        policy = tilt(policy, Q, eta)  # ... and tilt; stage 2: policy = q
    return np.array(history), policy


for eta in [1.0, 0.3]:
    history, policy = em(eta, 80)
    print(f"eta = {eta}: J =", history[[0, 1, 2, 3, 5, 10, 20, 79]])
    print("   pi(R | s) =", policy[:, R])
    assert np.all(np.diff(history) >= -1e-12)  # never worse
    assert np.isclose(history[-1], 6.4902, atol=1e-4)

# A tiny temperature is the greedy step of policy iteration (Chapter 4).
history, policy = em(1e-3, 4)
print("eta = 0.001: J =", history)
assert np.allclose(history[:3], [0.4385, 3.7805, 6.4902], atol=1e-4)

# %% [markdown]
# ## The temperature from a bound on the KL divergence (Section 4)
#
# REPS and MPO do not choose $\eta$. They choose how far the tilted policy
# may move, as a bound $D_{\max}$ on its average KL divergence from the
# current policy, and solve a one-dimensional problem for $\eta$.

# %%
V, Q, d = evaluate(coin_flip)
weights = d / d.sum()  # how often each state occurs
D_max = 0.1


def dual(eta):
    return eta * D_max + weights @ soft_max(coin_flip, Q, eta)


solution = minimize_scalar(dual, bounds=(1e-3, 100), method="bounded",
                           options={"xatol": 1e-10})
eta_star = solution.x
q_star = tilt(coin_flip, Q, eta_star)
print("eta =", eta_star)
print("average KL of the tilted policy:", weights @ kl(q_star, coin_flip))
print("q(R | s) =", q_star[:, R])
assert np.isclose(weights @ kl(q_star, coin_flip), D_max, atol=1e-5)
for eta in [0.5, 1.0, 2.0, 5.0]:
    print(f"eta = {eta}: average KL =",
          weights @ kl(tilt(coin_flip, Q, eta), coin_flip))

# %% [markdown]
# ## The M-step with a policy that has shared parameters (Section 3)
#
# The policy $\pi_\theta(R \mid s) = \sigma(\theta_0 + \theta_1 s)$ has two
# parameters for three cells. The M-step fits it to $q$ by weighted maximum
# likelihood, with a few Newton steps.


# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


design = np.stack([np.ones(3), np.arange(3.0)], axis=1)  # rows (1, s)


def table_of(theta):
    right = sigmoid(design @ theta)
    return np.stack([1 - right, right], axis=1)


def m_step(theta, q, weights, steps=20):
    """Maximize sum_s weights(s) sum_a q(a|s) log pi_theta(a|s)."""
    for _ in range(steps):
        right = sigmoid(design @ theta)
        gradient = design.T @ (weights * (q[:, R] - right))
        hessian = (design * (weights * right * (1 - right))[:, None]).T @ design
        theta = theta + np.linalg.solve(hessian + 1e-6 * np.eye(2), gradient)
    return theta


theta = np.zeros(2)  # the coin flip
shared = []
for k in range(30):
    policy = table_of(theta)
    V, Q, d = evaluate(policy)
    shared.append(prior @ V)
    q = tilt(policy, Q, eta=1.0)  # E-step
    theta = m_step(theta, q, d / d.sum())  # M-step
    if k == 0:
        print("first E-step q(R | s) =", q[:, R])
        print("first M-step pi_theta(R | s) =", table_of(theta)[:, R])
shared = np.array(shared)
print("J =", shared[[0, 1, 2, 3, 5, 10, 29]])
print("theta =", theta, " pi_theta(R | s) =", table_of(theta)[:, R])
assert shared[-1] > 6.4 and np.all(np.diff(shared) > -1e-9)

# %% [markdown]
# ## Both steps from samples (Section 4)
#
# Now the dynamics table is only sampled. Each iteration collects 5,000
# transitions with the current policy, learns $\hat Q$ from them by the
# fitted evaluation of Chapter 15, and re-estimates the policy from the
# actions actually taken, each weighted by $e^{\hat A / \eta}$: this is
# advantage-weighted regression.


# %%
def collect(rng, policy, count):
    """Transitions (s, a, r, s'); an episode ends with probability 1 - gamma."""
    data = np.zeros((count, 4), dtype=int)
    s = rng.choice(3, p=prior)
    for i in range(count):
        a = rng.choice(2, p=policy[s])
        s_next = rng.choice(3, p=dynamics[s, a])
        data[i] = s, a, 0, s_next
        s = s_next if rng.random() < gamma else rng.choice(3, p=prior)
    return data


def learn_Q(data, policy, sweeps=200):
    """Fitted evaluation with a table: the average under the policy at s'."""
    s, a, _, s_next = data.T
    Q = np.zeros((3, 2))
    for _ in range(sweeps):
        targets = reward[s, a] + gamma * (policy[s_next] * Q[s_next]).sum(axis=1)
        total, count = np.zeros((3, 2)), np.zeros((3, 2))
        np.add.at(total, (s, a), targets)
        np.add.at(count, (s, a), 1)
        Q = np.where(count > 0, total / np.maximum(count, 1), Q)
    return Q


rng = np.random.default_rng(0)
policy = coin_flip.copy()
sampled = []
for k in range(15):
    sampled.append(prior @ evaluate(policy)[0])  # exact J, for the record
    data = collect(rng, policy, 5000)
    Q_hat = learn_Q(data, policy)  # learned backward message
    A_hat = Q_hat - (policy * Q_hat).sum(axis=1, keepdims=True)
    s, a = data[:, 0], data[:, 1]
    weighted = np.zeros((3, 2))
    np.add.at(weighted, (s, a), np.exp(A_hat[s, a] / 1.0))  # weights exp(A/eta)
    policy = weighted / weighted.sum(axis=1, keepdims=True)  # weighted counts
sampled = np.array(sampled)
print("J =", sampled[[0, 1, 2, 3, 5, 10, 14]])
print("pi(R | s) =", policy[:, R])
assert sampled[-1] > 6.4

# %% [markdown]
# ## SAC: the soft fixed point (Section 5)
#
# Keep the reference policy fixed (the coin flip) and the temperature fixed,
# and iterate the soft maximum to its fixed point: soft value iteration.


# %%
def soft_value_iteration(eta, reference=coin_flip, sweeps=3000):
    V = np.zeros(3)
    for _ in range(sweeps):
        Q = reward + gamma * dynamics @ V  # eliminate s' by average
        V = soft_max(reference, Q, eta)  # eliminate a by soft maximum
    Q = reward + gamma * dynamics @ V
    return V, Q, tilt(reference, Q, eta)


V_star = np.array([5.4194, 7.5610, 10.0])
for eta in [5.0, 1.0, 0.5, 0.1, 0.01, 0.001]:
    V_soft, Q_soft, policy = soft_value_iteration(eta)
    J_true = prior @ evaluate(policy)[0]
    print(f"eta = {eta:5.3f}: V = {V_soft}, pi(R|s) = {policy[:, R]}, "
          f"J of that policy = {J_true:.4f}")
    assert np.allclose(policy.sum(axis=1), 1)
    gap = V_star - V_soft
    assert np.all(gap > -1e-3)
    assert np.all(gap <= eta * np.log(2) / (1 - gamma) + 1e-3)
assert np.allclose(soft_value_iteration(0.001)[0], V_star, atol=0.01)

V_soft, Q_soft, soft_policy = soft_value_iteration(1.0)
assert np.allclose(V_soft, [1.7619, 3.6034, 6.3205], atol=1e-4)

# %% [markdown]
# The same fixed point from the two alternating steps of SAC, done exactly:
# evaluate the soft value of the current policy, then replace the policy by
# the tilted reference.


# %%
def soft_evaluate(policy, eta, reference=coin_flip):
    """V of a policy when every step also pays -eta KL(pi || reference)."""
    P_pi = np.einsum("sa,sat->st", policy, dynamics)
    r_pi = (policy * reward).sum(axis=1) - eta * kl(policy, reference)
    V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
    return V, reward + gamma * dynamics @ V


policy = coin_flip.copy()
for k in range(200):
    V, Q = soft_evaluate(policy, eta=1.0)  # stage 1
    policy = tilt(coin_flip, Q, eta=1.0)  # stage 2
print("soft policy iteration: V =", V, " pi(R | s) =", policy[:, R])
assert np.allclose(V, V_soft, atol=1e-6) and np.allclose(policy, soft_policy)

# SAC's own convention uses the entropy instead of the KL to the coin flip.
# The two differ by the constant eta log 2 per step.
entropy_V = np.zeros(3)
for _ in range(3000):
    entropy_Q = reward + gamma * dynamics @ entropy_V
    entropy_V = 1.0 * logsumexp(entropy_Q / 1.0, axis=1)
print("entropy convention: V =", entropy_V)
assert np.allclose(entropy_V - V_soft, np.log(2) / (1 - gamma))
