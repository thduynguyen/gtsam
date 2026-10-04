# %% [markdown]
# # Chapter 12 examples: bootstrapped messages
#
# This notebook runs the examples of
# [Chapter 12](https://thduynguyen.github.io/gtsam/chapter12). On the endless
# track of Chapter 3, the backward message $V$ is learned from sampled
# transitions by making it consistent with itself one step later, and compared
# with its exact value.

# %%
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track, as a simulator (Section 1)
#
# The algorithm only calls `step`. The tables are used to compute the exact
# values that the estimates are compared with.

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
policy = np.full((3, 2), 0.5)  # the coin flip


cumulative = dynamics.cumsum(axis=2)


def step(s, a, u):
    """One move of the simulator: the reward and the next cell.

    u is a uniform random number, which decides how the move turns out.
    """
    return reward[s, a], int((u > cumulative[s, a]).sum())


def run(steps, rng, s=0):
    """Follow the coin flip for some steps: arrays of s, a, r, s'."""
    actions = rng.integers(2, size=steps)
    uniforms = rng.random(steps)
    out = np.zeros((steps, 4))
    for i in range(steps):
        r, s_next = step(s, actions[i], uniforms[i])
        out[i] = s, actions[i], r, s_next
        s = s_next
    return out[:, 0].astype(int), out[:, 1].astype(int), out[:, 2], \
        out[:, 3].astype(int)


# The exact backward message of Chapter 3, for comparison.
P_pi = np.einsum("sa,sat->st", policy, dynamics)
r_pi = (policy * reward).sum(axis=1)
V = np.linalg.solve(np.eye(3) - gamma * P_pi, r_pi)
Q = reward + gamma * dynamics @ V
A = Q - V[:, None]
print("exact V =", V)
print("exact A =\n", A)
assert np.allclose(V, [-0.2248, 1.1017, 4.1231], atol=1e-4)

# %% [markdown]
# ## TD(0): local consistency on samples (Section 2)
#
# After each transition $(s, r, s')$ the estimate of $V(s)$ is moved toward
# $r + \gamma \hat V(s')$. The step size of a cell decays with the number of
# visits to it.

# %%
rng = np.random.default_rng(0)
s, a, r, s_next = run(200000, rng)

V_hat = np.zeros(3)
visits = np.zeros(3)
checkpoints = {}
for i in range(len(s)):
    visits[s[i]] += 1
    alpha = 20 / (20 + visits[s[i]])
    residual = r[i] + gamma * V_hat[s_next[i]] - V_hat[s[i]]  # TD residual
    V_hat[s[i]] += alpha * residual
    if i + 1 in [100, 1000, 10000, 100000, 200000]:
        checkpoints[i + 1] = V_hat.copy()
        print(f"after {i + 1:6d} transitions: V_hat = {V_hat}, "
              f"largest error {np.abs(V_hat - V).max():.4f}")
assert np.allclose(V_hat, V, atol=0.05)

# %% [markdown]
# At the exact $V$ the TD residual averages to zero in every cell: this is
# the Bellman equation of Chapter 3, read on samples.

# %%
exact_residual = r + gamma * V[s_next] - V[s]
for cell in range(3):
    print(f"cell {cell}: average TD residual at the exact V = "
          f"{exact_residual[s == cell].mean():+.4f}")
    assert abs(exact_residual[s == cell].mean()) < 0.03

# %% [markdown]
# ## Returns over n steps, and their mixture (Section 3)
#
# The $n$-step target uses $n$ sampled rewards and then the estimate. For an
# imperfect estimate $\hat V = 0.5\, V$, its bias shrinks like $\gamma^n$ and
# its spread grows with $n$. Start in cell 1.

# %%
V_half = 0.5 * V  # a critic that is half right
error = V_half - V


def paths(start, action, M, length, rng):
    """M sampled paths from (start, action): rewards[M, length], states[M, length + 1]."""
    states = np.zeros((M, length + 1), dtype=int)
    rewards = np.zeros((M, length))
    states[:, 0] = start
    for t in range(length):
        a = rng.integers(2, size=M) if (t > 0 or action is None) else \
            np.full(M, action)
        rewards[:, t] = reward[states[:, t], a]
        u = rng.random(M)
        states[:, t + 1] = (
            u[:, None] > dynamics[states[:, t], a].cumsum(axis=1)).sum(axis=1)
    return rewards, states


rng = np.random.default_rng(1)
rewards, states = paths(1, None, 20000, 60, rng)
discounts = gamma ** np.arange(60)
print("exact V(1) =", V[1])
for n in [1, 2, 5, 10, 60]:
    target = (rewards[:, :n] * discounts[:n]).sum(axis=1) + \
        gamma ** n * V_half[states[:, n]]
    exact_bias = gamma ** n * (np.linalg.matrix_power(P_pi, n) @ error)[1]
    print(f"n = {n:2d}: mean {target.mean():.3f}  bias {target.mean() - V[1]:+.3f}"
          f"  (exact bias {exact_bias:+.3f})  standard deviation {target.std():.3f}")
    if n == 1:
        assert np.isclose(exact_bias, -0.8008, atol=1e-3)

# %% [markdown]
# ## GAE: a mixture of all n (Section 3)
#
# The advantage estimate $\hat A_t = \sum_l (\gamma\lambda)^l\, \hat\delta_{t+l}$
# for the pair $(s, a) = (1, R)$, with the same half-right critic. What matters
# for the gradient is the gap between the two actions; its bias is computed
# exactly, and the spread is measured on samples.

# %%
def gap_bias(lam, cell):
    """Exact bias of A_hat(cell, R) - A_hat(cell, L) for the critic V_half."""
    mix = np.eye(3) - lam * np.linalg.solve(
        np.eye(3) - gamma * lam * P_pi, (np.eye(3) - gamma * P_pi))
    return gamma * (dynamics[cell, R] - dynamics[cell, L]) @ mix @ error


rng = np.random.default_rng(2)
results = {}
for action in [L, R]:
    rewards, states = paths(1, action, 20000, 80, rng)
    residuals = rewards + gamma * V_half[states[:, 1:]] - V_half[states[:, :-1]]
    results[action] = residuals
print("exact advantages in cell 1:", A[1], " gap", A[1, R] - A[1, L])
gae_table = {}
for lam in [0.0, 0.5, 0.9, 0.95, 1.0]:
    weights = (gamma * lam) ** np.arange(80)
    estimate = {action: (results[action] * weights).sum(axis=1)
                for action in [L, R]}
    sampled_gap = estimate[R].mean() - estimate[L].mean()
    gae_table[lam] = (gap_bias(lam, 1), estimate[R].std())
    print(f"lambda = {lam:4.2f}: gap {sampled_gap:.3f}  "
          f"exact bias of the gap {gap_bias(lam, 1):+.3f}  "
          f"standard deviation of A_hat(1, R) {estimate[R].std():.3f}")
    assert abs(sampled_gap - (A[1, R] - A[1, L]) - gap_bias(lam, 1)) < 0.15
assert abs(gap_bias(1.0, 1)) < 1e-12
assert gae_table[0.0][1] < gae_table[1.0][1]

# %% [markdown]
# ## TD(lambda) with traces (Section 3)
#
# The same mixture, computed online: every cell keeps a trace of how recently
# it was visited, and each TD residual updates all cells in proportion to
# their traces.

# %%
rng = np.random.default_rng(3)
s, a, r, s_next = run(200000, rng)
lam = 0.8
V_trace = np.zeros(3)
trace = np.zeros(3)
for i in range(len(s)):
    trace *= gamma * lam
    trace[s[i]] += 1
    residual = r[i] + gamma * V_trace[s_next[i]] - V_trace[s[i]]
    V_trace += 6 / (20 + i / 3) * residual * trace
print("TD(0.8) with traces: V_hat =", V_trace,
      " largest error", np.abs(V_trace - V).max().round(4))
assert np.allclose(V_trace, V, atol=0.1)

# %% [markdown]
# ## A value function with parameters (Section 4)
#
# Let cells 0 and 1 share one value: $\hat V = (\theta_{V,0}, \theta_{V,0},
# \theta_{V,1})$, two numbers for three cells. The exact $V$ cannot be
# represented. TD settles at a fixed point of its own, close to but different
# from the best fit.

# %%
features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # one row per cell
# The coin flip visits the three cells equally often in the long run.
stationary = np.linalg.matrix_power(P_pi, 200)[0]
print("long-run visit frequencies:", stationary)
D = np.diag(stationary)

best_fit = np.linalg.solve(features.T @ D @ features, features.T @ D @ V)
td_fixed = np.linalg.solve(
    features.T @ D @ (np.eye(3) - gamma * P_pi) @ features,
    features.T @ D @ r_pi)
print("best fit of V:      theta_V =", best_fit, " V_hat =", features @ best_fit)
print("TD fixed point:     theta_V =", td_fixed, " V_hat =", features @ td_fixed)
print("exact V:                                         ", V)

# Semi-gradient TD(0) on samples reaches the TD fixed point.
rng = np.random.default_rng(4)
s, a, r, s_next = run(200000, rng)
theta_V = np.zeros(2)
for i in range(len(s)):
    residual = (r[i] + gamma * features[s_next[i]] @ theta_V
                - features[s[i]] @ theta_V)
    theta_V += 10 / (20 + i / 2) * residual * features[s[i]]
print("semi-gradient TD(0): theta_V =", theta_V)
assert np.allclose(theta_V, td_fixed, atol=0.1)
assert np.allclose(td_fixed, [0.625, 3.75]) and np.allclose(
    best_fit, [0.4385, 4.1231], atol=1e-4)
assert np.allclose(stationary, 1 / 3)

# %% [markdown]
# ## The deadly triad (Section 6)
#
# Two states. State 1 always leads to state 2, and state 2 to itself. All
# rewards are zero, so the true value is zero. The estimate has one parameter:
# $\hat V(1) = \theta_V$ and $\hat V(2) = 2\, \theta_V$. The expected update is
# applied 200 times from $\theta_V = 1$, with step size 0.5, under four
# settings.

# %%
feature = np.array([1.0, 2.0])
next_state = np.array([1, 1])  # both states lead to state 2 (index 1)


def expected_updates(weights, bootstrap, tabular, alpha=0.5, sweeps=200):
    """Apply the expected TD update repeatedly; return the final estimate."""
    estimate = np.array([1.0, 2.0]) if tabular else np.array([1.0])
    for _ in range(sweeps):
        value = estimate if tabular else feature * estimate[0]
        target = gamma * value[next_state] if bootstrap else np.zeros(2)
        residual = target - value
        if tabular:
            estimate = estimate + alpha * weights * residual
        else:
            estimate = estimate + alpha * (weights * residual) @ feature
    return estimate if tabular else feature * estimate[0]


on_policy = np.array([0.0, 1.0])  # in the long run the chain sits in state 2
uniform = np.array([0.5, 0.5])  # updates spread evenly: off-policy
settings = [
    ("all three: bootstrap, shared parameter, off-policy weights",
     dict(weights=uniform, bootstrap=True, tabular=False)),
    ("no bootstrapping: full returns as targets",
     dict(weights=uniform, bootstrap=False, tabular=False)),
    ("no shared parameter: a table",
     dict(weights=uniform, bootstrap=True, tabular=True)),
    ("on-policy weights",
     dict(weights=on_policy, bootstrap=True, tabular=False)),
]
triad = {}
for name, options in settings:
    triad[name] = expected_updates(**options)
    print(f"{name:60s} V_hat = {triad[name]}")
assert np.abs(triad[settings[0][0]]).max() > 1e6  # diverges
for name, _ in settings[1:]:
    assert np.abs(triad[name]).max() < 0.05  # shrinks toward zero

# The growth factor of the parameter per expected update, for mixed weights:
# 1 + alpha * (d1 * (2 gamma - 1) + 4 d2 * (gamma - 1)), with alpha = 0.5.
for d1 in [0.0, 1 / 3, 0.5, 1.0]:
    d2 = 1 - d1
    print(f"weight of state 1 = {d1:.3f}: growth factor per update = "
          f"{1 + 0.5 * (d1 * (2 * gamma - 1) + 4 * d2 * (gamma - 1)):.4f}")
