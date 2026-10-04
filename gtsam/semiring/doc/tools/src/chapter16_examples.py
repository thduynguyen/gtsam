# %% [markdown]
# # Chapter 16 examples: off-policy actor-critic
#
# This notebook runs the examples of
# [Chapter 16](https://thduynguyen.github.io/gtsam/chapter16): the
# deterministic policy gradient on the line of Chapter 1, checked against
# the Riccati solution and against finite differences, and a small version
# of DDPG and TD3 with a linear policy and a quadratic critic.

# %%
import numpy as np
from gtsam import HessianFactor, JacobianFactor
from gtsam import SemiringFactorGraph, SemiringGaussianFactor, noiseModel
from gtsam.symbol_shorthand import U, X

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The two-move line with a deterministic policy (Sections 1 and 2)
#
# The policy is $u_t = -K_t x_t$, optionally with Gaussian noise of variance
# $\Sigma_e$. Stage 1 is exact: a backward pass for the quadratic values and a
# forward pass for the second moment of the state.

# %%
SIGMA_W, MU_0, SIGMA_0 = 0.5, 2.0, 1.0  # the line of Chapter 1


def stage1(gains, sigma_e=0.0):
    """Exact messages of the policy u_t = -K_t x_t + e on the two-move line.

    Returns J, the blocks (H_uu, H_ux) of Q_t, and the forward messages
    m_t = E[x_t^2].
    """
    P, beta = 1.0, 0.0  # V_2(x) = -x^2
    H = {}
    for t in [1, 0]:
        K = gains[t]
        H_uu, H_ux, H_xx = 1 + P, P, 1 + P  # blocks of Q_t(x, u)
        H[t] = (H_uu, H_ux)
        beta = beta + P * SIGMA_W + H_uu * sigma_e
        P = H_xx - 2 * H_ux * K + H_uu * K ** 2  # V_t(x) = -(P x^2 + beta)
    m = {0: MU_0 ** 2 + SIGMA_0}
    m[1] = (1 - gains[0]) ** 2 * m[0] + sigma_e + SIGMA_W
    J = -(P * m[0] + beta)
    return J, H, m


def deterministic_gradient(gains):
    """dJ/dK_t = E[dmu/dK * dQ_t/du] = 2 (H_ux - H_uu K_t) m_t."""
    _, H, m = stage1(gains)
    return np.array([2 * (H[t][1] - H[t][0] * gains[t]) * m[t]
                     for t in range(2)])


def finite_differences(gains, sigma_e=0.0, h=1e-6):
    gains = np.asarray(gains, float)
    return np.array([
        (stage1(gains + h * e, sigma_e)[0] - stage1(gains - h * e, sigma_e)[0])
        / (2 * h) for e in np.eye(2)])


for gains in [(0.5, 0.5), (0.3, 0.8), (0.6, 0.5)]:
    J, H, m = stage1(gains)
    print(f"K = {gains}: J = {J:.4f}, m = ({m[0]:.3f}, {m[1]:.3f}), "
          f"gradient = {deterministic_gradient(gains)}, "
          f"finite differences = {finite_differences(gains)}")
    assert np.allclose(deterministic_gradient(gains),
                       finite_differences(gains), atol=1e-5)
assert np.isclose(stage1((0.5, 0.5))[0], -9.375)
assert np.isclose(stage1((0.5, 0.5), sigma_e=0.1)[0], -9.825)
assert np.isclose(stage1((0.6, 0.5))[0], -9.25)
assert np.allclose(deterministic_gradient((0.6, 0.5)), 0)
assert np.allclose(deterministic_gradient((0.5, 0.5)), [2.5, 0.0])

# %% [markdown]
# The same expected returns from the module. A deterministic policy is a
# hard constraint $u + K x = 0$: a `JacobianFactor` with a constrained noise
# model.

# %%
I, zero = np.eye(1), np.zeros(1)


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def expected_return(gains):
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([MU_0]),
                             noiseModel.Isotropic.Variance(1, SIGMA_0)))
    for t in range(2):
        graph.push_back(gaussian(U(t), I, X(t), gains[t] * I, zero,
                                 noiseModel.Constrained.All(1)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 noiseModel.Isotropic.Variance(1, SIGMA_W)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph.expectation()


for gains in [(0.5, 0.5), (0.3, 0.8), (0.6, 0.5)]:
    print(f"K = {gains}: graph.expectation() = {expected_return(gains):.4f}")
    assert np.isclose(expected_return(gains), stage1(gains)[0])

# %% [markdown]
# ## The limit of a Gaussian policy (Section 3)
#
# The gradient of $J$ for the policy $u = -K_t x + e$, with $e$ of variance
# $\Sigma_e$, tends to the deterministic policy gradient as $\Sigma_e \to 0$.

# %%
gains = (0.3, 0.8)
for sigma_e in [0.1, 0.01, 0.001, 0.0]:
    print(f"Sigma_e = {sigma_e:5.3f}: dJ/dK = "
          f"{finite_differences(gains, sigma_e)}")
print("deterministic policy gradient:", deterministic_gradient(gains))
assert np.allclose(finite_differences(gains, 0.001),
                   deterministic_gradient(gains), atol=0.01)

# %% [markdown]
# ## The endless line (Section 4)
#
# The same robot with no last move and the discount $\gamma = 0.9$ of
# Chapter 3, and one gain $K$ for every step. The exact reference is the
# fixed point of the discounted Riccati recursion.

# %%
gamma = 0.9

P = 1.0
for _ in range(1000):  # Riccati: eliminate x' by average, u by max
    H_uu, H_ux, H_xx = 1 + gamma * P, gamma * P, 1 + gamma * P
    P = H_xx - H_ux ** 2 / H_uu
K_star = H_ux / H_uu
m_0 = MU_0 ** 2 + SIGMA_0
beta_star = gamma * P * SIGMA_W / (1 - gamma)
J_star = -(P * m_0 + beta_star)
print(f"P* = {P:.4f}, K* = {K_star:.4f}, J* = {J_star:.4f}")
print(f"blocks of Q*: H_uu = {H_uu:.4f}, H_ux = {H_ux:.4f}, "
      f"H_xx = {H_xx:.4f}, constant = {gamma * (P * SIGMA_W + beta_star):.4f}")
assert np.isclose(K_star, 0.5884, atol=1e-4)


def stage1_endless(K):
    """Exact messages of u = -K x on the endless line."""
    P = (1 + K ** 2) / (1 - gamma * (1 - K) ** 2)  # V(x) = -(P x^2 + beta)
    beta = gamma * P * SIGMA_W / (1 - gamma)
    H_uu, H_ux = 1 + gamma * P, gamma * P
    # Discounted second moment: m = sum_t gamma^t E[x_t^2].
    m = (m_0 + gamma * SIGMA_W / (1 - gamma)) / (1 - gamma * (1 - K) ** 2)
    return -(P * m_0 + beta), H_uu, H_ux, m


def gradient_endless(K):
    _, H_uu, H_ux, m = stage1_endless(K)
    return 2 * (H_ux - H_uu * K) * m


for K in [0.3, K_star, 0.8]:
    h = 1e-6
    numeric = (stage1_endless(K + h)[0] - stage1_endless(K - h)[0]) / (2 * h)
    print(f"K = {K:.4f}: J = {stage1_endless(K)[0]:9.4f}, "
          f"gradient = {gradient_endless(K):8.4f}, "
          f"finite differences = {numeric:8.4f}")
    assert np.isclose(gradient_endless(K), numeric, atol=1e-4)
assert abs(gradient_endless(K_star)) < 1e-9

# Stage 2 with exact messages: gradient ascent on K.
K = 0.3
for k in range(200):
    K += 0.002 * gradient_endless(K)
print("gradient ascent with exact messages: K =", K)
assert np.isclose(K, K_star, atol=1e-4)

# %% [markdown]
# ## A small DDPG and a small TD3 (Sections 5 and 6)
#
# The actor is the gain $K$. The critic is a quadratic with four learned
# coefficients, $\hat Q(x, u) = -(\hat H_{xx} x^2 + 2 \hat H_{ux} x u +
# \hat H_{uu} u^2 + \hat H_0)$. Transitions come from the simulator, with
# exploration noise on the action, and are replayed from a buffer.

# %%
def features(x, u):
    return np.stack([x * x, 2 * x * u, u * u, np.ones_like(x)], axis=1)


def learn(seed, twin=False, steps=20_000, batch=64, critic_step=0.002,
          actor_step=0.002, mixing=0.01, explore=0.25, smoothing=0.04):
    """DDPG (twin=False) or TD3 (twin=True) on the endless line."""
    rng = np.random.default_rng(seed)
    K, K_frozen = 0.0, 0.0  # actor and its slowly updated copy
    critics = np.zeros((2 if twin else 1, 4))  # (H_xx, H_ux, H_uu, H_0)
    frozen = critics.copy()
    buffer = np.zeros((steps, 4))
    x = rng.normal(MU_0, np.sqrt(SIGMA_0))
    history = []
    for k in range(steps):
        # Act with exploration noise, and store the transition.
        u = -K * x + rng.normal(0, np.sqrt(explore))
        x_next = x + u + rng.normal(0, np.sqrt(SIGMA_W))  # the simulator
        buffer[k] = x, u, -(x * x + u * u), x_next
        continues = rng.random() < gamma
        x = x_next if continues else rng.normal(MU_0, np.sqrt(SIGMA_0))
        # Stage 1: one gradient step per critic, each on its own minibatch
        # of old transitions.
        for i in range(len(critics)):
            b_x, b_u, b_r, b_next = buffer[rng.integers(k + 1, size=batch)].T
            u_next = -K_frozen * b_next  # the action of the frozen actor
            if twin:  # TD3: smooth the target action, take the smaller critic
                u_next = u_next + np.clip(
                    rng.normal(0, np.sqrt(smoothing), batch), -0.5, 0.5)
            q_next = -(features(b_next, u_next) @ frozen.T).min(axis=1)
            target = b_r + gamma * q_next
            phi = features(b_x, b_u)
            residual = -(phi @ critics[i]) - target
            critics[i] += critic_step * (residual[:, None] * phi).mean(axis=0)
        # Stage 2: follow dQ/du through the policy. TD3 does it less often.
        if not twin or k % 2 == 0:
            H_xx, H_ux, H_uu, H_0 = critics[0]
            K += actor_step * (2 * (H_ux - H_uu * K) * b_x * b_x).mean()
        frozen = (1 - mixing) * frozen + mixing * critics
        K_frozen = (1 - mixing) * K_frozen + mixing * K
        if k % 4000 == 0:
            history.append((k, K))
    return K, critics, history


K_ddpg, critic_ddpg, history = learn(seed=0)
for k, K in history:
    print(f"step {k:5d}: K = {K:.4f}")
print(f"DDPG: K = {K_ddpg:.4f} (exact {K_star:.4f})")
print("DDPG critic (H_xx, H_ux, H_uu, H_0) =", critic_ddpg[0])
assert abs(K_ddpg - K_star) < 0.03

K_td3, critic_td3, _ = learn(seed=0, twin=True)
print(f"TD3:  K = {K_td3:.4f} (exact {K_star:.4f})")
print("TD3 critics =\n", critic_td3)
assert abs(K_td3 - K_star) < 0.03

# %% [markdown]
# ## A maximum over a noisy critic is too high (Section 6)
#
# At $x = 1$, the part of $Q^*$ that depends on the action is
# $-(H_{uu} u^2 + 2 H_{ux} u)$, with its maximum $H_{ux}^2 / H_{uu}$ at
# $u = -K^*$. Perturb the two coefficients by noise, as a learned critic
# would, and let the actor maximize the perturbed critic.

# %%
rng = np.random.default_rng(0)
count, noise = 100_000, 0.3
true_max = H_ux ** 2 / H_uu


def action_part(h_uu, h_ux, u):
    return -(h_uu * u ** 2 + 2 * h_ux * u)


h_uu = H_uu + noise * rng.normal(size=(2, count))  # two independent critics
h_ux = H_ux + noise * rng.normal(size=(2, count))
u_best = -h_ux[0] / h_uu[0]  # the actor maximizes the first critic
claimed = action_part(h_uu[0], h_ux[0], u_best)  # what that critic reports
second = action_part(h_uu[1], h_ux[1], u_best)  # the second critic, same u
achieved = action_part(H_uu, H_ux, u_best)  # the true value of that action
print(f"true maximum:                        {true_max:.4f}")
print(f"one critic, its own maximum:         {claimed.mean():.4f}")
print(f"second critic at the same action:    {second.mean():.4f}")
print(f"smaller of the two critics:          "
      f"{np.minimum(claimed, second).mean():.4f}")
print(f"true value of the chosen action:     {achieved.mean():.4f}")
assert claimed.mean() > true_max > achieved.mean()
assert np.minimum(claimed, second).mean() < claimed.mean()
