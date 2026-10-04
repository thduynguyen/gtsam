# %% [markdown]
# # Chapter 16 examples: off-policy actor-critic
#
# This notebook runs the examples of
# [Chapter 16](https://thduynguyen.github.io/gtsam/chapter16): the
# deterministic policy gradient on the line of Chapter 1, checked against
# the Riccati solution and against finite differences, and a small version
# of DDPG and TD3 with a linear policy and a quadratic critic. Every exact
# quantity is computed by elimination with the `gtsam/semiring` module; the
# sampling and the learning updates are plain numpy.

# %%
import numpy as np
from gtsam import HessianFactor, JacobianFactor, Ordering
from gtsam import SemiringFactorGraph, SemiringGaussianFactor, SemiringRules
from gtsam import SemiringSum
from gtsam import noiseModel
from gtsam.symbol_shorthand import U, X

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line of Chapter 1 as semiring factors
#
# A density is lifted to $(p, 0)$ and a quadratic penalty to $(1, r)$. A
# deterministic policy is a hard constraint $u + K x = 0$: a
# `JacobianFactor` with a constrained noise model.

# %%
SIGMA_W, MU_0, SIGMA_0 = 0.5, 2.0, 1.0  # the line of Chapter 1
I, zero = np.eye(1), np.zeros(1)
average, maximum = SemiringSum.Average(), SemiringSum.Maximum()


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def square(key):
    """The value z^2 on one variable, as the factor (1, z^2)."""
    return SemiringGaussianFactor.Reward(HessianFactor(key, 2 * I, zero, 0.0))


def ordering(*keys):
    """An ordering of the given keys."""
    result = Ordering()
    for key in keys:
        result.push_back(key)
    return result


def variance(v):
    """A scalar Gaussian noise model with the given variance."""
    return noiseModel.Isotropic.Variance(1, v)


prior = gaussian(X(0), I, np.array([MU_0]), variance(SIGMA_0))


def dynamics(t):
    """x' = x + u + w, the dynamics factor of move t."""
    return gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero, variance(SIGMA_W))


def policy(t, K, sigma_e=0.0):
    """u = -K x + e: a density, or a hard constraint if there is no noise."""
    model = variance(sigma_e) if sigma_e > 0 else noiseModel.Constrained.All(1)
    return gaussian(U(t), I, X(t), K * I, zero, model)


def blocks(bucket, t=0):
    """(H_uu, H_ux, H_xx, constant) of Q = -(H_uu u^2 + 2 H_ux u x + H_xx x^2
    + constant), read from the bucket of the action u_t."""
    Q = bucket.value()  # a HessianFactor: 0.5 z' G z - g' z + 0.5 f
    keys, G = list(Q.keys()), Q.augmentedInformation()
    u, x = keys.index(U(t)), keys.index(X(t))
    return -G[u, u] / 2, -G[u, x] / 2, -G[x, x] / 2, -G[-1, -1] / 2


# %% [markdown]
# ## The two-move line with a deterministic policy (Sections 1 and 2)
#
# The policy is $u_t = -K_t x_t$, optionally with Gaussian noise of variance
# $\Sigma_e$. Stage 1 is exact: a backward pass of eliminations for the
# quadratic values, and the second moment of the state as the forward
# message.


# %%
def second_moment(gains, t, sigma_e=0.0):
    """The forward message m_t = E[x_t^2]: the expectation of the value
    x_t^2 under the prior, the policy and the dynamics up to step t."""
    graph = SemiringFactorGraph()
    graph.push_back(prior)
    for k in range(t):
        graph.push_back(policy(k, gains[k], sigma_e))
        graph.push_back(dynamics(k))
    graph.push_back(square(X(t)))
    return graph.expectation()


def stage1(gains, sigma_e=0.0):
    """Exact messages of the policy u_t = -K_t x_t + e on the two-move line.

    Returns J, the blocks (H_uu, H_ux) of Q_t, and the forward messages
    m_t = E[x_t^2].
    """
    future = penalty(X(2))  # (1, V_2), with V_2(x) = -x^2
    H = {}
    for t in [1, 0]:
        # Eliminate the next state by average; add the rewards: (1, Q_t).
        phi = dynamics(t).multiply(future).sum(ordering(X(t + 1)))
        bucket = penalty(X(t)).multiply(penalty(U(t))).multiply(phi)
        H[t] = blocks(bucket, t)[:2]
        # Eliminate the action by average under the policy: (1, V_t).
        future = policy(t, gains[t], sigma_e).multiply(bucket).sum(
            ordering(U(t)))
    J = prior.multiply(future).expectation()
    m = {t: second_moment(gains, t, sigma_e) for t in range(2)}
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
assert np.allclose(deterministic_gradient((0.6, 0.5)), 0, atol=1e-9)
assert np.allclose(deterministic_gradient((0.5, 0.5)), [2.5, 0.0])
assert np.allclose(deterministic_gradient((0.3, 0.8)), [8.76, -3.54], atol=5e-3)

# %% [markdown]
# The same expected returns in one call, from the whole factor graph,
# eliminated backward in time.

# %%
backward = ordering(X(2), U(1), X(1), U(0), X(0))


def expected_return(gains):
    graph = SemiringFactorGraph()
    graph.push_back(prior)
    for t in range(2):
        graph.push_back(policy(t, gains[t]))
        graph.push_back(dynamics(t))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph.expectation(backward)


for gains in [(0.5, 0.5), (0.3, 0.8), (0.6, 0.5)]:
    print(f"K = {gains}: graph.expectation() = {expected_return(gains):.4f}")
    assert np.isclose(expected_return(gains), stage1(gains)[0])

# The gains at which the gradient vanishes are those of the maximum rule: the
# Riccati recursion of Chapter 6, as one elimination of the same graph without
# its policy factors.
graph = SemiringFactorGraph()
graph.push_back(prior)
for t in range(2):
    graph.push_back(dynamics(t))
    graph.push_back(penalty(X(t)))
    graph.push_back(penalty(U(t)))
graph.push_back(penalty(X(2)))
rules = SemiringRules()
rules.setAll([U(0), U(1)], maximum)
bayes_net = graph.eliminateSequential(backward, rules)
riccati = (bayes_net.at(3).conditional().S()[0, 0],
           bayes_net.at(1).conditional().S()[0, 0])
print("Riccati gains (K0, K1) =", riccati,
      " J* =", graph.expectation(backward, rules))
assert np.allclose(riccati, (0.6, 0.5))
assert np.isclose(graph.expectation(backward, rules), -9.25)

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
# Chapter 3, and one gain $K$ for every step. Every step of the chain is the
# same, so one set of factors is eliminated again and again, with the value of
# the next state scaled by $\gamma$. The exact reference is the fixed point
# with the maximum rule at the action: the discounted Riccati recursion.

# %%
gamma = 0.9
transition = dynamics(0)  # on x0, u0, x1
stage_reward = penalty(X(0)).multiply(penalty(U(0)))  # (1, -(x^2 + u^2))


def quadratic(key, P, beta):
    """The value -(P x^2 + beta) on one variable, as the factor (1, v)."""
    return SemiringGaussianFactor.Reward(
        HessianFactor(key, np.array([[-2.0 * P]]), zero, -2.0 * beta))


def read(factor):
    """(P, beta) of a factor on one variable whose value is -(P x^2 + beta)."""
    G = factor.value().augmentedInformation()
    return -G[0, 0] / 2, -G[1, 1] / 2


def action_bucket(P, beta, reward=stage_reward):
    """The bucket of the action, (1, Q): the reward now, plus gamma times the
    average over the next state of its value -(P x'^2 + beta)."""
    future = quadratic(X(1), gamma * P, gamma * beta)
    return reward.multiply(transition.multiply(future).sum(ordering(X(1))))


P, beta = 1.0, 0.0
for _ in range(1000):  # Riccati: eliminate x' by average, u by maximum
    bucket = action_bucket(P, beta)
    conditional, new_value = bucket.eliminate(ordering(U(0)), maximum)
    P, beta = read(new_value)
K_star = conditional.conditional().S()[0, 0]  # the conditional is u + K x = 0
H_uu, H_ux, H_xx, constant = blocks(action_bucket(P, beta))
m_0 = MU_0 ** 2 + SIGMA_0
J_star = prior.multiply(quadratic(X(0), P, beta)).expectation()
print(f"P* = {P:.4f}, K* = {K_star:.4f}, J* = {J_star:.4f}")
print(f"blocks of Q*: H_uu = {H_uu:.4f}, H_ux = {H_ux:.4f}, "
      f"H_xx = {H_xx:.4f}, constant = {constant:.4f}")
assert np.isclose(K_star, 0.5884, atol=1e-4)
assert np.isclose(P, 1.5884, atol=1e-4)
assert np.isclose(J_star, -15.0898, atol=1e-4)


def evaluate(K, reward=stage_reward):
    """(P, beta) of the value of u = -K x for a quadratic reward: eliminate
    with the average rule until the value stops changing."""
    rule = policy(0, K)
    P, beta = 0.0, 0.0
    for _ in range(2000):
        new_P, new_beta = read(rule.multiply(action_bucket(P, beta, reward)).sum(
            ordering(U(0))))
        if abs(new_P - P) + abs(new_beta - beta) < 1e-12:
            break
        P, beta = new_P, new_beta
    return new_P, new_beta


def stage1_endless(K):
    """Exact messages of u = -K x on the endless line."""
    P, beta = evaluate(K)  # V(x) = -(P x^2 + beta)
    J = prior.multiply(quadratic(X(0), P, beta)).expectation()
    H_uu, H_ux = blocks(action_bucket(P, beta))[:2]
    # The discounted second moment m = sum_t gamma^t E[x_t^2] is the expected
    # discounted sum of the "reward" x^2 (Chapter 3, Section 5): the same
    # elimination with x^2 in place of the reward.
    m = prior.multiply(quadratic(X(0), *evaluate(K, square(X(0))))).expectation()
    return J, H_uu, H_ux, m


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
assert abs(gradient_endless(K_star)) < 1e-8
assert np.isclose(gradient_endless(0.3), 31.557, atol=1e-3)
assert np.isclose(gradient_endless(0.8), -9.732, atol=1e-3)
# The forward message against its closed form.
assert np.isclose(stage1_endless(0.3)[3],
                  (m_0 + gamma * SIGMA_W / (1 - gamma)) /
                  (1 - gamma * (1 - 0.3) ** 2))

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
