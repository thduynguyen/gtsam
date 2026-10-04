# %% [markdown]
# # Chapter 6 examples: LQR and LQG
#
# This notebook runs the examples of
# [Chapter 6](https://thduynguyen.github.io/gtsam/chapter06) on the robot on a
# line of Chapter 1, with GTSAM: the Riccati recursion as elimination of a
# semiring factor graph, with the average at the states and the maximum at
# the actions; its fixed point for an endless problem; the Kalman filter as
# forward elimination of a Gaussian factor graph; and the two combined when
# the robot only has a noisy sensor. The textbook formulas appear in numpy
# only as independent checks.

# %%
import numpy as np
from gtsam import GaussianFactorGraph, HessianFactor, JacobianFactor, Ordering
from gtsam import SemiringFactorGraph, SemiringGaussianFactor
from gtsam import SemiringRules, SemiringSum, noiseModel
from gtsam.symbol_shorthand import U, X, Y

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line of Chapter 1, in matrix notation (Section 1)
#
# $x' = F x + B u + w$ with $w \sim N(0, \Sigma_w)$, reward
# $-(x^\top C_x x + u^\top C_u u)$ per move and $-x^\top C_T x$ at the end.
# For the line every matrix is the number 1, except $\Sigma_w = 0.5$.

# %%
F, B = np.eye(1), np.eye(1)
C_x, C_u, C_T = np.eye(1), np.eye(1), np.eye(1)
Sigma_w = 0.5 * np.eye(1)
mu_0, Sigma_0 = np.array([2.0]), np.eye(1)
moves = 2

# %% [markdown]
# ## A Gaussian semiring factor (Section 2)
#
# A `SemiringGaussianFactor` holds the pair in log-dual form: a Gaussian
# factor graph for the probability channel and a quadratic for the value
# channel.

# %%
I, zero = np.eye(1), np.zeros(1)


def variance(v):
    """A scalar Gaussian noise model with the given variance."""
    return noiseModel.Isotropic.Variance(1, v)


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def ordering(*keys):
    """An ordering holding the given keys."""
    result = Ordering()
    for key in keys:
        result.push_back(key)
    return result


def quadratic(factor, *keys):
    """The value channel as (M, c): v(z) = z' M z + c, z ordered as keys."""
    value = factor.value()
    position = [list(value.keys()).index(key) for key in keys]
    augmented = value.augmentedInformation()
    M = 0.5 * augmented[np.ix_(position, position)]
    return M, 0.5 * augmented[-1, -1]


dynamics = gaussian(X(1), I, X(0), -I, U(0), -I, zero, variance(0.5))
final = penalty(X(1))
print("dynamics: Gaussian factors =", dynamics.gaussian().size(),
      ", value =", dynamics.value())
print("penalty:  Gaussian factors =", final.gaussian().size(),
      ", value (M, c) =", quadratic(final, X(1)))

# Multiply: both channels add. Sum out x1: the expected value given (x0, u0).
phi = dynamics.multiply(final).sum(ordering(X(1)))
M, c = quadratic(phi, X(0), U(0))
print("phi(x0, u0): value = z' M z + c with M =\n", M, "\nand c =", c)
assert np.allclose(M, -np.ones((2, 2))) and np.isclose(c, -0.5)

# %% [markdown]
# ## The Riccati recursion (Section 3)
#
# The graph has no policy factors. The states are summed out by the average
# and the actions by the maximum, backward in time. The best expected return
# is left at the root, and the conditional of each action is the control law
# $u_t = -K_t x_t$.

# %%
maximum = SemiringSum.Maximum()


def line_graph():
    """The line of Chapter 1 without its policy factors."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, mu_0, variance(1.0)))
    for t in range(moves):
        graph.push_back(
            gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero, variance(0.5)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(moves)))
    return graph


rules = SemiringRules()
rules.setAll([U(0), U(1)], maximum)
backward = ordering(X(2), U(1), X(1), U(0), X(0))

graph = line_graph()
J_star = graph.expectation(backward, rules)
bayes_net = graph.eliminateSequential(backward, rules)
# The conditional of u_t is u_t + K_t x_t = 0: the gain is its parent block.
gains = {1: bayes_net.at(1).conditional().S().item(),
         0: bayes_net.at(3).conditional().S().item()}
print("J* =", J_star)
print("gains K_0, K_1 =", gains[0], gains[1])
assert np.isclose(J_star, -9.25)
assert np.isclose(gains[0], 0.6) and np.isclose(gains[1], 0.5)

# %% [markdown]
# The same pass one elimination at a time, for general matrices, to read
# $K_t$, $P_t$ and $\beta_t$ from the factors: $V_t(x) = -(x^\top P_t x +
# \beta_t)$ is the value of the new factor on $x_t$.


# %%
def cost(key, C):
    """Lift the cost z' C z on one variable to the reward (1, -z' C z)."""
    return SemiringGaussianFactor.Cost(
        HessianFactor(key, 2 * C, np.zeros(len(C)), 0.0))


def riccati_by_elimination(F, B, C_x, C_u, C_T, Sigma_w, moves):
    """Backward pass on semiring factors: gains K_t, P_t and beta_t."""
    n = len(F)
    noise = (noiseModel.Gaussian.Covariance(Sigma_w) if Sigma_w.any()
             else noiseModel.Constrained.All(n))  # deterministic dynamics
    K, P, beta = {}, {moves: C_T}, {moves: 0.0}
    value = cost(X(moves), C_T)  # (1, V_T)
    for t in reversed(range(moves)):
        # Eliminate the next state by average.
        step = gaussian(X(t + 1), np.eye(n), X(t), -F, U(t), -B, np.zeros(n),
                        noise)
        phi = step.multiply(value).sum(ordering(X(t + 1)))
        # Multiply with the rewards: values add, giving (1, Q_t).
        bucket = cost(X(t), C_x).multiply(cost(U(t), C_u)).multiply(phi)
        # Eliminate the action by maximum.
        conditional, value = bucket.eliminate(ordering(U(t)), maximum)
        # The conditional is u + K x = 0: copy the gain out of it.
        K[t] = conditional.conditional().S().copy()
        M, c = quadratic(value, X(t))  # V_t(x) = x' M x + c
        P[t], beta[t] = -M, -c
    return K, P, beta


K, P, beta = riccati_by_elimination(F, B, C_x, C_u, C_T, Sigma_w, moves)
for t in reversed(range(moves)):
    print(f"t = {t}: K = {K[t].item():.4f}, P = {P[t].item():.4f}, "
          f"beta = {beta[t]:.4f}")
assert np.isclose(K[1].item(), 0.5) and np.isclose(K[0].item(), 0.6)
assert np.isclose(P[1].item(), 1.5) and np.isclose(P[0].item(), 1.6)
assert np.isclose(beta[1], 0.5) and np.isclose(beta[0], 1.25)
assert np.isclose(
    -(mu_0 @ P[0] @ mu_0 + np.trace(P[0] @ Sigma_0) + beta[0]), J_star)

# %% [markdown]
# An independent check: the Riccati formulas of the chapter, written out.


# %%
def riccati(F, B, C_x, C_u, C_T, Sigma_w, moves):
    """The Riccati formulas: gains K_t, matrices P_t and constants beta_t."""
    P, beta = {moves: C_T}, {moves: 0.0}
    K = {}
    for t in reversed(range(moves)):
        H_uu = C_u + B.T @ P[t + 1] @ B
        H_ux = B.T @ P[t + 1] @ F
        H_xx = C_x + F.T @ P[t + 1] @ F
        K[t] = np.linalg.solve(H_uu, H_ux)  # maximize over u
        P[t] = H_xx - H_ux.T @ K[t]  # the Schur complement
        beta[t] = beta[t + 1] + np.trace(P[t + 1] @ Sigma_w)
    return K, P, beta


K_formula, P_formula, beta_formula = riccati(
    F, B, C_x, C_u, C_T, Sigma_w, moves)
for t in range(moves):
    assert np.allclose(K[t], K_formula[t]) and np.allclose(P[t], P_formula[t])
    assert np.isclose(beta[t], beta_formula[t])
print("the formulas agree with the elimination")

# %% [markdown]
# ## The endless line: a fixed point (Section 5)
#
# Every step has the same factors, so one step is eliminated again and again,
# and the value of the new factor is fed back in as the value of the next
# state. The matrix $P$ and the gain settle. For the line the limits are the
# golden ratio and its inverse.


# %%
def as_next_value(M, c, scale=1.0):
    """The value scale * (x' M x + c), as a factor on the next state x1."""
    return SemiringGaussianFactor.Reward(
        HessianFactor(X(1), 2 * scale * M, zero, 2 * scale * c))


step = gaussian(X(1), I, X(0), -I, U(0), -I, zero, variance(0.5))
stage_reward = penalty(X(0)).multiply(penalty(U(0)))


def backup(M, c, discount=1.0):
    """One step: the next state by average, then the action by maximum.

    The next state is worth discount * (x' M x + c). Returns the gain and the
    value (M, c) of the current state.
    """
    phi = step.multiply(as_next_value(M, c, discount)).sum(ordering(X(1)))
    conditional, value = stage_reward.multiply(phi).eliminate(
        ordering(U(0)), maximum)
    return conditional.conditional().S().item(), quadratic(value, X(0))


golden = (1 + np.sqrt(5)) / 2
M, c = -C_T, 0.0  # V_T(x) = -x^2
P_left, K_left = {}, {}
for k in range(1, 13):  # k moves left
    K_left[k], (M, c) = backup(M, c)
    P_left[k] = -M.item()
for k in [1, 2, 3, 4, 6, 12]:
    print(f"{k:2d} moves left: P = {P_left[k]:.6f}, K = {K_left[k]:.6f}")
print("golden ratio:", golden, " its inverse:", 1 / golden)
assert np.isclose(P_left[1], 1.5) and np.isclose(P_left[2], 1.6)
assert np.isclose(P_left[12], golden, atol=1e-6)
assert np.isclose(K_left[12], 1 / golden, atol=1e-6)
# The fixed point satisfies the algebraic Riccati equation.
assert np.isclose(golden, 1 + golden - golden ** 2 / (1 + golden))
print("reward lost per move to the noise, P * Sigma_w =", golden * 0.5)

# %% [markdown]
# With the discount $\gamma = 0.9$ of Chapter 3, the value of the next state
# is scaled by $\gamma$ in every backup, and the constant converges too.

# %%
gamma = 0.9
M, c = -C_T, 0.0
for sweep in range(500):
    K_gamma, (new_M, new_c) = backup(M, c, gamma)
    change = max(abs(new_M - M).max(), abs(new_c - c))
    M, c = new_M, new_c
    if change < 1e-13:
        break
P_gamma, beta_gamma = -M.item(), -c
# The value of the first state, averaged over where the robot starts.
start = gaussian(X(0), I, mu_0, variance(1.0))
V_gamma = SemiringGaussianFactor.Reward(HessianFactor(X(0), 2 * M, zero, 2 * c))
J_gamma = start.multiply(V_gamma).expectation()
print(f"discounted, after {sweep + 1} sweeps: P = {P_gamma:.4f}, "
      f"K = {K_gamma:.4f}, beta = {beta_gamma:.4f}, J* = {J_gamma:.4f}")
assert np.isclose(P_gamma, 1.5884, atol=1e-4)
assert np.isclose(K_gamma, 0.5884, atol=1e-4)
assert np.isclose(beta_gamma, 7.1478, atol=1e-4)
assert np.isclose(J_gamma, -15.0898, atol=1e-4)
# The constant is at its own fixed point, beta = gamma tr(P Sigma_w) / (1 - gamma).
assert np.isclose(beta_gamma, gamma * P_gamma * 0.5 / (1 - gamma), atol=1e-6)

# %% [markdown]
# ## The Kalman filter is the forward pass (Section 6)
#
# The robot now has a noisy position sensor, $y = G x + n$ with
# $n \sim N(0, \Sigma_y)$. Eliminating the states *forward* in time on the
# probability channel, an ordinary Gaussian factor graph, gives the filter:
# the last conditional is the estimate of the latest state.

# %%
G, Sigma_y = np.eye(1), 0.5 * np.eye(1)
y0, u0, y1, u1 = 2.6, -1.2, 0.9, -0.3  # one record of readings and moves


def estimate(graph, last):
    """Eliminate x0 .. x_last forward; the mean and variance of x_last."""
    conditional = graph.eliminateSequential(
        ordering(*[X(t) for t in range(last + 1)])).at(last)
    return ((conditional.d() / conditional.R()).item(),
            (1 / conditional.R() ** 2).item())


mean, predicted, updated = {}, {}, {}
graph = GaussianFactorGraph()
graph.add(JacobianFactor(X(0), I, mu_0, variance(1.0)))  # prior
_, predicted[0] = estimate(graph, 0)
graph.add(JacobianFactor(X(0), I, np.array([y0]), variance(0.5)))  # reading
mean[0], updated[0] = estimate(graph, 0)
graph.add(JacobianFactor(X(1), I, X(0), -I, np.array([u0]), variance(0.5)))
_, predicted[1] = estimate(graph, 1)
graph.add(JacobianFactor(X(1), I, np.array([y1]), variance(0.5)))
mean[1], updated[1] = estimate(graph, 1)
graph.add(JacobianFactor(X(2), I, X(1), -I, np.array([u1]), variance(0.5)))
_, predicted[2] = estimate(graph, 2)
# The filter gain L_t = Sigma_{t|t} G' Sigma_y^-1, from the variances.
L = {t: updated[t] / 0.5 for t in range(2)}
for t in range(2):
    print(f"step {t}: variance before the reading {predicted[t]:.4f}, "
          f"gain L = {L[t]:.4f}, after: mean {mean[t]:.4f}, "
          f"variance {updated[t]:.4f}")
print(f"variance of x2 before its reading: {predicted[2]:.4f}")
assert np.isclose(updated[0], 1 / 3) and np.isclose(updated[1], 0.3125)
assert np.isclose(predicted[1], 0.8333, atol=1e-4)
assert np.isclose(L[0], 2 / 3) and np.isclose(L[1], 0.625)

# %% [markdown]
# An independent check: the Kalman formulas of the chapter, written out.


# %%
def kalman(F, G, Sigma_w, Sigma_y, Sigma_0, moves):
    """Filter gains L_t and variances before and after each reading."""
    predicted, updated, L = {0: Sigma_0}, {}, {}
    for t in range(moves):
        S_y = G @ predicted[t] @ G.T + Sigma_y
        L[t] = predicted[t] @ G.T @ np.linalg.inv(S_y)
        updated[t] = predicted[t] - L[t] @ G @ predicted[t]
        predicted[t + 1] = F @ updated[t] @ F.T + Sigma_w
    return L, predicted, updated


L_formula, predicted_formula, updated_formula = kalman(
    F, G, Sigma_w, Sigma_y, Sigma_0, moves)
x_hat_0 = mu_0 + L_formula[0] @ (np.array([y0]) - G @ mu_0)
x_pred_1 = F @ x_hat_0 + B @ np.array([u0])
x_hat_1 = x_pred_1 + L_formula[1] @ (np.array([y1]) - G @ x_pred_1)
assert np.isclose(mean[0], x_hat_0.item()) and np.isclose(mean[1], x_hat_1.item())
for t in range(2):
    assert np.isclose(L[t], L_formula[t].item())
    assert np.isclose(updated[t], updated_formula[t].item())
for t in range(3):
    assert np.isclose(predicted[t], predicted_formula[t].item())
print("the formulas agree with the elimination")

# %% [markdown]
# ## Duality (Section 6)
#
# The predicted variance of the filter obeys the same recursion as the matrix
# $P$ of the controller, run in the other direction with the roles of the
# matrices exchanged: $F \to F^\top$, $B \to G^\top$, $C_x \to \Sigma_w$,
# $C_u \to \Sigma_y$, $C_T \to \Sigma_0$. So the backward elimination of
# Section 3, on the exchanged matrices, gives the filter's variances.

# %%
_, dual, _ = riccati_by_elimination(F.T, G.T, Sigma_w, Sigma_y, Sigma_0,
                                    np.zeros((1, 1)), 2)
# The "controller" runs backward from step 2; the filter forward from step 0.
print("backward elimination of the dual problem:",
      [dual[t].item() for t in (2, 1, 0)])
print("predicted variances of the filter:       ",
      [predicted[t] for t in (0, 1, 2)])
for t in range(3):
    assert np.isclose(dual[2 - t].item(), predicted[t])
assert np.allclose([predicted[t] for t in range(3)], [1, 0.8333, 0.8125],
                   atol=1e-4)

# %% [markdown]
# ## LQG: the filter and the controller together (Section 7)
#
# The robot acts on its estimate, $u_t = -K_t\, \hat x_t$, with the Riccati
# gains and the Kalman estimate. Each estimate is a linear function of the
# readings, so the policy is a hard linear constraint on $(u_t, y_0, \dots,
# y_t)$, and the closed loop is one semiring factor graph.


# %%
def lqg_graph(K0, K1, L0, L1):
    """The closed loop for controller gains K and estimator gains L."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, mu_0, variance(1.0)))
    for t in range(2):
        graph.push_back(gaussian(Y(t), I, X(t), -I, zero, variance(0.5)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 variance(0.5)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    # x_hat_0 = (1 - L0) mu_0 + L0 y0, and u0 = -K0 x_hat_0.
    hard = noiseModel.Constrained.All(1)
    graph.push_back(gaussian(U(0), I, Y(0), K0 * L0 * I,
                             -K0 * (1 - L0) * mu_0, hard))
    # x_hat_1 = (1 - L1) (1 - K0) x_hat_0 + L1 y1, and u1 = -K1 x_hat_1.
    carry = K1 * (1 - L1) * (1 - K0)
    graph.push_back(gaussian(U(1), I, Y(1), K1 * L1 * I, Y(0), carry * L0 * I,
                             -carry * (1 - L0) * mu_0, hard))
    return graph


# Eliminate backward in time: each variable after those that depend on it.
closed_loop_order = ordering(X(2), U(1), Y(1), X(1), U(0), Y(0), X(0))

K0, K1 = K[0].item(), K[1].item()
L0, L1 = L[0], L[1]
J_lqg = lqg_graph(K0, K1, L0, L1).expectation(closed_loop_order)
print("J of the LQG controller, by elimination:", J_lqg)

# The formula: the full-state optimum minus the price of not seeing the state.
H_uu = {t: (C_u + B.T @ P[t + 1] @ B).item() for t in range(2)}
price = sum(K[t].item() ** 2 * H_uu[t] * updated[t] for t in range(2))
print("price of the noisy sensor:", price)
print("J* of LQR minus the price:", J_star - price)
assert np.isclose(J_lqg, -9.70625) and np.isclose(J_star - price, J_lqg)

# %% [markdown]
# A Monte Carlo check of the same number: simulate the closed loop.

# %%
rng = np.random.default_rng(0)
samples = 400_000
x = 2.0 + rng.normal(size=samples)
x_hat, total = np.full(samples, 2.0), np.zeros(samples)
for t, (K_t, L_t) in enumerate([(K0, L0), (K1, L1)]):
    y = x + np.sqrt(0.5) * rng.normal(size=samples)
    x_hat = x_hat + L_t * (y - x_hat)  # update with the reading
    u = -K_t * x_hat
    total -= x ** 2 + u ** 2
    x = x + u + np.sqrt(0.5) * rng.normal(size=samples)
    x_hat = x_hat + u  # predict
total -= x ** 2
print(f"Monte Carlo: {total.mean():.4f} +- "
      f"{total.std() / np.sqrt(samples):.4f}")
assert abs(total.mean() - J_lqg) < 0.03

# %% [markdown]
# ## Separation (Section 7)
#
# Changing either the controller gains or the estimator gains away from the
# Riccati and Kalman values lowers the expected return. The two designs do
# not interact.

# %%
for dK0, dK1, dL0, dL1 in [(0.05, 0, 0, 0), (-0.05, 0, 0, 0), (0, 0.05, 0, 0),
                           (0, -0.05, 0, 0), (0, 0, 0.05, 0), (0, 0, -0.05, 0),
                           (0, 0, 0, 0.05), (0, 0, 0, -0.05)]:
    J = lqg_graph(K0 + dK0, K1 + dK1, L0 + dL0, L1 + dL1).expectation(
        closed_loop_order)
    print(f"K0{dK0:+.2f} K1{dK1:+.2f} L0{dL0:+.2f} L1{dL1:+.2f}: "
          f"J = {J:.5f}  (change {J - J_lqg:+.5f})")
    assert J < J_lqg
