# %% [markdown]
# # Chapter 6 examples: LQR and LQG
#
# This notebook runs the examples of
# [Chapter 6](https://thduynguyen.github.io/gtsam/chapter06) on the robot on a
# line of Chapter 1: the Riccati recursion as elimination with the semiring
# module, its fixed point for an endless problem, the Kalman filter as the
# forward pass, and the two combined when the robot only has a noisy sensor.

# %%
import numpy as np
from gtsam import GaussianFactorGraph, HessianFactor, JacobianFactor, Ordering
from gtsam import SemiringFactorGraph, SemiringGaussianFactor, noiseModel
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
# First in numpy, exactly as derived in the chapter.


# %%
def riccati(F, B, C_x, C_u, C_T, Sigma_w, moves):
    """Backward pass: gains K_t, matrices P_t and constants beta_t."""
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


K, P, beta = riccati(F, B, C_x, C_u, C_T, Sigma_w, moves)
J_star = -(mu_0 @ P[0] @ mu_0 + np.trace(P[0] @ Sigma_0) + beta[0])
for t in reversed(range(moves)):
    print(f"t = {t}: K = {K[t].item():.4f}, P = {P[t].item():.4f}, "
          f"beta = {beta[t]:.4f}")
print("J* =", J_star)
assert np.isclose(K[1].item(), 0.5) and np.isclose(K[0].item(), 0.6)
assert np.isclose(P[1].item(), 1.5) and np.isclose(P[0].item(), 1.6)
assert np.isclose(beta[1], 0.5) and np.isclose(beta[0], 1.25)
assert np.isclose(J_star, -9.25)

# %% [markdown]
# Then the same pass with the module. The next state is eliminated by
# average. The maximum over a move solves a linear equation for the gain, and
# the best policy is put back as the hard constraint $u = -K x$.

# %%
gains = {}
value = penalty(X(2))  # (1, V_2)
for t in reversed(range(moves)):
    # Eliminate the next state by average.
    step = gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero, variance(0.5))
    phi = step.multiply(value).sum(ordering(X(t + 1)))
    # Multiply with the rewards: values add, giving (1, Q_t).
    bucket = penalty(X(t)).multiply(penalty(U(t))).multiply(phi)
    # Maximize over u: solve H_uu u = -H_ux x for the gain.
    H, _ = quadratic(bucket, U(t), X(t))  # Q_t = z' H z + c, z = (u, x)
    gains[t] = H[0, 1] / H[0, 0]
    # The best policy u = -K x as a factor, then eliminate u.
    best = gaussian(U(t), I, X(t), gains[t] * I, zero,
                    noiseModel.Constrained.All(1))
    value = best.multiply(bucket).sum(ordering(U(t)))  # (1, V_t)
    M, c = quadratic(value, X(t))
    print(f"t = {t}: gain = {gains[t]:.4f}, V_t(x) = {M.item():.4f} x^2 "
          f"+ {c:.4f}")
prior = gaussian(X(0), I, mu_0, variance(1.0))
J_module = prior.multiply(value).expectation()
print("J* =", J_module)
assert np.isclose(gains[0], 0.6) and np.isclose(gains[1], 0.5)
assert np.isclose(J_module, -9.25)

# %% [markdown]
# ## The endless line: a fixed point (Section 4)
#
# Unrolling more and more moves, the matrix $P$ and the gain settle. For the
# line the limits are the golden ratio and its inverse.

# %%
golden = (1 + np.sqrt(5)) / 2
K_long, P_long, beta_long = riccati(F, B, C_x, C_u, C_T, Sigma_w, 12)
for k in [1, 2, 3, 4, 6, 12]:  # k moves left = step 12 - k
    print(f"{k:2d} moves left: P = {P_long[12 - k].item():.6f}, "
          f"K = {K_long[12 - k].item():.6f}")
print("golden ratio:", golden, " its inverse:", 1 / golden)
P_inf = P_long[0].item()
assert np.isclose(P_inf, golden, atol=1e-6)
assert np.isclose(K_long[0].item(), 1 / golden, atol=1e-6)
# The fixed point satisfies the algebraic Riccati equation.
assert np.isclose(golden, 1 + golden - golden ** 2 / (1 + golden))
print("reward lost per move to the noise, P * Sigma_w =", golden * 0.5)

# %% [markdown]
# With the discount $\gamma = 0.9$ of Chapter 3, the future value is scaled by
# $\gamma$ before each backup, and the constant converges too.

# %%
gamma = 0.9
P_gamma, beta_gamma = 1.0, 0.0
for k in range(500):
    K_gamma = gamma * P_gamma / (1 + gamma * P_gamma)
    beta_gamma = gamma * (beta_gamma + P_gamma * 0.5)
    P_gamma = 1 + gamma * P_gamma - (gamma * P_gamma) ** 2 / (1 + gamma * P_gamma)
J_gamma = -(P_gamma * (2.0 ** 2 + 1.0) + beta_gamma)
print(f"discounted: P = {P_gamma:.4f}, K = {K_gamma:.4f}, "
      f"beta = {beta_gamma:.4f}, J* = {J_gamma:.4f}")
assert np.isclose(P_gamma, 1.5884, atol=1e-4)
assert np.isclose(K_gamma, 0.5884, atol=1e-4)
assert np.isclose(beta_gamma, 7.1478, atol=1e-4)
assert np.isclose(J_gamma, -15.0898, atol=1e-4)

# %% [markdown]
# ## The Kalman filter is the forward pass (Section 5)
#
# The robot now has a noisy position sensor, $y = G x + n$ with
# $n \sim N(0, \Sigma_y)$. Eliminating the states *forward* in time on the
# probability channel, an ordinary Gaussian factor graph, gives the filter.

# %%
G, Sigma_y = np.eye(1), 0.5 * np.eye(1)
y0, u0, y1 = 2.6, -1.2, 0.9  # one possible record of readings and a move

graph = GaussianFactorGraph()
graph.add(JacobianFactor(X(0), I, mu_0, variance(1.0)))  # prior
graph.add(JacobianFactor(X(0), I, np.array([y0]), variance(0.5)))  # reading
after_first = graph.eliminateSequential(ordering(X(0))).at(0)
mean_00 = (after_first.d() / after_first.R()).item()
var_00 = (1 / after_first.R() ** 2).item()
print(f"after y0: mean {mean_00:.4f}, variance {var_00:.4f}")

graph.add(JacobianFactor(X(1), I, X(0), -I, np.array([u0]), variance(0.5)))
graph.add(JacobianFactor(X(1), I, np.array([y1]), variance(0.5)))
last = graph.eliminateSequential(ordering(X(0), X(1))).at(1)
mean_11 = (last.d() / last.R()).item()
var_11 = (1 / last.R() ** 2).item()
print(f"after y1: mean {mean_11:.4f}, variance {var_11:.4f}")


def kalman(F, G, Sigma_w, Sigma_y, Sigma_0, moves):
    """Filter gains L_t and variances before and after each reading."""
    predicted, updated, L = {0: Sigma_0}, {}, {}
    for t in range(moves):
        S_y = G @ predicted[t] @ G.T + Sigma_y
        L[t] = predicted[t] @ G.T @ np.linalg.inv(S_y)
        updated[t] = predicted[t] - L[t] @ G @ predicted[t]
        predicted[t + 1] = F @ updated[t] @ F.T + Sigma_w
    return L, predicted, updated


L, predicted, updated = kalman(F, G, Sigma_w, Sigma_y, Sigma_0, moves)
x_hat_0 = mu_0 + L[0] @ (np.array([y0]) - G @ mu_0)
x_pred_1 = F @ x_hat_0 + B @ np.array([u0])
x_hat_1 = x_pred_1 + L[1] @ (np.array([y1]) - G @ x_pred_1)
print(f"Kalman filter: L0 = {L[0].item():.4f}, L1 = {L[1].item():.4f}")
print(f"  after y0: mean {x_hat_0.item():.4f}, variance "
      f"{updated[0].item():.4f}")
print(f"  predicted x1: variance {predicted[1].item():.4f}")
print(f"  after y1: mean {x_hat_1.item():.4f}, variance "
      f"{updated[1].item():.4f}")
assert np.isclose(mean_00, x_hat_0.item()) and np.isclose(var_00, 1 / 3)
assert np.isclose(mean_11, x_hat_1.item()) and np.isclose(var_11, 0.3125)
assert np.isclose(L[0].item(), 2 / 3) and np.isclose(L[1].item(), 0.625)

# %% [markdown]
# ## Duality (Section 5)
#
# The predicted variance of the filter obeys the same recursion as the matrix
# $P$ of the controller, run in the other direction with the roles of the
# matrices exchanged: $F \to F^\top$, $B \to G^\top$, $C_x \to \Sigma_w$,
# $C_u \to \Sigma_y$, $C_T \to \Sigma_0$.

# %%
_, dual, _ = riccati(F.T, G.T, Sigma_w, Sigma_y, Sigma_0, np.zeros((1, 1)), 2)
# The "controller" runs backward from step 2; the filter forward from step 0.
print("control recursion on the dual problem:",
      [dual[t].item() for t in (2, 1, 0)])
print("predicted variances of the filter:    ",
      [predicted[t].item() for t in (0, 1, 2)])
for t in range(3):
    assert np.isclose(dual[2 - t].item(), predicted[t].item())

# %% [markdown]
# ## LQG: the filter and the controller together (Section 6)
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


K0, K1 = K[0].item(), K[1].item()
L0, L1 = L[0].item(), L[1].item()
J_lqg = lqg_graph(K0, K1, L0, L1).expectation()
print("J of the LQG controller, by elimination:", J_lqg)

# The formula: the full-state optimum minus the price of not seeing the state.
H_uu = {t: (C_u + B.T @ P[t + 1] @ B).item() for t in range(2)}
price = sum(K[t].item() ** 2 * H_uu[t] * updated[t].item() for t in range(2))
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
# ## Separation (Section 6)
#
# Changing either the controller gains or the estimator gains away from the
# Riccati and Kalman values lowers the expected return. The two designs do
# not interact.

# %%
for dK0, dK1, dL0, dL1 in [(0.05, 0, 0, 0), (-0.05, 0, 0, 0), (0, 0.05, 0, 0),
                           (0, -0.05, 0, 0), (0, 0, 0.05, 0), (0, 0, -0.05, 0),
                           (0, 0, 0, 0.05), (0, 0, 0, -0.05)]:
    J = lqg_graph(K0 + dK0, K1 + dK1, L0 + dL0, L1 + dL1).expectation()
    print(f"K0{dK0:+.2f} K1{dK1:+.2f} L0{dL0:+.2f} L1{dL1:+.2f}: "
          f"J = {J:.5f}  (change {J - J_lqg:+.5f})")
    assert J < J_lqg
