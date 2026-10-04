# %% [markdown]
# # Chapter 26 examples: robotics case studies
#
# This notebook runs the two small illustrations of
# [Chapter 26](https://thduynguyen.github.io/gtsam/chapter26) on the line of
# Chapter 1: a policy trained on a range of simulated models (domain
# randomization), and a learned correction on top of a nominal controller
# (residual learning). Both are exact: no sampling is involved.

# %%
import numpy as np
from gtsam import HessianFactor, JacobianFactor, noiseModel
from gtsam import SemiringFactorGraph, SemiringGaussianFactor
from gtsam.symbol_shorthand import U, X
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line, with an actuator gain $B$ (Section 2)
#
# $x' = x + B u + w$, $w \sim N(0, 0.5)$, rewards $-(x^2 + u^2)$ per move and
# $-x_2^2$ at the end, $x_0 \sim N(2, 1)$, two moves. Chapter 1 had $B = 1$.
# The policy is a gain per move, $u_t = -K_t x_t$.

# %%
sigma_w, mean0, variance0 = 0.5, 2.0, 1.0


def evaluate(gains, B):
    """Expected return of u_t = -K_t x_t on the line with actuator gain B.

    The backward recursion for a fixed linear policy: V_t(x) = -(P x^2 + beta).
    """
    P, beta = 1.0, 0.0  # V_2(x) = -x^2
    for K in reversed(gains):
        beta = beta + P * sigma_w
        P = 1 + K ** 2 + (1 - B * K) ** 2 * P
    return -(P * (mean0 ** 2 + variance0) + beta)


def riccati(B):
    """The best gains for actuator gain B, by the Riccati recursion."""
    P, gains = 1.0, []
    for _ in range(2):
        K = B * P / (1 + B ** 2 * P)
        gains.insert(0, K)
        P = 1 + P - (B * P) ** 2 / (1 + B ** 2 * P)
    return np.array(gains)


nominal = riccati(1.0)
print("Riccati gains for B = 1:", nominal, " J =", evaluate(nominal, 1.0))
assert np.allclose(nominal, [0.6, 0.5])
assert np.isclose(evaluate(nominal, 1.0), -9.25)
assert np.isclose(evaluate([0.5, 0.5], 1.0), -9.375)

# %% [markdown]
# The same evaluation with the module, as a check: a `SemiringFactorGraph` of
# Gaussian factors, with the policy as a hard constraint $u + K x = 0$.

# %%
I, zero = np.eye(1), np.zeros(1)


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def module_evaluate(gains, B):
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([mean0]),
                             noiseModel.Isotropic.Variance(1, variance0)))
    for t, K in enumerate(gains):
        graph.push_back(gaussian(U(t), I, X(t), K * I, zero,
                                 noiseModel.Constrained.All(1)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -B * I, zero,
                                 noiseModel.Isotropic.Variance(1, sigma_w)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph.expectation()


for B in [0.5, 1.0]:
    print(f"B = {B}: module {module_evaluate(nominal, B):.4f}, "
          f"recursion {evaluate(nominal, B):.4f}")
    assert np.isclose(module_evaluate(nominal, B), evaluate(nominal, B))

# %% [markdown]
# ## Domain randomization (Section 2)
#
# The simulator's actuator gain is uncertain: 0.5, 1 or 1.5, equally likely.
# One policy is trained on the average over the three models, and compared
# with the policy designed for $B = 1$ alone and with the best policy of each
# model.

# %%
models = [0.5, 1.0, 1.5]


def randomized_objective(gains):
    return -np.mean([evaluate(gains, B) for B in models])


randomized = minimize(randomized_objective, nominal).x
print("gains trained on B = 1 only:        ", nominal)
print("gains trained on the three models:  ", randomized)
table = {}
for B in models:
    table[B] = (evaluate(nominal, B), evaluate(randomized, B),
                evaluate(riccati(B), B))
    print(f"B = {B}: nominal {table[B][0]:8.4f}, randomized "
          f"{table[B][1]:8.4f}, best for this B {table[B][2]:8.4f}")
print("average over the models: nominal",
      np.mean([row[0] for row in table.values()]), " randomized",
      np.mean([row[1] for row in table.values()]))
assert -randomized_objective(randomized) >= -randomized_objective(nominal)
assert all(row[2] >= max(row[0], row[1]) - 1e-9 for row in table.values())

# %% [markdown]
# ## Residual learning (Section 5)
#
# The real robot has $B = 0.5$. The nominal controller was designed for
# $B = 1$. A correction to its gains is learned on the real system by
# gradient ascent, and compared with learning the gains from zero with the
# same step size.

# %%
B_real = 0.5
best = riccati(B_real)
J_best = evaluate(best, B_real)
print("best gains for the real system:", best, " J =", J_best)
print("nominal controller on the real system: J =", evaluate(nominal, B_real))
print("no controller on the real system:      J =", evaluate([0, 0], B_real))


def gradient(gains):
    """Gradient of J on the real system, by central differences."""
    h = 1e-6
    return np.array([
        (evaluate(gains + h * e, B_real) - evaluate(gains - h * e, B_real))
        / (2 * h) for e in np.eye(2)])


def learn(start, iterations, step=0.02):
    gains, history = np.array(start, float), []
    for k in range(iterations + 1):
        history.append(evaluate(gains, B_real))
        gains = gains + step * gradient(gains)
    return gains, history


residual_gains, residual = learn(nominal, 100)
scratch_gains, scratch = learn([0.0, 0.0], 100)
for k in [0, 1, 2, 5, 10, 20, 50, 100]:
    print(f"iteration {k:3d}: residual J = {residual[k]:8.4f}, "
          f"from zero J = {scratch[k]:8.4f}")
print("learned correction to the nominal gains:", residual_gains - nominal)


def iterations_needed(history, tolerance=0.01):
    return next(k for k, J in enumerate(history) if J > J_best - tolerance)


print("iterations to come within 0.01 of the best: residual",
      iterations_needed(residual), ", from zero", iterations_needed(scratch))
assert np.allclose(residual_gains, best, atol=1e-2)
assert iterations_needed(residual) < iterations_needed(scratch)
assert residual[0] > scratch[0]
