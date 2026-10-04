# %% [markdown]
# # Chapter 26 examples: robotics case studies
#
# This notebook runs the two small illustrations of
# [Chapter 26](https://thduynguyen.github.io/gtsam/chapter26) on the line of
# Chapter 1, with the `gtsam/semiring` module: a policy trained on a range of
# simulated models (domain randomization), and a learned correction on top of
# a nominal controller (residual learning). Both are exact: no sampling is
# involved. Every expected return is one elimination of a Gaussian
# `SemiringFactorGraph`, and the best gains come from the same graph with the
# maximum at the controls.

# %%
import numpy as np
from gtsam import HessianFactor, JacobianFactor, Ordering, noiseModel
from gtsam import SemiringFactorGraph, SemiringGaussianFactor
from gtsam import SemiringRules, SemiringSum
from gtsam.symbol_shorthand import U, X
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line, with an actuator gain $B$ (Section 2)
#
# $x' = x + B u + w$, $w \sim N(0, 0.5)$, rewards $-(x^2 + u^2)$ per move and
# $-x_2^2$ at the end, $x_0 \sim N(2, 1)$, two moves. Chapter 1 had $B = 1$.
# The policy is a gain per move, $u_t = -K_t x_t$, which enters the graph as a
# hard constraint $u_t + K_t x_t = 0$.

# %%
sigma_w, mean0, variance0 = 0.5, 2.0, 1.0
I, zero = np.eye(1), np.zeros(1)


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def ordering(*keys):
    """An ordering of the given keys."""
    result = Ordering()
    for key in keys:
        result.push_back(key)
    return result


def line(B, gains=None):
    """The factor graph of the line, with or without policy factors."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([mean0]),
                             noiseModel.Isotropic.Variance(1, variance0)))
    for t in range(2):
        if gains is not None:
            graph.push_back(gaussian(U(t), I, X(t), gains[t] * I, zero,
                                     noiseModel.Constrained.All(1)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -B * I, zero,
                                 noiseModel.Isotropic.Variance(1, sigma_w)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph


backward = ordering(X(2), U(1), X(1), U(0), X(0))


def evaluate(gains, B):
    """Expected return of u_t = -K_t x_t on the line with actuator gain B."""
    return line(B, gains).expectation(backward)


# %% [markdown]
# The best gains: the same graph without policy factors, with the controls
# eliminated by maximum. The conditional that the elimination leaves on each
# control is the best law $u_t + K_t x_t = 0$. This is the Riccati recursion
# of Chapter 4.

# %%
controls = SemiringRules()
controls.setAll([U(0), U(1)], SemiringSum.Maximum())


def riccati(B):
    """The best gains for actuator gain B, and their expected return."""
    bayes_net = line(B).eliminateSequential(backward, controls)
    # The conditionals of u1 and u0 are at positions 1 and 3.
    gains = [bayes_net.at(position).conditional().S()[0, 0]
             for position in (3, 1)]
    return np.array(gains), line(B).expectation(backward, controls)


nominal, J_nominal = riccati(1.0)
print("Riccati gains for B = 1:", nominal, " J =", J_nominal)
assert np.allclose(nominal, [0.6, 0.5]) and np.isclose(J_nominal, -9.25)
assert np.isclose(evaluate(nominal, 1.0), -9.25)
assert np.isclose(evaluate([0.5, 0.5], 1.0), -9.375)

# %% [markdown]
# A check of the module against the two recursions in closed form.


# %%
def evaluate_by_recursion(gains, B):
    """V_t(x) = -(P x^2 + beta) for a fixed linear policy, backward."""
    P, beta = 1.0, 0.0  # V_2(x) = -x^2
    for K in reversed(gains):
        beta = beta + P * sigma_w
        P = 1 + K ** 2 + (1 - B * K) ** 2 * P
    return -(P * (mean0 ** 2 + variance0) + beta)


def riccati_by_recursion(B):
    P, gains = 1.0, []
    for _ in range(2):
        K = B * P / (1 + B ** 2 * P)
        gains.insert(0, K)
        P = 1 + P - (B * P) ** 2 / (1 + B ** 2 * P)
    return np.array(gains)


for B in [0.5, 1.0, 1.5]:
    gains, J = riccati(B)
    print(f"B = {B}: module {evaluate(nominal, B):.4f}, "
          f"recursion {evaluate_by_recursion(nominal, B):.4f}, "
          f"best gains {gains}")
    assert np.isclose(evaluate(nominal, B), evaluate_by_recursion(nominal, B))
    assert np.allclose(gains, riccati_by_recursion(B))
    assert np.isclose(J, evaluate(gains, B))

# %% [markdown]
# ## Domain randomization (Section 2)
#
# The simulator's actuator gain is uncertain: 0.5, 1 or 1.5, equally likely.
# One policy is trained on the average over the three models, and compared
# with the policy designed for $B = 1$ alone and with the best policy of each
# model. The objective is three eliminations, one per model; the search over
# the two gains is scipy's.

# %%
models = [0.5, 1.0, 1.5]


def randomized_objective(gains):
    return -np.mean([evaluate(gains, B) for B in models])


randomized = minimize(randomized_objective, nominal).x
print("gains trained on B = 1 only:        ", nominal)
print("gains trained on the three models:  ", randomized)
table = {}
for B in models:
    table[B] = (evaluate(nominal, B), evaluate(randomized, B), riccati(B)[1])
    print(f"B = {B}: nominal {table[B][0]:8.4f}, randomized "
          f"{table[B][1]:8.4f}, best for this B {table[B][2]:8.4f}")
print("average over the models: nominal",
      np.mean([row[0] for row in table.values()]), " randomized",
      np.mean([row[1] for row in table.values()]))
assert -randomized_objective(randomized) >= -randomized_objective(nominal)
assert all(row[2] >= max(row[0], row[1]) - 1e-9 for row in table.values())
assert np.allclose(randomized, [0.553, 0.448], atol=5e-4)
assert np.allclose([table[B] for B in models],
                   [[-12.647, -12.649, -12.607], [-9.250, -9.286, -9.250],
                    [-8.022, -7.874, -7.812]], atol=5e-4)
assert np.isclose(-randomized_objective(nominal), -9.973, atol=5e-4)
assert np.isclose(-randomized_objective(randomized), -9.936, atol=5e-4)

# %% [markdown]
# ## Residual learning (Section 5)
#
# The real robot has $B = 0.5$. The nominal controller was designed for
# $B = 1$. A correction to its gains is learned on the real system by
# gradient ascent, and compared with learning the gains from zero with the
# same step size. Every evaluation of $J$ is one elimination.

# %%
B_real = 0.5
best, J_best = riccati(B_real)
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
assert np.allclose(best, [0.621, 0.4], atol=5e-4)
assert np.isclose(J_best, -12.607, atol=5e-4)
assert np.isclose(evaluate([0, 0], B_real), -16.5)
assert np.allclose([residual[k] for k in [0, 1, 2, 5, 10, 20]],
                   [-12.647, -12.635, -12.627, -12.614, -12.609, -12.607],
                   atol=5e-4)
assert np.allclose([scratch[k] for k in [0, 1, 2, 5, 10, 20]],
                   [-16.5, -14.368, -13.463, -12.735, -12.619, -12.607],
                   atol=5e-4)
assert iterations_needed(residual) == 5 and iterations_needed(scratch) == 11
