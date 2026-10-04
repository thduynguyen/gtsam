# %% [markdown]
# # Chapter 7 examples: policy optimization with exact messages
#
# This notebook runs the examples of
# [Chapter 7](https://thduynguyen.github.io/gtsam/chapter07) on the robot on a
# line: a linear-Gaussian policy is evaluated by elimination (Stage 1), and
# its gain is improved by gradient, natural-gradient, damped and trust-region
# steps (Stage 2), the last also with GTSAM's own optimizers.

# %%
import numpy as np
import gtsam
from gtsam import HessianFactor, JacobianFactor, Ordering
from gtsam import SemiringFactorGraph, SemiringGaussianFactor, noiseModel
from gtsam.symbol_shorthand import U, X
from scipy.optimize import minimize_scalar

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line, with a linear-Gaussian policy (Section 1)
#
# $x' = x + u + w$, $w \sim N(0, 0.5)$; reward $-(x^2 + u^2)$ per move and
# $-x_2^2$ at the end; $x_0 \sim N(2, 1)$. The policy is
# $u = -K_t\, x + e$ with $e \sim N(0, \Sigma_e)$ and $\Sigma_e = 0.1$.

# %%
I, zero = np.eye(1), np.zeros(1)
Sigma_w, Sigma_e, mu_0, Sigma_0 = 0.5, 0.1, 2.0, 1.0


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


def build(K, policy_variance=Sigma_e):
    """The semiring factor graph for the gains K = (K_0, K_1)."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([mu_0]), variance(Sigma_0)))
    for t in range(2):
        model = (variance(policy_variance) if policy_variance > 0
                 else noiseModel.Constrained.All(1))
        graph.push_back(gaussian(U(t), I, X(t), K[t] * I, zero, model))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 variance(Sigma_w)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph


print("J for K = (0.5, 0.5):", build([0.5, 0.5]).expectation())
assert np.isclose(build([0.5, 0.5]).expectation(), -9.825)

# %% [markdown]
# ## Stage 1: backward and forward messages by elimination (Section 2)
#
# The backward message is read from the conditional of each action: its
# surprise is the advantage $A_t(x, u)$, a quadratic whose blocks are
# $H_{uu}$ and $H_{ux}$. The forward message is the marginal of each state, a
# Gaussian with a mean and a variance.


# %%
def stage1(K):
    """J, the blocks (H_uu, H_ux) of each advantage, and E[x_t^2]."""
    graph = build(K)
    bayes_net = graph.eliminateSequential(
        ordering(X(2), U(1), X(1), U(0), X(0)))
    blocks = {}
    for t, position in [(1, 1), (0, 3)]:  # the conditionals of u_1 and u_0
        surprise = bayes_net.at(position).surprise()
        assert list(surprise.keys()) == [U(t), X(t)]
        info = surprise.augmentedInformation()  # A = z' (info / 2) z
        blocks[t] = (-0.5 * info[0, 0], -0.5 * info[0, 1])  # H_uu, H_ux
    bayes_tree = graph.eliminateMultifrontal()
    second_moment = {}
    for t in range(2):
        marginal = bayes_tree.marginalFactor(X(t)).conditional()
        mean = (marginal.d() / marginal.R()).item()
        var = (1 / marginal.R() ** 2).item()
        second_moment[t] = mean ** 2 + var
    return graph.expectation(), blocks, second_moment


J, blocks, second_moment = stage1([0.5, 0.5])
for t in range(2):
    print(f"t = {t}: H_uu = {blocks[t][0]:.4f}, H_ux = {blocks[t][1]:.4f}, "
          f"E[x^2] = {second_moment[t]:.4f}")
assert np.allclose(blocks[0], (2.5, 1.5)) and np.allclose(blocks[1], (2, 1))
assert np.isclose(second_moment[0], 5.0)
assert np.isclose(second_moment[1], 1.85)

# %% [markdown]
# The same messages from the Lyapunov recursion, in numpy.


# %%
def lyapunov(K):
    """P_t and beta_t of V_t(x) = -(P_t x^2 + beta_t), and E[x_t^2]."""
    P, beta = {2: 1.0}, {2: 0.0}
    for t in [1, 0]:
        P[t] = 1 + K[t] ** 2 + (1 - K[t]) ** 2 * P[t + 1]
        beta[t] = beta[t + 1] + P[t + 1] * Sigma_w + (1 + P[t + 1]) * Sigma_e
    m = {0: mu_0 ** 2 + Sigma_0}
    for t in range(2):
        m[t + 1] = (1 - K[t]) ** 2 * m[t] + Sigma_e + Sigma_w
    return P, beta, m


def J_formula(K):
    P, beta, m = lyapunov(K)
    return -(P[0] * m[0] + beta[0])


P, beta, m = lyapunov([0.5, 0.5])
print("P =", P, "\nbeta =", beta, "\nE[x^2] =", m)
assert np.isclose(P[1], 1.5) and np.isclose(P[0], 1.625)
assert np.isclose(beta[1], 0.7) and np.isclose(beta[0], 1.7)
assert np.isclose(J_formula([0.5, 0.5]), -9.825)

# %% [markdown]
# ## The gradient: forward message times local term times backward message (Section 3)
#
# $\partial J / \partial K_t = 2\, (H_{ux} - H_{uu} K_t)\; \mathbb{E}[x_t^2]$.


# %%
def gradient(K, blocks, second_moment):
    return np.array([2 * (blocks[t][1] - blocks[t][0] * K[t]) *
                     second_moment[t] for t in range(2)])


def finite_differences(K, h=1e-5):
    K = np.asarray(K, float)
    return np.array([(build(K + h * e).expectation() -
                      build(K - h * e).expectation()) / (2 * h)
                     for e in np.eye(2)])


for K in [[0.5, 0.5], [0.0, 0.0], [0.9, 0.2]]:
    J, blocks, second_moment = stage1(K)
    g = gradient(K, blocks, second_moment)
    numeric = finite_differences(K)
    print(f"K = {K}: J = {J:.4f}, gradient = {g}, finite differences = "
          f"{numeric}")
    assert np.allclose(g, numeric, atol=1e-5)
J, blocks, second_moment = stage1([0.5, 0.5])
assert np.allclose(gradient([0.5, 0.5], blocks, second_moment), [2.5, 0.0])

# %% [markdown]
# The Fisher matrix of the Gaussian policy is diagonal, with entries
# $\mathbb{E}[x_t^2] / \Sigma_e$. A Monte Carlo check of both it and the
# gradient, as the averages $\mathbb{E}[g\, g^\top]$ and $\mathbb{E}[g\, R]$
# of Chapter 5:

# %%
rng = np.random.default_rng(0)
samples = 1_000_000
K = np.array([0.5, 0.5])
x = mu_0 + rng.normal(size=samples)
score, ret = np.zeros((samples, 2)), np.zeros(samples)
for t in range(2):
    e = np.sqrt(Sigma_e) * rng.normal(size=samples)
    u = -K[t] * x + e
    score[:, t] = -e * x / Sigma_e  # d log pi / d K_t
    ret -= x ** 2 + u ** 2
    x = x + u + np.sqrt(Sigma_w) * rng.normal(size=samples)
ret -= x ** 2
fisher_mc = score.T @ score / samples
gradient_mc = (score * (ret - ret.mean())[:, None]).mean(axis=0)
print("Monte Carlo Fisher matrix =\n", fisher_mc)
print("exact diagonal:", [second_moment[t] / Sigma_e for t in range(2)])
print("Monte Carlo gradient:", gradient_mc)
assert np.allclose(np.diag(fisher_mc), [50.0, 18.5], rtol=0.02)
assert abs(fisher_mc[0, 1]) < 0.3
assert np.allclose(gradient_mc, [2.5, 0.0], atol=0.3)

# %% [markdown]
# ## Stage 2: four updates (Section 4)
#
# Each iteration is one elimination (Stage 1) and one update of the gains.


# %%
def fisher(second_moment):
    return np.diag([second_moment[t] / Sigma_e for t in range(2)])


def run(update, iterations, K=(0.0, 0.0)):
    K = np.array(K, float)
    history = []
    for _ in range(iterations + 1):
        J, blocks, second_moment = stage1(K)  # stage 1
        history.append((J, K.copy()))
        g = gradient(K, blocks, second_moment)
        K = K + update(K, J, g, fisher(second_moment))  # stage 2
    return history


def gradient_step(K, J, g, F, alpha=0.03):
    return alpha * g


def natural_step(K, J, g, F, alpha=1.0):
    return alpha * np.linalg.solve(F, g)


def damped_step(K, J, g, F, damping=20.0):
    return np.linalg.solve(F + damping * np.eye(2), g)


def trust_region_step(K, J, g, F, D_max=0.5):
    """The largest step along the natural gradient with KL <= D_max,
    halved until the expected return improves."""
    direction = np.linalg.solve(F, g)
    step = np.sqrt(2 * D_max / (g @ direction)) * direction
    while build(K + step).expectation() < J and np.abs(step).max() > 1e-12:
        step = step / 2
    return step


histories = {
    "gradient": run(gradient_step, 40),
    "natural gradient": run(natural_step, 40),
    "damped (LM)": run(damped_step, 40),
    "trust region": run(trust_region_step, 40),
}
for k in [0, 1, 2, 5, 10, 20, 40]:
    print(f"iteration {k:2d}: " + "  ".join(
        f"{name} {history[k][0]:.4f}" for name, history in histories.items()))
for name, history in histories.items():
    print(f"{name:17s} final K = {history[-1][1]}, J = {history[-1][0]:.5f}")
    assert np.allclose(history[-1][1], [0.6, 0.5], atol=2e-3)

# %% [markdown]
# The step size of the plain gradient must be tuned: with 0.1 in place of
# 0.03 the iteration diverges.

# %%
K, diverging = np.zeros(2), []
for _ in range(6):  # with the Lyapunov formulas, which tolerate huge gains
    P, beta, m = lyapunov(K)
    diverging.append(J_formula(K))
    K = K + 0.1 * np.array([2 * (P[t + 1] - (1 + P[t + 1]) * K[t]) * m[t]
                            for t in range(2)])
print("J with gradient steps of size 0.1:", [f"{J:.3g}" for J in diverging])
assert diverging[-1] < diverging[0]

# %% [markdown]
# Dividing the natural gradient once more, by $2\, \Sigma_e H_{uu}$, jumps
# to the greedy gain $K^+_t = H_{uu}^{-1} H_{ux}$. That is policy iteration:
# it reaches the Riccati gains in two sweeps.

# %%
K = np.array([0.0, 0.0])
for sweep in range(3):
    J, blocks, second_moment = stage1(K)
    print(f"sweep {sweep}: K = {K}, J = {J:.4f}")
    K = np.array([blocks[t][1] / blocks[t][0] for t in range(2)])  # K^+
assert np.allclose(K, [0.6, 0.5])

# %% [markdown]
# ## The exact special case (Section 5)
#
# With one gain per move, the iteration must reach the Riccati gains
# $0.6$ and $0.5$ of Chapter 6. With the jitter $\Sigma_e = 0.1$ the return
# there is $J^* - \Sigma_e (H_{uu,0} + H_{uu,1}) = -9.25 - 0.45$; without the
# jitter it is $J^* = -9.25$.

# %%
K_final = histories["natural gradient"][-1][1]
print("gains:", K_final)
print("J with jitter:   ", build(K_final).expectation())
print("J without jitter:", build(K_final, policy_variance=0).expectation())
assert np.isclose(build([0.6, 0.5]).expectation(), -9.70)
assert np.isclose(build([0.6, 0.5], policy_variance=0).expectation(), -9.25)

# %% [markdown]
# A shared gain: one $K$ for both moves. Its gradient is the sum of the two
# per-step gradients. The best shared gain is computed independently, by a
# scalar search on the formula for $J$.

# %%
best = minimize_scalar(lambda k: -J_formula([k, k]), bounds=(0, 1),
                       method="bounded", options={"xatol": 1e-10})
print(f"best shared gain by scalar search: K = {best.x:.5f}, "
      f"J = {-best.fun:.5f}")

k = 0.0
for iteration in range(31):
    J, blocks, second_moment = stage1([k, k])  # stage 1
    g = gradient([k, k], blocks, second_moment).sum()
    if iteration in [0, 1, 2, 5, 10, 30]:
        print(f"iteration {iteration:2d}: K = {k:.5f}, J = {J:.5f}")
    k = k + g / (sum(second_moment.values()) / Sigma_e)  # natural gradient
assert np.isclose(k, best.x, atol=1e-4)
assert np.isclose(best.x, 0.58281, atol=1e-4)
assert np.isclose(-best.fun, -9.72386, atol=1e-4)

# %% [markdown]
# ## Stage 2 with GTSAM's optimizers (Section 6)
#
# GTSAM's nonlinear optimizers solve, at each iteration, the linear system
# $(W^\top W + \text{damping})\, \Delta = -W^\top h$ for a factor with residual
# $h$ and Jacobian $W$. A custom factor with $W = \mathcal{I}^{1/2}$ and
# $h = -\mathcal{I}^{-1/2}\, \nabla J$ makes that system the natural-gradient
# system $\mathcal{I}\, \Delta = \nabla J$. Every evaluation of the factor
# runs Stage 1.

# %%
KEY = gtsam.symbol("k", 0)
evaluations = []


def natural_gradient_factor(this, values, H):
    K = values.atVector(KEY)
    J, blocks, second_moment = stage1(K)  # stage 1
    evaluations.append(J)
    root = np.sqrt(np.diag(fisher(second_moment)))  # I^(1/2), diagonal
    if H is not None:
        H[0] = np.diag(root)
    return -gradient(K, blocks, second_moment) / root


def optimize(optimizer_class, params):
    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.CustomFactor(noiseModel.Unit.Create(2), [KEY],
                                 natural_gradient_factor))
    initial = gtsam.Values()
    initial.insert(KEY, np.zeros(2))
    evaluations.clear()
    optimizer = optimizer_class(graph, initial, params)
    result = optimizer.optimize().atVector(KEY)
    return result, optimizer.iterations()


for name, optimizer_class, params in [
        ("Gauss-Newton", gtsam.GaussNewtonOptimizer,
         gtsam.GaussNewtonParams()),
        ("Levenberg-Marquardt", gtsam.LevenbergMarquardtOptimizer,
         gtsam.LevenbergMarquardtParams()),
        ("Dogleg", gtsam.DoglegOptimizer, gtsam.DoglegParams())]:
    # Stop only when the natural gradient has vanished.
    params.setRelativeErrorTol(0.0)
    params.setAbsoluteErrorTol(0.0)
    params.setErrorTol(1e-12)
    params.setMaxIterations(100)
    result, iterations = optimize(optimizer_class, params)
    print(f"{name:20s} K = {result}, J = {build(result).expectation():.5f}, "
          f"{iterations} iterations, {len(evaluations)} eliminations")
    assert np.allclose(result, [0.6, 0.5], atol=1e-3)
