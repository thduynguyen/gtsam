# %% [markdown]
# # Chapter 20 examples: guided policy search
#
# This notebook runs the examples of
# [Chapter 20](https://thduynguyen.github.io/gtsam/chapter20). Several local
# trajectory problems on the line, one per start position, are made to agree
# with one global policy. The agreement is a factor with a dual variable, and
# each local problem is a semiring factor graph, eliminated with the
# `gtsam/semiring` module: the maximum at the actions, the average at the
# states.

# %%
import numpy as np
from gtsam import GaussianBayesNet, GaussianFactorGraph, JacobianFactor
from gtsam import Ordering, SemiringFactorGraph, SemiringGaussianFactor
from gtsam import SemiringRules, SemiringSum, noiseModel
from gtsam.symbol_shorthand import U, X
from scipy.optimize import minimize_scalar

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The local problem as a factor graph (Section 2)
#
# One start position, two moves on the line, $x' = F x + B u$ for the mean
# trajectory. The unknowns are $u_0, x_1, u_1, x_2$. The factors are the
# dynamics (hard constraints), the rewards (quadratic), and one *agreement
# factor* per move, which pulls the action toward the global policy
# $u = -K_t x$ and carries the dual variable $\nu_t$.

# %%
I = np.eye(1)
zero = np.zeros(1)
STARTS = [1.0, 2.0, 3.0]
MOVES = 2
hard = noiseModel.Constrained.All(1)
penalty = noiseModel.Isotropic.Sigma(1, 1 / np.sqrt(2))  # error z^2

BACKWARD = Ordering()
for key in [X(2), U(1), X(1), U(0), X(0)]:
    BACKWARD.push_back(key)
MAXIMUM_AT_ACTIONS = SemiringRules()
MAXIMUM_AT_ACTIONS.setAll([U(0), U(1)], SemiringSum.Maximum())


def constraint(*args):
    """Lift a Gaussian factor, here a hard constraint, to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def cost(*args):
    """Lift a quadratic, given as a JacobianFactor, to the reward (1, -error)."""
    return SemiringGaussianFactor.Cost(JacobianFactor(*args))


def local_factors(start, gains, duals, weight, F=1.0, B=1.0):
    """The factors of one local problem: constraints, then quadratics."""
    constraints = [(X(0), I, np.array([start]), hard)]
    quadratics = []
    for t in range(MOVES):
        # Dynamics of the mean trajectory: x' - F x - B u = 0.
        constraints.append(
            (X(t + 1), I, X(t), -F * I, U(t), -B * I, zero, hard))
        quadratics.append((X(t), I, zero, penalty))  # reward -x^2
        quadratics.append((U(t), I, zero, penalty))  # reward -u^2
        if weight > 0:
            # Agreement: (weight / 2) (u + K x + nu / weight)^2.
            quadratics.append(
                (U(t), I, X(t), gains[t] * I, np.array([-duals[t] / weight]),
                 noiseModel.Isotropic.Sigma(1, 1 / np.sqrt(weight))))
    quadratics.append((X(MOVES), I, zero, penalty))  # reward -x_2^2
    return constraints, quadratics


def local_graph(start, gains, duals, weight, F=1.0, B=1.0):
    """One local problem as a semiring factor graph, without a policy."""
    constraints, quadratics = local_factors(start, gains, duals, weight, F, B)
    graph = SemiringFactorGraph()
    for factor in constraints:
        graph.push_back(constraint(*factor))
    for factor in quadratics:
        graph.push_back(cost(*factor))
    return graph


def local_step(start, gains, duals, weight, F=1.0, B=1.0):
    """Best trajectory from one start, given the policy and the duals.

    Maximizes  R(tau) - sum_t [nu_t h_t + (weight / 2) h_t^2],
    with h_t = u_t + K_t x_t. The backward pass eliminates the states by
    average and the actions by maximum; the forward pass is the
    back-substitution of the conditionals it leaves.
    Returns the positions x_0, x_1 and the actions u_0, u_1.
    """
    graph = local_graph(start, gains, duals, weight, F, B)
    bayes_net = graph.eliminateSequential(BACKWARD, MAXIMUM_AT_ACTIONS)
    conditionals = GaussianBayesNet()
    for i in range(bayes_net.size()):
        conditionals.push_back(bayes_net.at(i).conditional())
    solution = conditionals.optimize()
    positions = np.array([solution.at(X(t))[0] for t in range(MOVES)])
    actions = np.array([solution.at(U(t))[0] for t in range(MOVES)])
    return positions, actions


# Without the agreement factors, each local solution is the best trajectory
# from its start. Its actions follow the Riccati gains of Chapter 6.
for start in STARTS:
    positions, actions = local_step(start, [0, 0], [0, 0], weight=0)
    print(f"start {start}: positions {positions}, actions {actions}, "
          f"-u/x = {-actions / positions}")
    assert np.allclose(-actions / positions, [0.6, 0.5])

# %% [markdown]
# The dynamics are hard constraints, so there is no luck to average over, and
# eliminating the states by maximum gives the same trajectory. The local
# problem is then an ordinary `GaussianFactorGraph`, a linear least-squares
# problem.

# %%
gains, duals = [0.3, 0.7], [0.4, -0.2]
constraints, quadratics = local_factors(2.0, gains, duals, weight=2.0)
least_squares = GaussianFactorGraph()
for factor in constraints + quadratics:
    least_squares.add(JacobianFactor(*factor))
solution = least_squares.optimize()
positions, actions = local_step(2.0, gains, duals, weight=2.0)
assert np.allclose(positions, [solution.at(X(t))[0] for t in range(MOVES)])
assert np.allclose(actions, [solution.at(U(t))[0] for t in range(MOVES)])
print("local solution, by the module:      ", positions, actions)
print("the same by joint least squares: OK")

# %% [markdown]
# ## The alternation (Section 3)
#
# Local step for every start; then the global step, a regression of the
# actions on the positions; then the dual step.


# %%
def guided_policy_search(stationary, weight, iterations, models=None):
    """ADMM on the agreement u_t = -K_t x_t. Returns the history."""
    gains = np.zeros(MOVES)
    duals = np.zeros((len(STARTS), MOVES))
    history = []
    for k in range(iterations):
        # Local step (Stage 1): one trajectory problem per start.
        positions, actions = [], []
        for i, start in enumerate(STARTS):
            F, B = (1.0, 1.0) if models is None else models(i, k)
            x, u = local_step(start, gains, duals[i], weight, F, B)
            positions.append(x), actions.append(u)
        positions, actions = np.array(positions), np.array(actions)
        # Global step (Stage 2): fit the policy to the local actions.
        targets = actions + duals / weight
        if stationary:  # one gain for both moves
            gains[:] = -(positions * targets).sum() / (positions ** 2).sum()
        else:  # one gain per move
            gains = -(positions * targets).sum(axis=0) / \
                (positions ** 2).sum(axis=0)
        # Dual step: raise the price of the remaining disagreement.
        disagreement = actions + gains * positions
        duals = duals + weight * disagreement
        history.append((gains.copy(), np.abs(disagreement).max(),
                        duals.copy(), positions, actions))
    return history


# %% [markdown]
# ## An exact special case: a gain per move (Section 4)
#
# With a linear system and a policy with one gain per move, the method must
# converge to the Riccati gains $K_0 = 0.6$, $K_1 = 0.5$.

# %%
history = guided_policy_search(stationary=False, weight=2.0, iterations=30)
for k in [0, 1, 2, 5, 10, 20, 29]:
    gains, disagreement = history[k][:2]
    print(f"iteration {k:2d}: K = {gains}, largest disagreement = "
          f"{disagreement:.1e}")
assert np.allclose(history[-1][0], [0.6, 0.5], atol=1e-6)

# %% [markdown]
# ## A policy that cannot follow the local solutions: one gain for both moves (Section 3)
#
# Now the global policy has a single gain. The local solutions want $0.6$ and
# then $0.5$, which no single gain reproduces. The dual variables reshape the
# local problems until their solutions are something the policy can do.


# %%
def policy_factor(t, gain):
    """The deterministic policy u = -gain * x, as a hard constraint."""
    return constraint(U(t), I, X(t), gain * I, zero, hard)


def stationary_cost(gain):
    """Total cost of u = -gain * x from the three starts, without noise.

    For each start, the local graph with the policy as a factor is eliminated
    by average; its expectation is minus the cost.
    """
    total = 0.0
    for start in STARTS:
        graph = local_graph(start, [0, 0], [0, 0], weight=0)
        for t in range(MOVES):
            graph.push_back(policy_factor(t, gain))
        total -= graph.expectation(BACKWARD)
    return total


best = minimize_scalar(stationary_cost, bounds=(0, 1), method="bounded",
                       options={"xatol": 1e-10})
print(f"best single gain, by direct search: K = {best.x:.4f}, "
      f"cost = {best.fun:.4f}")

history = guided_policy_search(stationary=True, weight=2.0, iterations=60)
for k in [0, 1, 2, 5, 10, 20, 40, 59]:
    gains, disagreement = history[k][:2]
    print(f"iteration {k:2d}: K = {gains[0]:.4f}, largest disagreement = "
          f"{disagreement:.5f}")
final_gain = history[-1][0][0]
assert abs(final_gain - best.x) < 1e-4 and history[-1][1] < 1e-6
print("dual variables at the end (rows: starts 1, 2, 3; columns: moves):")
print(history[-1][2])
print("local trajectories at the end: positions\n", history[-1][3],
      "\nactions\n", history[-1][4])
# The value left at the root of a free local problem is minus its cost.
unconstrained = -sum(
    local_graph(start, [0, 0], [0, 0], weight=0).expectation(
        BACKWARD, MAXIMUM_AT_ACTIONS) for start in STARTS)
print(f"cost of the three free local solutions: {unconstrained:.4f}")
print(f"cost with one shared gain:              {best.fun:.4f}")

# %% [markdown]
# ## What each policy is worth on the real, noisy line (Section 4)
#
# Evaluated exactly, for $x_0 \sim N(2, 1)$ and wheel slip of variance $0.5$.


# %%
def true_return(gains):
    """J of u_t = -K_t x_t on the true line, by elimination."""
    graph = SemiringFactorGraph()
    graph.push_back(constraint(X(0), I, np.array([2.0]),
                               noiseModel.Isotropic.Variance(1, 1.0)))
    for t in range(MOVES):
        graph.push_back(policy_factor(t, gains[t]))
        graph.push_back(constraint(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                   noiseModel.Isotropic.Variance(1, 0.5)))
        graph.push_back(cost(X(t), I, zero, penalty))
        graph.push_back(cost(U(t), I, zero, penalty))
    graph.push_back(cost(X(MOVES), I, zero, penalty))
    return graph.expectation(BACKWARD)


print("a gain per move, K = (0.6, 0.5):  J =", true_return([0.6, 0.5]))
print(f"one gain, K = {final_gain:.4f}:           J =",
      true_return([final_gain, final_gain]))
assert np.isclose(true_return([0.6, 0.5]), -9.25)
assert np.isclose(true_return([0.5, 0.5]), -9.375)

# %% [markdown]
# ## Local models fitted to rollouts (Section 5)
#
# The local step so far used the true $F = B = 1$. Guided policy search does
# not know them. Each local problem runs a few rollouts on the real, noisy
# system around its current trajectory, fits its own linear model to them by
# least squares (Chapter 18), and plans with that model.

# %%
rng = np.random.default_rng(0)
data = [([], [], []) for _ in STARTS]  # per start: x, u, x'
plans = [np.zeros(MOVES) for _ in STARTS]
fitted = {}


def fitted_models(i, k):
    """Roll out the current plan of start i with exploration; refit F, B."""
    xs, us, ns = data[i]
    for rollout in range(5):
        x = STARTS[i]
        for t in range(MOVES):
            u = plans[i][t] + 0.5 * rng.normal()
            n = x + u + np.sqrt(0.5) * rng.normal()  # the real system
            xs.append(x), us.append(u), ns.append(n)
            x = n
    Z = np.stack([xs, us], axis=1)
    fitted[i] = np.linalg.solve(Z.T @ Z, Z.T @ np.array(ns))
    return fitted[i]


def guided_policy_search_fitted(weight, iterations):
    gains = np.zeros(MOVES)
    duals = np.zeros((len(STARTS), MOVES))
    for k in range(iterations):
        positions, actions = [], []
        for i, start in enumerate(STARTS):
            F, B = fitted_models(i, k)
            x, u = local_step(start, gains, duals[i], weight, F, B)
            plans[i] = u
            positions.append(x), actions.append(u)
        positions, actions = np.array(positions), np.array(actions)
        targets = actions + duals / weight
        gains = -(positions * targets).sum(axis=0) / \
            (positions ** 2).sum(axis=0)
        duals = duals + weight * (actions + gains * positions)
    return gains


gains = guided_policy_search_fitted(weight=2.0, iterations=30)
print("transitions used per local problem:", len(data[0][0]))
print("fitted local models (F, B):")
for i, start in enumerate(STARTS):
    print(f"  start {start}: {fitted[i]}")
print("gains with fitted local models:", gains)
print("true J of that policy:", true_return(gains), " (best: -9.25)")
assert np.allclose(gains, [0.6, 0.5], atol=0.1)
assert true_return(gains) > -9.3
