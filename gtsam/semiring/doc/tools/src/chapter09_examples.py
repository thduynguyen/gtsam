# %% [markdown]
# # Chapter 9 examples: nonlinear dynamics
#
# This notebook runs the examples of
# [Chapter 9](https://thduynguyen.github.io/gtsam/chapter09): iLQR and DDP on
# a robot with a weak motor, with the backward pass done by the
# `gtsam/semiring` module; the same problem as a MAP trajectory optimization
# with GTSAM's Levenberg-Marquardt optimizer; and model predictive control
# under noise. The rollouts of the nonlinear robot are plain numpy.

# %%
import numpy as np
from scipy.optimize import least_squares, minimize

import gtsam
from gtsam import CustomFactor, GaussianFactorGraph, HessianFactor
from gtsam import JacobianFactor, Ordering, SemiringFactorGraph
from gtsam import SemiringGaussianFactor, SemiringRules, SemiringSum
from gtsam import VectorValues, noiseModel
from gtsam.symbol_shorthand import U, X

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The weak motor (Section 1)
#
# The robot of the line example, with a motor that saturates: a command $u$
# moves it by $\tanh(u)$, never more than 1. It starts at $x_0 = 3$ and makes
# four moves. The rewards are those of the line.

# %%
T = 4  # moves
x_start = 3.0


def f(x, u):
    """The mean of the dynamics: x' = f(x, u) + noise."""
    return x + np.tanh(u)


def f_u(u):
    """The derivative of f with respect to u; the one with respect to x is 1."""
    return 1 - np.tanh(u) ** 2


def rollout(controls, x0):
    """The states visited when the controls are applied without noise."""
    states = [x0]
    for u in controls:
        states.append(f(states[-1], u))
    return np.array(states)


def total_reward(controls, x0):
    """The return of the noise-free rollout."""
    states = rollout(controls, x0)
    return -(np.sum(states[:-1] ** 2) + np.sum(np.square(controls))
             + states[-1] ** 2)


print("doing nothing:", total_reward(np.zeros(T), x_start))
assert np.isclose(total_reward(np.zeros(T), x_start), -45.0)

# %% [markdown]
# ## Stage 1 with the module: linearize, then eliminate (Section 2)
#
# Around the current trajectory the dynamics factor is replaced by its
# linearization, a linear-Gaussian factor. The rewards are already quadratic.
# The backward pass is then the elimination of Chapter 6: the next state is
# summed out by the average and the action by the maximum. The conditional
# that the maximum leaves on an action is the local policy.

# %%
I = np.eye(1)
zero = np.zeros(1)
maximum = SemiringSum.Maximum()


def noise_model(variance):
    """Gaussian noise of the given variance; a hard constraint if it is 0."""
    if variance == 0:
        return noiseModel.Constrained.All(1)
    return noiseModel.Isotropic.Variance(1, variance)


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


def linearized(t, states, controls, noise):
    """The dynamics factor of move t, linearized around the trajectory.

    x' = x + B u + constant, with B = df/du at the current control.
    """
    B = f_u(controls[t])
    constant = states[t + 1] - states[t] - B * controls[t]
    return gaussian(X(t + 1), I, X(t), -I, U(t), -B * I,
                    np.array([constant]), noise_model(noise))


def stage1(states, controls, noise=0.0, ddp=False):
    """One backward pass on the graph linearized around a trajectory.

    Returns the gains K_t and offsets o_t of the best local policy
    u = -K_t x + o_t, the value factor (1, V_0) left on the first state, and
    the curvatures P_t of the values V_t.
    """
    moves = len(controls)
    gains, offsets = np.zeros(moves), np.zeros(moves)
    curvatures = np.ones(moves + 1)
    value = penalty(X(moves))  # (1, V_T)
    for t in reversed(range(moves)):
        # Eliminate the next state by average.
        dynamics = linearized(t, states, controls, noise)
        phi = dynamics.multiply(value).sum(ordering(X(t + 1)))
        bucket = penalty(X(t)).multiply(penalty(U(t))).multiply(phi)
        if ddp:  # DDP adds the curvature of the dynamics, as one more factor
            bucket = bucket.multiply(
                dynamics_curvature(t, value, states, controls))
        # Eliminate the action by maximum. Its conditional is u + K x = o.
        conditional, value = bucket.eliminate(ordering(U(t)), maximum)
        policy = conditional.conditional()
        gains[t], offsets[t] = policy.S()[0, 0], policy.d()[0]
        curvatures[t] = -value.value().information()[0, 0] / 2
    return gains, offsets, value, curvatures


def dynamics_curvature(t, next_value, states, controls):
    """The second-order term of DDP for move t, as a value factor on u_t.

    The value of the next state changes by V'(x') f_uu (u - u_bar)^2 / 2 when
    the dynamics are expanded to second order in the control.
    """
    quadratic = next_value.value()  # V_{t+1} = 1/2 G x^2 - g x + f/2
    gradient = (quadratic.information()[0, 0] * states[t + 1]
                - np.ravel(quadratic.linearTerm())[0])
    f_uu = -2 * np.tanh(controls[t]) * f_u(controls[t])
    c, u_bar = float(gradient * f_uu), float(controls[t])
    # A HessianFactor with (G, g, f) = (c, c u_bar, c u_bar^2) has error
    # c (u - u_bar)^2 / 2, and Reward lifts it to a value of that error.
    return SemiringGaussianFactor.Reward(HessianFactor(
        U(t), c * I, np.array([c * u_bar]), c * u_bar ** 2))


def value_at(value, x0):
    """Read V_0(x0) from the value factor on the first state."""
    point = VectorValues()
    point.insert(X(0), np.array([x0]))
    return value.value(point)


controls = np.zeros(T)
states = rollout(controls, x_start)
gains, offsets, value, _ = stage1(states, controls)
print("first backward pass, around the do-nothing trajectory")
print("gains K_t  =", gains)
print("offsets o_t =", offsets)
print("planned changes =", offsets - gains * states[:-1] - controls)
print("local model's value at the start:", value_at(value, x_start))
assert np.allclose(gains, [0.6176, 0.6154, 0.6, 0.5], atol=1e-4)
assert np.allclose(offsets - gains * states[:-1] - controls,
                   [-1.8529, -1.8462, -1.8, -1.5], atol=1e-4)
assert np.isclose(value_at(value, x_start), -14.559, atol=1e-3)

# %% [markdown]
# The same pass as one graph: all the linearized factors, a rule per
# variable, and one call. The Bayes net holds the local policy of every move.

# %%
graph, rules, backward = SemiringFactorGraph(), SemiringRules(), []
for t in range(T):
    graph.push_back(linearized(t, states, controls, 0.0))
    graph.push_back(penalty(X(t)))
    graph.push_back(penalty(U(t)))
    rules.set(U(t), maximum)
    backward = [X(t + 1), U(t)] + backward
graph.push_back(penalty(X(T)))
# The first state has no prior here, so it is left in the graph.
bayes_net, _ = graph.eliminatePartialSequential(ordering(*backward), rules)
graph_gains = np.array([bayes_net.at(2 * (T - 1 - t) + 1).conditional().S()[0, 0]
                        for t in range(T)])
print("gains from one elimination of the whole graph:", graph_gains)
assert np.allclose(graph_gains, gains)

# %% [markdown]
# ## Stage 2: roll out, search along the step, relinearize (Section 3)


# %%
def forward(states, controls, gains, offsets, step, x0):
    """Roll out the local policy, with the planned change scaled by step."""
    x, new_controls = x0, np.zeros(len(controls))
    for t in range(len(controls)):
        change = offsets[t] - gains[t] * states[t] - controls[t]
        new_controls[t] = (controls[t] + step * change
                           - gains[t] * (x - states[t]))
        x = f(x, new_controls[t])
    return new_controls


def ilqr(x0, moves, backward, verbose=False, tolerance=1e-8):
    """Relinearize and repeat until the return stops improving.

    Returns the controls and the return before each backward pass.
    """
    controls, history = np.zeros(moves), []
    for iteration in range(100):
        states = rollout(controls, x0)
        current = total_reward(controls, x0)
        history.append(current)
        gains, offsets = backward(states, controls)[:2]
        step = 1.0
        while True:  # halve the step until the return improves
            candidate = forward(states, controls, gains, offsets, step, x0)
            improvement = total_reward(candidate, x0) - current
            if improvement > 0 or step < 1e-8:
                break
            step /= 2
        if verbose:
            print(f"pass {iteration + 1:2d}: return before {current:11.6f}, "
                  f"step {step:g}, improvement {max(improvement, 0):.2e}")
        if improvement > 0:
            controls = candidate
        if improvement < tolerance:
            break
    return controls, history


controls, history = ilqr(x_start, T, stage1, verbose=True)
states = rollout(controls, x_start)
print("controls:", controls)
print("states:  ", states)
print("return:  ", total_reward(controls, x_start))

# An independent check with a general-purpose optimizer.
check = minimize(lambda c: -total_reward(c, x_start), np.zeros(T), tol=1e-12)
print("scipy:   ", check.x, -check.fun)
assert np.isclose(total_reward(controls, x_start), -19.549965, atol=1e-5)
assert np.allclose(controls, check.x, atol=1e-3)
assert np.allclose(controls, [-1.2701, -1.0257, -0.7213, -0.3610], atol=1e-3)
print("backward passes:", len(history))
assert len(history) == 13
ilqr_controls, ilqr_states, ilqr_history = controls, states, history

# %% [markdown]
# ## DDP (Section 4)
#
# DDP is the same backward pass with one more value factor per move, the
# curvature of the dynamics. It needs fewer passes.

# %%
print("DDP:")
ddp_controls, ddp_history = ilqr(
    x_start, T, lambda s, c: stage1(s, c, ddp=True), verbose=True)
assert np.allclose(ddp_controls, ilqr_controls, atol=1e-4)
print("backward passes:", len(ddp_history))
assert len(ddp_history) == 6

# %% [markdown]
# ## An independent check of the backward pass, in numpy (Sections 2 and 4)
#
# The formulas of the chapter, in deviations from the current trajectory: the
# pass carries the gradient and the curvature $P_t$ of the value. The module
# must agree with them, for iLQR and for DDP.


# %%
def backward_numpy(states, controls, ddp=False):
    """Gains and offsets of the local policy; also the curvatures P_t."""
    moves = len(controls)
    gains, offsets = np.zeros(moves), np.zeros(moves)
    P = np.ones(moves + 1)  # V_T = -x^2
    value_gradient = -2 * states[-1]
    for t in reversed(range(moves)):
        B = f_u(controls[t])
        Q_x = -2 * states[t] + value_gradient  # gradient of Q_t in x
        Q_u = -2 * controls[t] + B * value_gradient  # gradient of Q_t in u
        H_uu, H_ux, H_xx = 1 + B * P[t + 1] * B, B * P[t + 1], 1 + P[t + 1]
        if ddp:  # the second derivative of tanh(u) is -2 tanh(u) (1 - tanh^2)
            H_uu -= 0.5 * value_gradient * (-2 * np.tanh(controls[t]) * B)
        gains[t] = H_ux / H_uu
        change = 0.5 * Q_u / H_uu  # the planned change of u_t
        offsets[t] = controls[t] + change + gains[t] * states[t]
        value_gradient = Q_x - gains[t] * Q_u
        P[t] = H_xx - H_ux ** 2 / H_uu
    return gains, offsets, P


for trajectory in [(rollout(np.zeros(T), x_start), np.zeros(T)),
                   (ilqr_states, ilqr_controls),
                   (rollout(ilqr_controls / 2, x_start), ilqr_controls / 2)]:
    for ddp in [False, True]:
        module = stage1(*trajectory, ddp=ddp)
        formulas = backward_numpy(*trajectory, ddp=ddp)
        assert np.allclose(module[0], formulas[0])  # gains
        assert np.allclose(module[1], formulas[1])  # offsets
        assert np.allclose(module[3], formulas[2])  # curvatures
gains_at_solution, _, _, curvatures = stage1(ilqr_states, ilqr_controls)
print("slopes B_t at the start:    ", f_u(np.zeros(T)))
print("gains at the solution:      ", gains_at_solution)
print("curvatures P_t of the value:", curvatures)

# %% [markdown]
# ## An exact special case: the line (Section 7)
#
# With linear dynamics the linearization is exact, so one backward pass is the
# Riccati recursion of Chapter 6: gains 0.6 and 0.5, and $J^* = -9.25$ for
# $x_0 \sim N(2, 1)$.


# %%
def line(gains=None, slip_variance=0.5):
    """The factor graph of the line; with gains, the policy u = -K_t x."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([2.0]), noise_model(1.0)))
    for t in range(2):
        if gains is not None:  # the hard constraint u + K x = 0
            graph.push_back(gaussian(U(t), I, X(t), gains[t] * I, zero,
                                     noiseModel.Constrained.All(1)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 noise_model(slip_variance)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph


line_backward = ordering(X(2), U(1), X(1), U(0), X(0))
line_rules = SemiringRules()
line_rules.setAll([U(0), U(1)], maximum)

J_star = line().expectation(line_backward, line_rules)
bayes_net = line().eliminateSequential(line_backward, line_rules)
line_gains = np.array([bayes_net.at(3).conditional().S()[0, 0],
                       bayes_net.at(1).conditional().S()[0, 0]])
print("gains on the line:", line_gains, " J* =", J_star)
assert np.allclose(line_gains, [0.6, 0.5]) and np.isclose(J_star, -9.25)

# %% [markdown]
# ## Under noise: three ways to use the solution (Section 6)
#
# Now the moves slip: $x' = x + \tanh(u) + w$ with $w \sim N(0, 0.2)$. The
# plan is executed open loop, with the local feedback of the last backward
# pass, and by replanning at every step (MPC). The returns are averages over
# 100000 simulated runs with the same noise.

# %%
noise = 0.2
samples = 100000
rng = np.random.default_rng(0)
slips = rng.normal(0.0, np.sqrt(noise), (samples, T))


def simulate(policy):
    """Average return of a policy u = policy(t, x) over the simulated runs."""
    x, total = np.full(samples, x_start), np.zeros(samples)
    for t in range(T):
        u = policy(t, x)
        total -= x ** 2 + u ** 2
        x = f(x, u) + slips[:, t]
    total -= x ** 2
    return total.mean(), total.std() / np.sqrt(samples)


# The local model's own prediction: the noise adds a trace term per move.
local_gains, local_offsets, local_value, _ = stage1(
    ilqr_states, ilqr_controls, noise)
predicted = value_at(local_value, x_start)
print("return predicted by the local model:", predicted)

# MPC: the first control of a fresh plan, tabulated over the state.
grid = np.linspace(-3.0, 6.0, 181)
mpc_table = np.array([
    [ilqr(x, T - t, stage1)[0][0] for x in grid] for t in range(T)])

results = {
    "open loop": simulate(lambda t, x: np.full(samples, ilqr_controls[t])),
    "local feedback": simulate(
        lambda t, x: local_offsets[t] - local_gains[t] * x),
    "MPC": simulate(lambda t, x: np.interp(x, grid, mpc_table[t])),
}
for name, (mean, error) in results.items():
    print(f"{name:15s} return {mean:8.3f} +- {error:.3f}")
assert np.isclose(predicted, -20.958, atol=1e-3)
assert np.isclose(results["open loop"][0], -21.57, atol=0.05)
assert np.isclose(results["local feedback"][0], -21.22, atol=0.05)
assert np.isclose(results["MPC"][0], -21.12, atol=0.05)

# %% [markdown]
# ## MAP trajectory optimization with GTSAM (Section 5)
#
# The way a SLAM practitioner would pose the problem: all states and controls
# are unknowns of one nonlinear least-squares problem. Each reward is a prior
# factor pulling a variable to zero, and each dynamics factor is a soft
# constraint with the covariance of the noise.


# %%
def dynamics_error(this, values, jacobians):
    """The error x' - f(x, u) of a dynamics factor on (x, u, x')."""
    key_x, key_u, key_next = this.keys()
    x = values.atVector(key_x)[0]
    u = values.atVector(key_u)[0]
    x_next = values.atVector(key_next)[0]
    if jacobians is not None:
        jacobians[0] = np.array([[-1.0]])
        jacobians[1] = np.array([[-f_u(u)]])
        jacobians[2] = np.array([[1.0]])
    return np.array([x_next - f(x, u)])


def map_plan(x0, moves, variance):
    """Maximize the product of all factors over all states and controls."""
    graph = gtsam.NonlinearFactorGraph()
    cost = noiseModel.Isotropic.Variance(1, 0.5)  # error z^2
    graph.add(gtsam.PriorFactorVector(
        X(0), np.array([x0]), noiseModel.Constrained.All(1)))
    for t in range(moves):
        graph.add(gtsam.PriorFactorVector(X(t), zero, cost))
        graph.add(gtsam.PriorFactorVector(U(t), zero, cost))
        graph.add(CustomFactor(noiseModel.Isotropic.Variance(1, variance),
                               [X(t), U(t), X(t + 1)], dynamics_error))
    graph.add(gtsam.PriorFactorVector(X(moves), zero, cost))
    initial = gtsam.Values()
    for t in range(moves + 1):
        initial.insert(X(t), np.array([x0]))
    for t in range(moves):
        initial.insert(U(t), zero)
    parameters = gtsam.LevenbergMarquardtParams()
    parameters.setMaxIterations(500)
    parameters.setRelativeErrorTol(1e-14)
    parameters.setAbsoluteErrorTol(1e-14)
    result = gtsam.LevenbergMarquardtOptimizer(
        graph, initial, parameters).optimize()
    plan_u = np.array([result.atVector(U(t))[0] for t in range(moves)])
    plan_x = np.array([result.atVector(X(t))[0] for t in range(moves + 1)])
    return plan_u, plan_x


def planned_reward(plan_u, plan_x):
    """The return along a planned trajectory, slips included."""
    return -(np.sum(plan_x[:-1] ** 2) + np.sum(plan_u ** 2) + plan_x[-1] ** 2)


# Almost deterministic dynamics: the same trajectory as iLQR.
map_u, map_x = map_plan(x_start, T, 1e-6)
print("MAP, dynamics noise 1e-6: controls", map_u)
print("                          return  ", planned_reward(map_u, map_x))
assert np.allclose(map_u, ilqr_controls, atol=1e-3)

# With the real noise: the plan counts on slips in its favor.
map_u, map_x = map_plan(x_start, T, noise)
assumed_slips = map_x[1:] - f(map_x[:-1], map_u)
print("MAP, dynamics noise 0.2:  controls", map_u)
print("                          states  ", map_x)
print("                          slips it assumes", assumed_slips)
print("                          return it plans ",
      planned_reward(map_u, map_x))
assert np.allclose(map_u, [-0.9654, -0.5814, -0.2489, -0.0776], atol=1e-3)
assert np.isclose(planned_reward(map_u, map_x), -12.573, atol=0.01)

# %% [markdown]
# MAP inside MPC: replan at every step and apply the first control. The table
# of first controls is computed with scipy on the same least-squares problem,
# to keep the notebook fast.


# %%
def map_first_control(x0, moves, variance):
    def residuals(z):
        u, x = z[:moves], np.concatenate([[x0], z[moves:]])
        return np.concatenate(
            [x, u, (x[1:] - f(x[:-1], u)) / np.sqrt(2 * variance)])
    start = np.concatenate([np.zeros(moves), np.full(moves, x0)])
    return least_squares(residuals, start, xtol=1e-12, ftol=1e-12).x[0]


assert np.isclose(map_first_control(x_start, T, noise), map_u[0], atol=1e-4)
map_table = np.array([
    [map_first_control(x, T - t, noise) for x in grid] for t in range(T)])
results["MAP inside MPC"] = simulate(
    lambda t, x: np.interp(x, grid, map_table[t]))
mean, error = results["MAP inside MPC"]
print(f"MAP inside MPC  return {mean:8.3f} +- {error:.3f}")
print("first control at x = 3:  iLQR", np.interp(3.0, grid, mpc_table[0]),
      " MAP", np.interp(3.0, grid, map_table[0]))
assert np.isclose(mean, -21.80, atol=0.05)

# %% [markdown]
# ## The optimism of MAP, exactly, on the line (Section 5)
#
# On the line MAP is a linear least-squares problem, an ordinary
# `GaussianFactorGraph`: each penalty $z^2$ is a factor with error $z^2$, and
# each dynamics factor is a soft constraint with the variance of the slip.
# MAP treats the slip $w$ as a second control with cost $w^2 / (2 \Sigma_w)$,
# which here equals $w^2$, so the robot commands only half of the move it
# plans.

# %%
slip_variance = 0.5


def map_line(x0=None):
    """The MAP graph of the line, with the first state fixed at x0 if given."""
    graph = GaussianFactorGraph()
    if x0 is not None:
        graph.add(JacobianFactor(X(0), I, np.array([x0]),
                                 noiseModel.Constrained.All(1)))
    for t in range(2):
        graph.add(JacobianFactor(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 noise_model(slip_variance)))
        for key in [X(t), U(t)]:
            graph.add(HessianFactor(key, 2 * I, zero, 0.0))  # error z^2
    graph.add(HessianFactor(X(2), 2 * I, zero, 0.0))
    return graph


# Eliminating backward leaves a conditional on each action given its state.
bayes_net, _ = map_line().eliminatePartialSequential(
    ordering(X(2), U(1), X(1), U(0)))
map_gains = np.array([
    (bayes_net.at(position).S() / bayes_net.at(position).R()).item()
    for position in [3, 1]])
# What MAP expects from x0 = 2: minus the error of its best plan.
plan = map_line(2.0).optimize()
map_belief = -map_line(2.0).error(plan)
print("MAP gains on the line:", map_gains, " Riccati gains:", line_gains)
print("MAP's optimum from x0 = 2:", map_belief)
# For comparison, with P_0 = 1.6 and beta_0 = 1.25 from the Riccati recursion:
print("best return from x0 = 2 without noise:", -1.6 * 4)
print("best expected return from x0 = 2 with noise:", -(1.6 * 4 + 1.25))
assert map_belief > -1.6 * 4 > -(1.6 * 4 + 1.25)


def evaluate_line(gains):
    """Exact expected return of u = -K_t x on the line, with the module."""
    return line(gains).expectation(line_backward)


print("expected return with the Riccati gains:", evaluate_line(line_gains))
print("expected return with the MAP gains:    ", evaluate_line(map_gains))
assert np.allclose(map_gains, [4 / 11, 1 / 3])
assert np.isclose(map_belief, -60 / 11)
assert np.isclose(evaluate_line(line_gains), -9.25)
assert np.isclose(evaluate_line(map_gains), -10.0888, atol=1e-3)

# %% [markdown]
# ## Numbers for the figures
#
# The trajectories of the first iterations, and the two MPC policies.

# %%
controls = np.zeros(T)
for iteration in range(3):
    states = rollout(controls, x_start)
    print(f"after {iteration} passes: controls {controls}")
    print(f"                states   {states}, "
          f"return {total_reward(controls, x_start):.4f}")
    gains, offsets = stage1(states, controls)[:2]
    step = 1.0 if iteration == 0 else 0.5
    controls = forward(states, controls, gains, offsets, step, x_start)
print("final:       states", ilqr_states)
for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
    print(f"x = {x}: first control iLQR {np.interp(x, grid, mpc_table[0]):.3f}"
          f"  MAP {np.interp(x, grid, map_table[0]):.3f}")
