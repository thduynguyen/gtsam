# %% [markdown]
# # Chapter 10 examples: sampling-based control
#
# This notebook runs the examples of
# [Chapter 10](https://thduynguyen.github.io/gtsam/chapter10): MPPI as a
# sampled soft maximum and CEM as a sampled maximum, first on the line, where
# the exact answer is known, then on the weak motor of Chapter 9, inside
# model predictive control.

# %%
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line as a planning problem (Section 1)
#
# The robot is at $x_0 = 2$ and plans two moves with the noise-free model
# $x' = x + u$. A plan is an action sequence; its return is the return of its
# rollout.


# %%
def line_return(plans, x0):
    """Returns of plans on the line; plans has shape (..., moves)."""
    x = np.broadcast_to(np.asarray(x0, float), plans.shape[:-1]).copy()
    total = np.zeros(plans.shape[:-1])
    for t in range(plans.shape[-1]):
        total -= x ** 2 + plans[..., t] ** 2
        x = x + plans[..., t]
    return total - x ** 2


x0 = 2.0
eta = 1.0  # temperature of the soft maximum
variance = 1.0  # Sigma_e: variance of the sampling distribution
print("return of doing nothing:", line_return(np.zeros(2), x0))
print("return of the LQR plan (-1.2, -0.4):",
      line_return(np.array([-1.2, -0.4]), x0))
assert np.isclose(line_return(np.array([-1.2, -0.4]), x0), -6.4)

# %% [markdown]
# ## The exact soft maximum, by Gaussian elimination (Section 4)
#
# Eliminating an action by soft maximum under a Gaussian $N(0, \Sigma_e)$ is
# the max step of the Riccati recursion with the control cost increased by
# $\eta / (2 \Sigma_e)$, plus a constant.


# %%
def soft_riccati(moves, eta, variance, verbose=False):
    """V(x) = -(P x^2 + beta) of the soft maximum on the line, and the gains."""
    extra = eta / (2 * variance) if eta > 0 else 0.0
    P, beta, gains = 1.0, 0.0, np.zeros(moves)
    for t in reversed(range(moves)):
        H_uu, H_ux, H_xx = 1 + P, P, 1 + P
        gains[t] = H_ux / (H_uu + extra)
        if eta > 0:
            beta += 0.5 * eta * np.log(1 + 2 * variance * H_uu / eta)
        P = H_xx - H_ux ** 2 / (H_uu + extra)
        if verbose:
            print(f"step {t}: H_uu = {H_uu:.4f}, gain = {gains[t]:.4f}, "
                  f"P = {P:.4f}, beta = {beta:.4f}")
    return P, beta, gains


P, beta, soft_gains = soft_riccati(2, eta, variance, verbose=True)
soft_value = -(P * x0 ** 2 + beta)
print("soft maximum: P_0 =", P, " beta_0 =", beta, " gains =", soft_gains)
print("soft value at x0 = 2:", soft_value)
_, _, lqr_gains = soft_riccati(2, 0.0, variance)
print("gains of the hard maximum (LQR):", lqr_gains)
assert np.allclose(lqr_gains, [0.6, 0.5])
assert np.isclose(soft_value, -8.8138, atol=1e-4)

# The same number from one Gaussian integral over the whole plan. On the line
# the return is quadratic in the plan: R(u) = -(u'Mu + 2 b'u + c).
lower = np.tril(np.ones((2, 2)))  # x_{t+1} = x0 + (lower @ u)_t
M_matrix = np.eye(2) + lower.T @ lower
b_vector = lower.T @ np.ones(2) * x0
c_scalar = 3 * x0 ** 2
precision = np.eye(2) / variance + 2 * M_matrix / eta  # of the tilted plan
tilted_mean = np.linalg.solve(precision, -2 * b_vector / eta)
log_Z = (-0.5 * np.linalg.slogdet(variance * precision)[1]
         + 0.5 * (2 * b_vector / eta) @ np.linalg.solve(
             precision, 2 * b_vector / eta) - c_scalar / eta)
average_return = -(c_scalar + variance * np.trace(M_matrix))
print("average return of plans drawn from the sampling distribution:",
      average_return)
assert np.isclose(average_return, -17.0)
print("soft value from one Gaussian integral:", eta * log_Z)
print("mean of the tilted distribution over plans:", tilted_mean)
assert np.isclose(eta * log_Z, soft_value)
assert np.isclose(tilted_mean[0], -soft_gains[0] * x0)
assert np.allclose(tilted_mean, [-1.0323, -0.3871], atol=1e-4)

# The soft maximum is the best expected return minus eta times a KL
# divergence from the sampling distribution, attained by the tilted one.
tilted_covariance = np.linalg.inv(precision)
expected_return = -(np.trace(M_matrix @ tilted_covariance)
                    + tilted_mean @ M_matrix @ tilted_mean
                    + 2 * b_vector @ tilted_mean + c_scalar)
kl = 0.5 * (np.trace(tilted_covariance) / variance
            + tilted_mean @ tilted_mean / variance - 2
            - np.linalg.slogdet(tilted_covariance / variance)[1])
print("E_q[R] =", expected_return, " KL(q || pi) =", kl,
      " E_q[R] - eta KL =", expected_return - eta * kl)
assert np.isclose(expected_return - eta * kl, soft_value)

# %% [markdown]
# ## The sampled soft maximum (Sections 2 and 4)
#
# Draw $M$ plans, roll them out, and replace the average over plans by the
# average over samples.


# %%
def mppi_estimate(returns, eta):
    """Soft value, normalized weights and effective number of samples."""
    log_weights = returns / eta - logsumexp(returns / eta)
    weights = np.exp(log_weights)
    value = eta * (logsumexp(returns / eta) - np.log(len(returns)))
    return value, weights, 1 / np.sum(weights ** 2)


rng = np.random.default_rng(0)
print("     M   soft value    error   first action   effective samples")
errors = {}
for M in [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]:
    plans = rng.normal(0.0, np.sqrt(variance), (M, 2))
    value, weights, effective = mppi_estimate(line_return(plans, x0), eta)
    errors[M] = value - soft_value
    print(f"{M:7d}   {value:9.4f}  {errors[M]:+8.4f}   "
          f"{weights @ plans[:, 0]:9.4f}   {effective:12.1f}")
assert abs(errors[10 ** 6]) < 0.005 and abs(errors[10]) > 0.1

# %% [markdown]
# ## The temperature (Section 4)
#
# As $\eta \to 0$ the soft maximum becomes the maximum and the mean of the
# tilted plan becomes the LQR plan $(-1.2, -0.4)$. The samples that carry the
# weight become few.

# %%
rng = np.random.default_rng(1)
plans = rng.normal(0.0, np.sqrt(variance), (10 ** 4, 2))
returns = line_return(plans, x0)
print("   eta   exact soft value   exact tilted mean      "
      "sampled mean (M = 10^4)   effective samples")
for temperature in [100, 10, 1, 0.1, 0.01, 0.001]:
    P_, beta_, gains_ = soft_riccati(2, temperature, variance)
    exact_precision = np.eye(2) / variance + 2 * M_matrix / temperature
    exact_mean = np.linalg.solve(exact_precision, -2 * b_vector / temperature)
    _, weights, effective = mppi_estimate(returns, temperature)
    print(f"{temperature:6g}   {-(P_ * x0 ** 2 + beta_):12.4f}      "
          f"{exact_mean}      {weights @ plans}      {effective:10.1f}")
P_, beta_, _ = soft_riccati(2, 0.001, variance)
assert np.isclose(-(P_ * x0 ** 2 + beta_), -6.4, atol=0.02)
assert np.allclose(exact_mean, [-1.2, -0.4], atol=1e-3)

# %% [markdown]
# ## Stage 2: move the sampling distribution to the tilted mean (Section 3)
#
# Sampling around the current plan $\bar u$ and moving to the weighted mean,
# repeatedly. For a quadratic return each step is exactly a damped step: it
# maximizes $R(u) - \frac{\eta}{2 \Sigma_e} \lVert u - \bar u \rVert^2$.

# %%
damping = eta / (2 * variance)
plan = np.zeros(2)
print("exact iteration")
for k in range(6):
    print(f"  {k}: plan {plan}, return {line_return(plan, x0):.6f}")
    plan = np.linalg.solve(M_matrix + damping * np.eye(2),
                           damping * plan - b_vector)
assert np.allclose(plan, [-1.2, -0.4], atol=1e-3)

rng = np.random.default_rng(2)
plan = np.zeros(2)
print("sampled iteration, M = 1000")
for k in range(6):
    print(f"  {k}: plan {plan}, return {line_return(plan, x0):.4f}")
    plans = plan + rng.normal(0.0, np.sqrt(variance), (1000, 2))
    _, weights, _ = mppi_estimate(line_return(plans, x0), eta)
    plan = weights @ plans
assert np.allclose(plan, [-1.2, -0.4], atol=0.1)

# %% [markdown]
# ## CEM: a sampled maximum (Section 3)
#
# Keep the best plans, refit the mean and the spread of the sampling
# distribution to them, repeat.


# %%
def cem(plan_return, moves, samples, elites, iterations, rng, verbose=True):
    mean, spread = np.zeros(moves), np.ones(moves)
    for k in range(iterations):
        if verbose:
            print(f"  {k:2d}: mean {mean}, spread {spread}, "
                  f"return {plan_return(mean):.5f}")
        plans = mean + spread * rng.normal(size=(samples, moves))
        best = plans[np.argsort(plan_return(plans))[-elites:]]
        mean, spread = best.mean(axis=0), best.std(axis=0)
    return mean


cem_plan = cem(lambda plans: line_return(plans, x0), moves=2, samples=100,
               elites=10, iterations=10, rng=np.random.default_rng(3))
print("CEM plan on the line:", cem_plan, line_return(cem_plan, x0))
assert np.allclose(cem_plan, [-1.2, -0.4], atol=0.01)

# %% [markdown]
# ## The weak motor of Chapter 9 (Section 5)
#
# $x' = x + \tanh(u)$, four moves from $x_0 = 3$. The planners only call the
# model; they never differentiate it.

# %%
T = 4
x_start = 3.0


def f(x, u):
    return x + np.tanh(u)


def motor_return(plans, x0):
    """Returns of plans on the weak motor; plans has shape (..., moves)."""
    x = np.broadcast_to(np.asarray(x0, float), plans.shape[:-1]).copy()
    total = np.zeros(plans.shape[:-1])
    for t in range(plans.shape[-1]):
        total -= x ** 2 + plans[..., t] ** 2
        x = f(x, plans[..., t])
    return total - x ** 2


best = minimize(lambda u: -motor_return(u, x_start), np.zeros(T), tol=1e-12)
print("best plan (Chapter 9):", best.x, " return", -best.fun)
assert np.isclose(-best.fun, -19.549965, atol=1e-5)

motor_eta, motor_variance = 0.5, 0.25
rng = np.random.default_rng(0)
plan = np.zeros(T)
print("MPPI, M = 1000")
mppi_history = []
for k in range(15):
    mppi_history.append(motor_return(plan, x_start))
    plans = plan + rng.normal(0.0, np.sqrt(motor_variance), (1000, T))
    _, weights, effective = mppi_estimate(
        motor_return(plans, x_start), motor_eta)
    if k in [0, 1, 2, 3, 5, 10, 14]:
        print(f"  {k:2d}: return of the plan {mppi_history[-1]:9.4f}, "
              f"effective samples {effective:6.1f}")
    plan = weights @ plans
print("  plan:", plan)
assert -19.65 < mppi_history[-1] < -19.55

print("CEM, M = 100, 10 elites")
cem_plan = cem(lambda plans: motor_return(plans, x_start), moves=T,
               samples=100, elites=10, iterations=15,
               rng=np.random.default_rng(0))
print("  plan:", cem_plan, " return", motor_return(cem_plan, x_start))
assert np.isclose(motor_return(cem_plan, x_start), -19.549965, atol=1e-3)

# %% [markdown]
# ## Inside MPC, under slips (Section 5)
#
# The real moves slip, $x' = x + \tanh(u) + w$ with $w \sim N(0, 0.2)$. Each
# planner replans at every step with the noise-free model and applies its
# first action. The reference is MPC with iLQR, from Chapter 9. All planners
# face the same 5000 sequences of slips, so they are compared run by run.


# %%
def ilqr_first_action(x0, moves):
    """The first control of the iLQR plan (Chapter 9), in numpy."""
    def rollout(controls):
        states = [x0]
        for u in controls:
            states.append(f(states[-1], u))
        return np.array(states)

    controls = np.zeros(moves)
    for _ in range(100):
        states, current = rollout(controls), motor_return(controls, x0)
        gains, changes = np.zeros(moves), np.zeros(moves)
        P, gradient = 1.0, -2 * states[-1]
        for t in reversed(range(moves)):
            B = 1 - np.tanh(controls[t]) ** 2
            Q_x, Q_u = -2 * states[t] + gradient, -2 * controls[t] + B * gradient
            H_uu, H_ux, H_xx = 1 + B * P * B, B * P, 1 + P
            gains[t], changes[t] = H_ux / H_uu, 0.5 * Q_u / H_uu
            gradient, P = Q_x - gains[t] * Q_u, H_xx - H_ux ** 2 / H_uu
        step = 1.0
        while True:
            x, candidate = x0, np.zeros(moves)
            for t in range(moves):
                candidate[t] = (controls[t] + step * changes[t]
                                - gains[t] * (x - states[t]))
                x = f(x, candidate[t])
            improvement = motor_return(candidate, x0) - current
            if improvement > 0 or step < 1e-8:
                break
            step /= 2
        if improvement > 0:
            controls = candidate
        if improvement < 1e-8:
            break
    return controls[0]


grid = np.linspace(-3.0, 6.0, 181)
ilqr_table = np.array([
    [ilqr_first_action(x, T - t) for x in grid] for t in range(T)])

noise, runs = 0.2, 5000
slips = np.random.default_rng(0).normal(0.0, np.sqrt(noise), (runs, T))


def simulate(planner):
    """Returns of all runs under MPC with the given planner."""
    x, total, memory = np.full(runs, x_start), np.zeros(runs), None
    for t in range(T):
        u, memory = planner(t, x, memory)
        total -= x ** 2 + u ** 2
        x = f(x, u) + slips[:, t]
    return total - x ** 2


def ilqr_planner(t, x, memory):
    return np.interp(x, grid, ilqr_table[t]), None


def mppi_planner(samples, iterations, seed):
    rng = np.random.default_rng(seed)

    def planner(t, x, mean):
        moves = T - t
        # Start from the previous plan, shifted by one step.
        mean = np.zeros((runs, moves)) if mean is None else mean[:, 1:]
        for _ in range(iterations):
            plans = mean[:, None, :] + rng.normal(
                0.0, np.sqrt(motor_variance), (runs, samples, moves))
            returns = motor_return(plans, x[:, None])
            weights = np.exp(returns / motor_eta - logsumexp(
                returns / motor_eta, axis=1, keepdims=True))
            mean = np.einsum("rm,rmt->rt", weights, plans)
        return mean[:, 0], mean
    return planner


def cem_planner(samples, elites, iterations, seed):
    rng = np.random.default_rng(seed)

    def planner(t, x, memory):
        moves = T - t
        mean, spread = np.zeros((runs, moves)), np.ones((runs, moves))
        for _ in range(iterations):
            plans = mean[:, None, :] + spread[:, None, :] * rng.normal(
                size=(runs, samples, moves))
            order = np.argsort(motor_return(plans, x[:, None]), axis=1)
            best = np.take_along_axis(
                plans, order[:, -elites:, None], axis=1)
            mean, spread = best.mean(axis=1), best.std(axis=1)
        return mean[:, 0], None
    return planner


reference = simulate(ilqr_planner)
print(f"iLQR                        return {reference.mean():8.3f} "
      f"+- {reference.std() / np.sqrt(runs):.3f}")
gaps = {}
for name, planner in [
        ("MPPI,  64 samples x 2", mppi_planner(64, 2, seed=1)),
        ("MPPI, 256 samples x 2", mppi_planner(256, 2, seed=1)),
        ("MPPI, 256 samples x 4", mppi_planner(256, 4, seed=1)),
        ("CEM,   64 samples x 4", cem_planner(64, 8, 4, seed=2)),
        ("CEM,   64 samples x 8", cem_planner(64, 8, 8, seed=2))]:
    returns = simulate(planner)
    gap = returns - reference
    gaps[name] = gap.mean()
    print(f"{name:27s} return {returns.mean():8.3f}   "
          f"gap to iLQR {gap.mean():+.3f} +- {gap.std() / np.sqrt(runs):.3f}")
assert np.isclose(reference.mean(), -21.171, atol=1e-3)
assert np.isclose(gaps["MPPI,  64 samples x 2"], -0.109, atol=0.01)
assert np.isclose(gaps["MPPI, 256 samples x 4"], -0.011, atol=0.005)
assert np.isclose(gaps["CEM,   64 samples x 8"], -0.003, atol=0.003)

# %% [markdown]
# ## What breaks: long plans (Section 7)
#
# On the line with a longer horizon, the same 10000 samples cover the space of
# plans less and less. The exact soft value is known for every horizon.

# %%
print(" moves   exact soft value   estimate (M = 10^4)   effective samples")
for moves in [2, 5, 10, 20, 40]:
    plans = np.random.default_rng(2).normal(0.0, 1.0, (10 ** 4, moves))
    value, _, effective = mppi_estimate(line_return(plans, x0), 1.0)
    P_, beta_, _ = soft_riccati(moves, 1.0, 1.0)
    exact = -(P_ * x0 ** 2 + beta_)
    print(f"{moves:6d}   {exact:14.3f}   {value:16.3f}   {effective:14.1f}")
    if moves == 2:
        assert abs(value - exact) < 0.05
    if moves == 40:
        assert abs(value - exact) > 5 and effective < 3

# %% [markdown]
# ## What breaks: tilting the slips too (Section 7)
#
# If the planner samples the slips of the line together with the actions and
# weights whole rollouts by $e^{R / \eta}$, the tilt applies to the slips as
# well. The plan then counts on lucky slips, like the MAP plan of Chapter 9.

# %%
slip_variance = 0.5
# Variables z = (u0, w0, u1, w1); x1 = x0 + u0 + w0, x2 = x1 + u1 + w1.
steps = np.array([[1, 1, 0, 0], [1, 1, 1, 1.0]])
M_z = np.diag([1, 0, 1, 0.0]) + steps.T @ steps
b_z = steps.T @ np.ones(2) * x0
prior_precision = np.diag(
    [1 / variance, 1 / slip_variance, 1 / variance, 1 / slip_variance])
tilted_z = np.linalg.solve(prior_precision + 2 * M_z / eta, -2 * b_z / eta)
print("tilting the actions only:  first action", tilted_mean[0])
print("tilting actions and slips: first action", tilted_z[0],
      " and it expects a first slip of", tilted_z[1])
assert np.isclose(tilted_z[0], -0.557, atol=1e-3)

rng = np.random.default_rng(4)
samples = rng.normal(0.0, 1.0, (10 ** 6, 4)) * np.sqrt(
    [variance, slip_variance, variance, slip_variance])
x1 = x0 + samples[:, 0] + samples[:, 1]
x2 = x1 + samples[:, 2] + samples[:, 3]
noisy_returns = -(x0 ** 2 + samples[:, 0] ** 2 + x1 ** 2
                  + samples[:, 2] ** 2 + x2 ** 2)
_, weights, _ = mppi_estimate(noisy_returns, eta)
print("the same by sampling, M = 10^6: first action", weights @ samples[:, 0])
assert np.isclose(weights @ samples[:, 0], tilted_z[0], atol=0.02)
