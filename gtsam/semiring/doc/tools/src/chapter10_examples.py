# %% [markdown]
# # Chapter 10 examples: sampling-based control
#
# This notebook runs the examples of
# [Chapter 10](https://thduynguyen.github.io/gtsam/chapter10): MPPI as a
# sampled soft maximum and CEM as a sampled maximum, first on the line, where
# the exact answer is known, then on the weak motor of Chapter 9, inside
# model predictive control. The sampling is plain numpy. The exact numbers
# that the samples are checked against come from the `gtsam/semiring` module:
# the sampling distribution is the policy factor of the actions, and the
# actions are summed out by the soft maximum.

# %%
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from gtsam import GaussianBayesNet, GaussianFactorGraph, HessianFactor
from gtsam import JacobianFactor, Ordering, SemiringFactorGraph
from gtsam import SemiringGaussianFactor, SemiringRules, SemiringSum
from gtsam import noiseModel
from gtsam.symbol_shorthand import U, X

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
# The same problem as a factor graph of the module: a hard prior on $x_0$, a
# dynamics factor and two reward factors per move and, when the actions are
# sampled, a policy factor on every action: the sampling distribution.

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


def backward(moves):
    """The elimination order backward in time: x_T, u_{T-1}, ..., u_0, x_0."""
    keys = []
    for t in reversed(range(moves)):
        keys += [X(t + 1), U(t)]
    return ordering(*keys, X(0))


def sampling(plan, variance):
    """The sampling distribution: a factor N(plan_t, variance) per action."""
    return [JacobianFactor(U(t), I, np.array([mean]), noise_model(variance))
            for t, mean in enumerate(plan)]


def dynamics(t, slips=0.0):
    """The dynamics factor x' = x + u + w, for slips w of the given variance."""
    return gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero, noise_model(slips))


def line(x0, moves, policies=None, slips=0.0):
    """The graph of the line from a known x0, with or without policy factors."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([x0]), noise_model(0)))
    for t in range(moves):
        if policies is not None:
            graph.push_back(SemiringGaussianFactor(policies[t]))
        graph.push_back(dynamics(t, slips))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(moves)))
    return graph


def rules(moves, at_actions, at_states=SemiringSum.Average()):
    """One rule for the actions and one for the states."""
    result = SemiringRules()
    result.setAll([U(t) for t in range(moves)], at_actions)
    result.setAll([X(t) for t in range(moves + 1)], at_states)
    return result


def action(bayes_net, moves, t):
    """The conditional on the action u_t, in a Bayes net eliminated backward."""
    return bayes_net.at(2 * (moves - 1 - t) + 1)


def solve(conditionals, moves):
    """The actions that a list of GaussianConditionals determines."""
    bayes_net = GaussianBayesNet()
    for conditional in conditionals:
        bayes_net.push_back(conditional)
    values = bayes_net.optimize()
    return np.array([values.at(U(t))[0] for t in range(moves)])


# %% [markdown]
# The two ends of Section 2, each by one elimination. The maximum over plans:
# no policy factor, and the maximum rule at the actions. The hard constraints
# that it leaves on the actions are the LQR policy of Chapter 6. The average
# over plans: the sampling distribution as the policy factor, and the default
# rule, the average.

# %%
graph = line(x0, 2)
best_return = graph.expectation(backward(2), rules(2, maximum))
bayes_net = graph.eliminateSequential(backward(2), rules(2, maximum))
lqr_gains = [action(bayes_net, 2, t).conditional().S()[0, 0] for t in range(2)]
lqr_plan = solve([bayes_net.at(i).conditional() for i in range(5)], 2)
print("maximum over plans:", best_return, " gains:", lqr_gains,
      " plan:", lqr_plan)
assert np.isclose(best_return, -6.4)
assert np.allclose(lqr_gains, [0.6, 0.5])
assert np.allclose(lqr_plan, [-1.2, -0.4])

average_return = line(x0, 2, sampling(np.zeros(2), variance)).expectation(
    backward(2))
print("average return of plans drawn from the sampling distribution:",
      average_return)
assert np.isclose(average_return, -17.0)

# %% [markdown]
# ## The exact soft maximum, by Gaussian elimination (Section 4)
#
# The plan is eliminated with the soft maximum: the rule
# `SemiringSum.SoftMaximum(eta)` at the actions, whose policy factors are the
# sampling distribution $N(0, \Sigma_e)$. The states follow the noise-free
# model, so they keep the default rule.
#
# The conditional that the rule leaves on an action holds the sampling
# density and the surprise $A_t(u, x) = Q_t(x, u) - V_t(x)$. Its tilted
# version, the density times $e^{A_t / \eta}$, is the improved distribution of
# the action. Both factors are Gaussian, so their product is taken with plain
# GTSAM.


# %%
def tilted(conditional, eta):
    """The tilted version of the conditional on an action.

    The product of the sampling density and e^{A / eta}, as a
    GaussianConditional R u + S x = d: the action has mean (d - S x) / R.
    """
    surprise = conditional.surprise()  # the quadratic A(u, x)
    u, x = surprise.keys()
    G = -surprise.information() / eta
    g = -surprise.linearTerm().ravel() / eta
    exponent = HessianFactor(u, x, G[:1, :1], G[:1, 1:], g[:1], G[1:, 1:],
                             g[1:], -surprise.constantTerm() / eta)
    graph = GaussianFactorGraph()
    graph.add(conditional.conditional())  # the sampling density
    graph.add(exponent)  # error -A / eta: the factor e^{A / eta}
    bayes_net, _ = graph.eliminatePartialSequential(ordering(u))
    return bayes_net.at(0)


def soft_plan(x0, moves, eta, variance, plan=None, slips=0.0,
              at_states=SemiringSum.Average()):
    """Eliminate the line with the soft maximum at the actions.

    The sampling distribution is N(plan_t, variance) on every action. Returns
    the soft value, the tilted conditionals of the actions, and the Bayes net.
    """
    plan = np.zeros(moves) if plan is None else plan
    graph = line(x0, moves, sampling(plan, variance), slips)
    soft = rules(moves, SemiringSum.SoftMaximum(eta), at_states)
    value = graph.expectation(backward(moves), soft)
    bayes_net = graph.eliminateSequential(backward(moves), soft)
    policies = [tilted(action(bayes_net, moves, t), eta) for t in range(moves)]
    return value, policies, bayes_net


def tilted_mean(policies, bayes_net):
    """The mean of the tilted distribution over plans, for a noise-free model.

    The tilted conditionals of the actions replace the sampling densities in
    the Bayes net; solving it gives the mean of every variable.
    """
    moves = len(policies)
    conditionals = [bayes_net.at(i).conditional() for i in range(2 * moves + 1)]
    for t in range(moves):
        conditionals[2 * (moves - 1 - t) + 1] = policies[t]
    return solve(conditionals, moves)


soft_value, policies, bayes_net = soft_plan(x0, 2, eta, variance)
soft_gains = np.array([(policy.S() / policy.R()).item() for policy in policies])
mean_plan = tilted_mean(policies, bayes_net)
print("soft value at x0 = 2:", soft_value)
print("gains of the tilted conditionals:", soft_gains)
print("mean of the tilted distribution over plans:", mean_plan)
assert np.isclose(soft_value, -8.8138, atol=1e-4)
assert np.allclose(soft_gains, [0.516, 0.4], atol=1e-3)
assert np.allclose(mean_plan, [-1.0323, -0.3871], atol=1e-4)
assert np.isclose(mean_plan[0], -soft_gains[0] * x0)
assert best_return > soft_value > average_return

# %% [markdown]
# The same elimination one variable at a time, to read the table of the
# chapter: the curvature $H_{uu}$ of $Q_t$, the gain, and the value
# $V_t(x) = -(P_t\, x^2 + \beta_t)$.

# %%
rule = SemiringSum.SoftMaximum(eta)
policy_factors = sampling(np.zeros(2), variance)
value = penalty(X(2))  # (1, V_2)
table = {}
for t in [1, 0]:
    # Eliminate the next state by average: (1, Q_t).
    Q = penalty(X(t)).multiply(penalty(U(t))).multiply(
        dynamics(t).multiply(value).sum(ordering(X(t + 1))))
    quadratic = Q.value()
    index = list(quadratic.keys()).index(U(t))
    H_uu = -quadratic.information()[index, index] / 2
    # Eliminate the action by soft maximum, under its sampling density.
    bucket = SemiringGaussianFactor(policy_factors[t]).multiply(Q)
    conditional, value = bucket.eliminate(ordering(U(t)), rule)
    policy = tilted(conditional, eta)
    quadratic = value.value()  # V_t
    table[t] = (H_uu, (policy.S() / policy.R()).item(),
                -quadratic.information()[0, 0] / 2,
                -quadratic.constantTerm() / 2)
    print("step {}: H_uu = {:.4f}, gain = {:.4f}, P = {:.4f}, "
          "beta = {:.4f}".format(t, *table[t]))
assert np.allclose(table[1], [2, 0.4, 1.6, 0.805], atol=1e-3)
assert np.allclose(table[0], [2.6, 0.516, 1.774, 1.717], atol=1e-3)
assert np.isclose(-(table[0][2] * x0 ** 2 + table[0][3]), soft_value)

# %% [markdown]
# ### Two independent checks, in numpy
#
# The recursion of Section 4: the Riccati recursion with the control cost
# increased by $\eta / (2 \Sigma_e)$, plus a constant. And one Gaussian
# integral over the whole plan.


# %%
def soft_riccati(moves, eta, variance):
    """V(x) = -(P x^2 + beta) of the soft maximum on the line, and the gains."""
    extra = eta / (2 * variance)
    P, beta, gains = 1.0, 0.0, np.zeros(moves)
    for t in reversed(range(moves)):
        H_uu, H_ux, H_xx = 1 + P, P, 1 + P
        gains[t] = H_ux / (H_uu + extra)
        beta += 0.5 * eta * np.log(1 + 2 * variance * H_uu / eta)
        P = H_xx - H_ux ** 2 / (H_uu + extra)
    return P, beta, gains


P, beta, gains = soft_riccati(2, eta, variance)
print("recursion: P_0 =", P, " beta_0 =", beta, " gains =", gains)
assert np.isclose(-(P * x0 ** 2 + beta), soft_value)
assert np.allclose(gains, soft_gains)

# On the line the return is quadratic in the plan: R(u) = -(u'Mu + 2 b'u + c).
lower = np.tril(np.ones((2, 2)))  # x_{t+1} = x0 + (lower @ u)_t
M_matrix = np.eye(2) + lower.T @ lower
b_vector = lower.T @ np.ones(2) * x0
c_scalar = 3 * x0 ** 2
precision = np.eye(2) / variance + 2 * M_matrix / eta  # of the tilted plan
integral_mean = np.linalg.solve(precision, -2 * b_vector / eta)
log_Z = (-0.5 * np.linalg.slogdet(variance * precision)[1]
         + 0.5 * (2 * b_vector / eta) @ np.linalg.solve(
             precision, 2 * b_vector / eta) - c_scalar / eta)
print("one Gaussian integral: soft value", eta * log_Z,
      " tilted mean", integral_mean)
assert np.isclose(eta * log_Z, soft_value)
assert np.allclose(integral_mean, mean_plan)
assert np.isclose(-(c_scalar + variance * np.trace(M_matrix)), average_return)

# %% [markdown]
# ### What the soft maximum optimizes (Section 2)
#
# The soft maximum is the best expected return minus $\eta$ times a KL
# divergence from the sampling distribution, attained by the tilted one. The
# expected return under the tilted distribution $q$ is the plain expectation
# of the graph with the tilted conditionals as policy factors. The KL
# divergence is the expectation of $\log (q / \pi_\theta) = \sum_t A_t / \eta$:
# the same graph with the surprises as its reward factors.

# %%
expected_return = line(x0, 2, policies).expectation(backward(2))

graph = SemiringFactorGraph()
graph.push_back(gaussian(X(0), I, np.array([x0]), noise_model(0)))
for t in range(2):
    graph.push_back(SemiringGaussianFactor(policies[t]))
    graph.push_back(dynamics(t))
    graph.push_back(SemiringGaussianFactor.Reward(
        action(bayes_net, 2, t).surprise()))
kl = graph.expectation(backward(2)) / eta
print("E_q[R] =", expected_return, " KL(q || pi) =", kl,
      " E_q[R] - eta KL =", expected_return - eta * kl)
assert np.isclose(expected_return, -7.296, atol=1e-3)
assert np.isclose(kl, 1.518, atol=1e-3)
assert np.isclose(expected_return - eta * kl, soft_value)

# The same two numbers from the Gaussian over plans, in numpy.
tilted_covariance = np.linalg.inv(precision)
assert np.isclose(expected_return, -(
    np.trace(M_matrix @ tilted_covariance)
    + integral_mean @ M_matrix @ integral_mean
    + 2 * b_vector @ integral_mean + c_scalar))
assert np.isclose(kl, 0.5 * (
    np.trace(tilted_covariance) / variance
    + integral_mean @ integral_mean / variance - 2
    - np.linalg.slogdet(tilted_covariance / variance)[1]))

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
# weight become few. The exact columns are one elimination per temperature.

# %%
rng = np.random.default_rng(1)
plans = rng.normal(0.0, np.sqrt(variance), (10 ** 4, 2))
returns = line_return(plans, x0)
expected = {100: (-16.483, [-0.075, -0.037]), 10: (-13.615, [-0.473, -0.218]),
            1: (-8.814, [-1.032, -0.387]), 0.1: (-6.861, [-1.180, -0.400]),
            0.01: (-6.469, [-1.198, -0.400]),
            0.001: (-6.409, [-1.200, -0.400])}
print("   eta   exact soft value   exact tilted mean      "
      "sampled mean (M = 10^4)   effective samples")
for temperature in [100, 10, 1, 0.1, 0.01, 0.001]:
    exact_value, policies_, bayes_net_ = soft_plan(
        x0, 2, temperature, variance)
    exact_mean = tilted_mean(policies_, bayes_net_)
    _, weights, effective = mppi_estimate(returns, temperature)
    print(f"{temperature:6g}   {exact_value:12.4f}      "
          f"{exact_mean}      {weights @ plans}      {effective:10.1f}")
    assert np.isclose(exact_value, expected[temperature][0], atol=1e-3)
    assert np.allclose(exact_mean, expected[temperature][1], atol=1e-3)
    # The checks in numpy: the recursion, and the Gaussian over plans.
    P, beta, _ = soft_riccati(2, temperature, variance)
    assert np.isclose(exact_value, -(P * x0 ** 2 + beta))
    assert np.allclose(exact_mean, np.linalg.solve(
        np.eye(2) / variance + 2 * M_matrix / temperature,
        -2 * b_vector / temperature))
assert np.isclose(exact_value, best_return, atol=0.02)
assert np.allclose(exact_mean, lqr_plan, atol=1e-3)

# %% [markdown]
# ## Stage 2: move the sampling distribution to the tilted mean (Section 3)
#
# Sampling around the current plan $\bar u$ and moving to the weighted mean,
# repeatedly. The exact iteration is one elimination per round, with the
# sampling distribution centered on the current plan. For a quadratic return
# each step is exactly a damped step: it maximizes
# $R(u) - \frac{\eta}{2 \Sigma_e} \lVert u - \bar u \rVert^2$. That maximum is
# one more elimination: the damping as a reward factor on every action, and
# the maximum rule.


# %%
def damped_step(plan, damping):
    """The plan that maximizes R(u) - damping ||u - plan||^2."""
    graph = line(x0, 2)
    for t in range(2):
        # The cost damping (u - plan_t)^2, as a HessianFactor.
        graph.push_back(SemiringGaussianFactor.Cost(HessianFactor(
            U(t), 2 * damping * I, np.array([2 * damping * plan[t]]),
            2 * damping * plan[t] ** 2)))
    bayes_net = graph.eliminateSequential(backward(2), rules(2, maximum))
    return solve([bayes_net.at(i).conditional() for i in range(5)], 2)


damping = eta / (2 * variance)
plan = np.zeros(2)
print("exact iteration")
for k in range(6):
    print(f"  {k}: plan {plan}, return {line_return(plan, x0):.6f}")
    _, policies_, bayes_net_ = soft_plan(x0, 2, eta, variance, plan)
    new_plan = tilted_mean(policies_, bayes_net_)
    assert np.allclose(new_plan, damped_step(plan, damping))
    # The same step in numpy.
    assert np.allclose(new_plan, np.linalg.solve(
        M_matrix + damping * np.eye(2), damping * plan - b_vector))
    plan = new_plan
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
# first action. The reference is MPC with iLQR, from Chapter 9: its backward
# pass is the elimination of the linearized graph with the maximum at the
# actions, with the module. All planners face the same 5000 sequences of
# slips, so they are compared run by run.


# %%
def stage1(states, controls):
    """The backward pass of iLQR (Chapter 9), with the module.

    Returns the gains K_t and offsets o_t of the local policy u = -K_t x + o_t.
    """
    moves = len(controls)
    gains, offsets = np.zeros(moves), np.zeros(moves)
    value = penalty(X(moves))  # (1, V_T)
    for t in reversed(range(moves)):
        # Linearized dynamics: x' = x + B u + constant, with B = df/du.
        B = 1 - np.tanh(controls[t]) ** 2
        constant = states[t + 1] - states[t] - B * controls[t]
        linearized = gaussian(X(t + 1), I, X(t), -I, U(t), -B * I,
                              np.array([constant]), noise_model(0))
        # Eliminate the next state by average and the action by maximum.
        phi = linearized.multiply(value).sum(ordering(X(t + 1)))
        bucket = penalty(X(t)).multiply(penalty(U(t))).multiply(phi)
        conditional, value = bucket.eliminate(ordering(U(t)), maximum)
        policy = conditional.conditional()  # the constraint u + K x = o
        gains[t], offsets[t] = policy.S()[0, 0], policy.d()[0]
    return gains, offsets


def ilqr_first_action(x0, moves):
    """The first control of the iLQR plan (Chapter 9)."""
    def rollout(controls):
        states = [x0]
        for u in controls:
            states.append(f(states[-1], u))
        return np.array(states)

    controls = np.zeros(moves)
    for _ in range(100):
        states, current = rollout(controls), motor_return(controls, x0)
        gains, offsets = stage1(states, controls)
        changes = offsets - gains * states[:-1] - controls
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
# plans less and less. The exact soft value is one elimination for every
# horizon.

# %%
expected = {2: -8.814, 5: -11.839, 10: -16.575, 20: -26.045, 40: -44.985}
print(" moves   exact soft value   estimate (M = 10^4)   effective samples")
for moves in [2, 5, 10, 20, 40]:
    plans = np.random.default_rng(2).normal(0.0, 1.0, (10 ** 4, moves))
    value, _, effective = mppi_estimate(line_return(plans, x0), 1.0)
    exact = line(x0, moves, sampling(np.zeros(moves), 1.0)).expectation(
        backward(moves), rules(moves, SemiringSum.SoftMaximum(1.0)))
    print(f"{moves:6d}   {exact:14.3f}   {value:16.3f}   {effective:14.1f}")
    assert np.isclose(exact, expected[moves], atol=1e-3)
    P, beta, _ = soft_riccati(moves, 1.0, 1.0)  # the check in numpy
    assert np.isclose(exact, -(P * x0 ** 2 + beta))
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
#
# With the module this is one more rule: the dynamics factor gets the slips,
# and the states get the tilted mean `SemiringSum.Tilted(1 / eta)`, as in the
# control as inference of Chapter 8.

# %%
slip_variance = 0.5
tilt = SemiringSum.Tilted(1 / eta)
_, policies, bayes_net = soft_plan(x0, 2, eta, variance, slips=slip_variance,
                                   at_states=tilt)
# The tilted conditional of the first action, with the hard prior on x0.
first_action = solve([policies[0], bayes_net.at(4).conditional()], 1)[0]
print("tilting the actions only:  first action", mean_plan[0])
print("tilting actions and slips: first action", first_action)
assert np.isclose(first_action, -0.557, atol=1e-3)

# With a tilt at every variable, the tilted distribution is an ordinary
# Gaussian factor graph: every reward -z^2 becomes a factor e^{-z^2 / eta}, a
# HessianFactor with G = 2 / eta (Chapter 8). Its solution is the tilted mean
# of every variable, and so the slip that the plan expects.
graph = GaussianFactorGraph()
graph.add(JacobianFactor(X(0), I, np.array([x0]), noise_model(0)))
for t in range(2):
    graph.add(JacobianFactor(U(t), I, zero, noise_model(variance)))
    graph.add(JacobianFactor(X(t + 1), I, X(t), -I, U(t), -I, zero,
                             noise_model(slip_variance)))
    for key in [X(t), U(t)]:
        graph.add(HessianFactor(key, 2 / eta * I, zero, 0.0))
graph.add(HessianFactor(X(2), 2 / eta * I, zero, 0.0))
mean = graph.optimize()
first_slip = mean.at(X(1))[0] - x0 - mean.at(U(0))[0]
print("as a Gaussian factor graph:  first action", mean.at(U(0))[0],
      " and it expects a first slip of", first_slip)
assert np.isclose(mean.at(U(0))[0], first_action)
assert np.isclose(first_slip, -0.835, atol=1e-3)

# The check in numpy, over z = (u0, w0, u1, w1), with x1 = x0 + u0 + w0 and
# x2 = x1 + u1 + w1.
steps = np.array([[1, 1, 0, 0], [1, 1, 1, 1.0]])
M_z = np.diag([1, 0, 1, 0.0]) + steps.T @ steps
b_z = steps.T @ np.ones(2) * x0
prior_precision = np.diag(
    [1 / variance, 1 / slip_variance, 1 / variance, 1 / slip_variance])
tilted_z = np.linalg.solve(prior_precision + 2 * M_z / eta, -2 * b_z / eta)
assert np.allclose(tilted_z[:2], [first_action, first_slip])

rng = np.random.default_rng(4)
samples = rng.normal(0.0, 1.0, (10 ** 6, 4)) * np.sqrt(
    [variance, slip_variance, variance, slip_variance])
x1 = x0 + samples[:, 0] + samples[:, 1]
x2 = x1 + samples[:, 2] + samples[:, 3]
noisy_returns = -(x0 ** 2 + samples[:, 0] ** 2 + x1 ** 2
                  + samples[:, 2] ** 2 + x2 ** 2)
_, weights, _ = mppi_estimate(noisy_returns, eta)
print("the same by sampling, M = 10^6: first action", weights @ samples[:, 0])
assert np.isclose(weights @ samples[:, 0], first_action, atol=0.02)
