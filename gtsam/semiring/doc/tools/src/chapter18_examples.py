# %% [markdown]
# # Chapter 18 examples: learning the dynamics factor
#
# This notebook runs the examples of
# [Chapter 18](https://thduynguyen.github.io/gtsam/chapter18). The dynamics of
# the line of Chapter 1 are estimated from sampled transitions with a GTSAM
# factor graph, and the learned factor is then used for control as if it were
# the true one.

# %%
import gtsam
import numpy as np
from gtsam import HessianFactor, JacobianFactor, Ordering, noiseModel
from gtsam import SemiringFactorGraph, SemiringGaussianFactor
from gtsam import SemiringRules, SemiringSum
from gtsam.symbol_shorthand import U, X

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The line, and transitions sampled from it (Section 1)
#
# The true dynamics are $x' = F x + B u + w$ with $F = B = 1$ and
# $w \sim N(0, 0.5)$. The learner does not know $F$, $B$ or the noise
# variance. It sees transitions $(x, u, x')$.

# %%
F_TRUE, B_TRUE, SIGMA_W = 1.0, 1.0, 0.5


def sample_transitions(rng, count, gain=None, jitter=1.0):
    """Two-move episodes from x0 ~ N(2, 1); returns arrays x, u, x'.

    With gain=None the actions are random, u ~ N(0, jitter). Otherwise they
    follow the policy u = -gain * x + e, e ~ N(0, jitter).
    """
    xs, us, ns = [], [], []
    while len(xs) < count:
        x = 2.0 + rng.normal()
        for t in range(2):
            mean = 0.0 if gain is None else -gain * x
            u = mean + np.sqrt(jitter) * rng.normal()
            n = F_TRUE * x + B_TRUE * u + np.sqrt(SIGMA_W) * rng.normal()
            xs.append(x), us.append(u), ns.append(n)
            x = n
    return (np.array(xs[:count]), np.array(us[:count]), np.array(ns[:count]))


rng = np.random.default_rng(6)
x, u, x_next = sample_transitions(rng, 20)
print("first transitions (x, u, x'):")
print(np.stack([x, u, x_next], axis=1)[:4])

# %% [markdown]
# ## The fit as a factor graph (Section 2)
#
# One variable, $\theta_p = (F, B)$, and one factor per transition with the
# error $x' - F x - B u$. This is the graph of a calibration problem.

# %%
THETA_P = gtsam.symbol("p", 0)


def transition_factor(x, u, x_next, variance):
    """A factor on theta_p = (F, B) with error x' - F x - B u."""

    def error(this, values, jacobians):
        theta_p = values.atVector(this.keys()[0])
        if jacobians is not None:
            jacobians[0] = np.array([[-x, -u]])
        return np.array([x_next - theta_p[0] * x - theta_p[1] * u])

    return gtsam.CustomFactor(
        noiseModel.Isotropic.Variance(1, variance), [THETA_P], error)


def fit(x, u, x_next):
    """Estimate theta_p = (F, B), the noise variance, and cov(theta_p)."""

    def solve(variance):
        graph = gtsam.NonlinearFactorGraph()
        for i in range(len(x)):
            graph.add(transition_factor(x[i], u[i], x_next[i], variance))
        initial = gtsam.Values()
        initial.insert(THETA_P, np.zeros(2))
        result = gtsam.GaussNewtonOptimizer(graph, initial).optimize()
        return graph, result

    # The estimate of (F, B) does not depend on the noise variance, so fit
    # once with variance 1, estimate the variance from the residuals, and
    # build the graph again with it to get the covariance.
    _, result = solve(1.0)
    theta_p = result.atVector(THETA_P)
    residual = x_next - theta_p[0] * x - theta_p[1] * u
    variance = residual @ residual / (len(x) - 2)
    graph, result = solve(variance)
    covariance = gtsam.Marginals(graph, result).marginalCovariance(THETA_P)
    return theta_p, variance, covariance


theta_p, variance, covariance = fit(x, u, x_next)
print("20 transitions with random actions")
print("  F, B estimated :", theta_p)
print("  noise variance :", variance)
print("  covariance of (F, B) =\n", covariance)
print("  standard deviations:", np.sqrt(np.diag(covariance)))

# The same numbers from the normal equations.
Z = np.stack([x, u], axis=1)
check = np.linalg.solve(Z.T @ Z, Z.T @ x_next)
assert np.allclose(theta_p, check)
assert np.allclose(covariance, variance * np.linalg.inv(Z.T @ Z))

# %% [markdown]
# More data shrinks the uncertainty about $(F, B)$ and leaves the noise of
# the dynamics where it is.

# %%
rng = np.random.default_rng(6)
x_all, u_all, n_all = sample_transitions(rng, 10000)
for count in [6, 20, 100, 1000, 10000]:
    theta_p, variance, covariance = fit(
        x_all[:count], u_all[:count], n_all[:count])
    print(f"M = {count:5d}:  F = {theta_p[0]:.3f}  B = {theta_p[1]:.3f}  "
          f"noise variance = {variance:.3f}  "
          f"std of F, B = {np.sqrt(np.diag(covariance))}")
assert np.allclose(theta_p, [1, 1], atol=0.02) and abs(variance - 0.5) < 0.02

# %% [markdown]
# ## What the data can and cannot tell (Section 2)
#
# If the transitions come from a policy with little jitter, $u \approx -0.5 x$,
# the two columns of the regression are nearly proportional. The combination
# $F - 0.5 B$ is well determined, and $F$ and $B$ separately are not.

# %%
rng = np.random.default_rng(0)
x_pol, u_pol, n_pol = sample_transitions(rng, 100, gain=0.5, jitter=0.01)
theta_pol, variance_pol, covariance_pol = fit(x_pol, u_pol, n_pol)
closed_loop = np.array([1.0, -0.5])
print("100 transitions from the policy u = -0.5 x + small jitter")
print("  F, B estimated:", theta_pol)
print("  std of F, B   :", np.sqrt(np.diag(covariance_pol)))
print("  F - 0.5 B     :", closed_loop @ theta_pol, "+-",
      np.sqrt(closed_loop @ covariance_pol @ closed_loop))
theta_rand, _, covariance_rand = fit(x_all[:100], u_all[:100], n_all[:100])
print("100 transitions with random actions")
print("  std of F, B   :", np.sqrt(np.diag(covariance_rand)))
assert np.sqrt(covariance_pol[1, 1]) > 5 * np.sqrt(covariance_rand[1, 1])
assert np.sqrt(closed_loop @ covariance_pol @ closed_loop) < 0.1

# %% [markdown]
# ## The line as a semiring factor graph
#
# Every exact computation below is an elimination of this graph: with the
# learned or the true dynamics factor, and with a policy factor or with the
# actions left free.

# %%
I = np.eye(1)
zero = np.zeros(1)


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


def dynamics_factor(t, F, B, variance):
    """x' - F x - B u = w, with w of the given variance."""
    return gaussian(X(t + 1), I, X(t), -F * I, U(t), -B * I, zero,
                    noiseModel.Isotropic.Variance(1, variance))


def policy_factor(t, gain):
    """The deterministic policy u = -gain * x, as a hard constraint."""
    return gaussian(U(t), I, X(t), gain * I, zero,
                    noiseModel.Constrained.All(1))


def start_factor(mean=2.0, variance=1.0):
    """The first state: x0 ~ N(mean, variance), or x0 = mean if variance is 0."""
    model = (noiseModel.Constrained.All(1) if variance == 0 else
             noiseModel.Isotropic.Variance(1, variance))
    return gaussian(X(0), I, np.array([mean]), model)


def line(F, B, variance, gains=None, start=None, rewards=True):
    """The two-move line as a semiring factor graph.

    With gains=None there is no policy factor and the actions are free.
    """
    graph = SemiringFactorGraph()
    graph.push_back(start if start is not None else start_factor())
    for t in range(2):
        if gains is not None:
            graph.push_back(policy_factor(t, gains[t]))
        graph.push_back(dynamics_factor(t, F, B, variance))
        if rewards:
            graph.push_back(penalty(X(t)))
            graph.push_back(penalty(U(t)))
    if rewards:
        graph.push_back(penalty(X(2)))
    return graph


BACKWARD = ordering(X(2), U(1), X(1), U(0), X(0))

# %% [markdown]
# ## Two kinds of uncertainty in a prediction (Section 3)
#
# Predict the position $x_2$ after two moves of the policy $u = -0.5 x$ from
# $x_0 = 2$. For one linear model the prediction is exact: the forward message
# on $x_2$ is a Gaussian, and its mean and second moment are the expectations
# of the "rewards" $x_2$ and $x_2^2$ on the graph of that model.

# %%
GAIN, START = 0.5, 2.0


def forward_message(F, B, noise_variance):
    """Mean and variance of x2 under one linear model, by elimination."""
    moments = []
    for G, g in [(np.zeros((1, 1)), np.array([-1.0])),  # the value x2
                 (2 * I, zero)]:  # the value x2^2
        graph = line(F, B, noise_variance, gains={0: GAIN, 1: GAIN},
                     start=start_factor(START, 0.0), rewards=False)
        graph.push_back(
            SemiringGaussianFactor.Reward(HessianFactor(X(2), G, g, 0.0)))
        moments.append(graph.expectation(BACKWARD))
    return moments[0], moments[1] - moments[0] ** 2


def member_predictions(F, B, noise_variance):
    """The same two numbers in closed form, for arrays of models F, B."""
    a = F - B * GAIN  # the closed loop: x' = a x + w
    return a ** 2 * START, (a ** 2 + 1) * noise_variance


true_mean, true_variance = forward_message(F_TRUE, B_TRUE, SIGMA_W)
print(f"true system: mean {true_mean:.3f}, variance {true_variance:.3f}")
assert np.allclose([true_mean, true_variance],
                   member_predictions(F_TRUE, B_TRUE, SIGMA_W))
assert np.isclose(true_variance, 0.625)

# %% [markdown]
# Now draw many models $(F, B)$ from the estimate and its covariance, an
# *ensemble*. Each member predicts a mean and a variance for $x_2$. The
# average of the members' variances is the noise of the dynamics; the spread
# of the members' means is the uncertainty about the dynamics. The members are
# evaluated with the closed form, which the cell above checked against the
# elimination.

# %%
rng = np.random.default_rng(1)
decomposition = {}
for count in [20, 1000]:
    theta_p, variance, covariance = fit(
        x_all[:count], u_all[:count], n_all[:count])
    members = rng.multivariate_normal(theta_p, covariance, size=100000)
    means, variances = member_predictions(
        members[:, 0], members[:, 1], variance)
    point_mean, point_variance = forward_message(*theta_p, variance)
    assert np.allclose([point_mean, point_variance],
                       member_predictions(*theta_p, variance))
    decomposition[count] = (variances.mean(), means.var())
    print(f"M = {count:4d}: point estimate predicts mean {point_mean:.3f}, "
          f"variance {point_variance:.3f}")
    print(f"          ensemble: mean {means.mean():.3f}, "
          f"noise of the dynamics {variances.mean():.3f}, "
          f"uncertainty about the dynamics {means.var():.4f}")
assert decomposition[1000][1] < 0.05 * decomposition[20][1]
assert abs(decomposition[1000][0] - true_variance) < 0.05

# %% [markdown]
# The uncertainty about the dynamics is one draw of $(F, B)$ that holds for
# the whole rollout. Treating it as fresh noise at every move, by drawing a
# new model at each step, loses that and predicts a smaller spread.

# %%
theta20, variance20, covariance20 = fit(x_all[:20], u_all[:20], n_all[:20])
SAMPLES = 400000


def rollout(rng, draw_model):
    """Sample x2 after two moves; draw_model() returns arrays F, B."""
    position = np.full(SAMPLES, START)
    for step in range(2):
        F, B = draw_model()
        position = (F - B * GAIN) * position + np.sqrt(variance20) * \
            rng.normal(size=SAMPLES)
    return position


rng = np.random.default_rng(2)
one_model = rng.multivariate_normal(theta20, covariance20, size=SAMPLES)
shared = rollout(rng, lambda: (one_model[:, 0], one_model[:, 1]))


def fresh():
    draw = rng.multivariate_normal(theta20, covariance20, size=SAMPLES)
    return draw[:, 0], draw[:, 1]


independent = rollout(rng, fresh)
print(f"one model per rollout:     variance of x2 = {shared.var():.3f}")
print(f"a new model at every move: variance of x2 = {independent.var():.3f}")
assert shared.var() > independent.var()

# %% [markdown]
# ## Stage 1 on the learned factor: certainty-equivalent control (Section 4)
#
# The backward pass of Chapter 6 is one elimination of the graph with the
# *learned* dynamics factor and no policy factor: the average at the states
# and the maximum at the actions. The conditionals it leaves on the actions
# hold the gains. The gains are then evaluated on the *true* system, by one
# more elimination, with the policy as a factor.

# %%
MAXIMUM_AT_ACTIONS = SemiringRules()
MAXIMUM_AT_ACTIONS.setAll([U(0), U(1)], SemiringSum.Maximum())


def best_gains(F, B, variance):
    """The best gains for the given dynamics, and the return they promise."""
    graph = line(F, B, variance)
    bayes_net = graph.eliminateSequential(BACKWARD, MAXIMUM_AT_ACTIONS)
    # The conditional on u_t is u + K_t x = 0; K_t is the block on its parent.
    gains = {1: float(bayes_net.at(1).conditional().S()[0, 0]),
             0: float(bayes_net.at(3).conditional().S()[0, 0])}
    return gains, graph.expectation(BACKWARD, MAXIMUM_AT_ACTIONS)


def evaluate(gains, F=F_TRUE, B=B_TRUE, variance=SIGMA_W):
    """Expected return of u_t = -gains[t] x_t, by elimination."""
    return line(F, B, variance, gains).expectation(BACKWARD)


gains, J_star = best_gains(F_TRUE, B_TRUE, SIGMA_W)
print("true model:    gains", gains, " J* =", J_star)
assert np.isclose(gains[0], 0.6) and np.isclose(gains[1], 0.5)
assert np.isclose(J_star, -9.25) and np.isclose(evaluate(gains), -9.25)
assert np.isclose(evaluate({0: 0.5, 1: 0.5}), -9.375)

# %% [markdown]
# The learned controller, as the amount of data grows. `predicted` is the
# return the learned model promises; `true` is what the controller collects on
# the real system.

# %%
for count in [6, 20, 100, 1000, 10000]:
    theta_p, variance, _ = fit(x_all[:count], u_all[:count], n_all[:count])
    gains, predicted = best_gains(theta_p[0], theta_p[1], variance)
    print(f"M = {count:5d}:  K0 = {gains[0]:.3f}  K1 = {gains[1]:.3f}  "
          f"predicted J = {predicted:7.3f}  true J = {evaluate(gains):7.4f}")
assert abs(gains[0] - 0.6) < 0.01 and abs(gains[1] - 0.5) < 0.01
assert abs(evaluate(gains) + 9.25) < 1e-3

# %% [markdown]
# ## What model error costs (Section 4)
#
# Averaged over 300 independent data sets of each size. The loss
# $J^* - J$ falls in proportion to $1 / M$: it is quadratic in the error of
# the gains, and that error falls like $1 / \sqrt{M}$.


# %%
rng = np.random.default_rng(3)
losses = {}
for count in [6, 20, 100, 1000]:
    loss = []
    for trial in range(300):
        xs, us, ns = sample_transitions(rng, count)
        Z = np.stack([xs, us], axis=1)
        F_hat, B_hat = np.linalg.solve(Z.T @ Z, Z.T @ ns)
        # The gains do not depend on the noise variance of the model.
        learned_gains, _ = best_gains(F_hat, B_hat, SIGMA_W)
        loss.append(-9.25 - evaluate(learned_gains))
    losses[count] = np.mean(loss)
    print(f"M = {count:5d}:  mean loss J* - J = {losses[count]:.5f}   "
          f"M * loss = {count * losses[count]:.3f}")
assert losses[1000] < losses[100] < losses[20] < losses[6]
assert 0.5 < (100 * losses[100]) / (1000 * losses[1000]) < 2.0
assert np.allclose([losses[6], losses[20], losses[100], losses[1000]],
                   [0.870, 0.055, 0.0106, 0.00086], rtol=0.02)
