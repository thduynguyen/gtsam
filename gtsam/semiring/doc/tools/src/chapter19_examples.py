# %% [markdown]
# # Chapter 19 examples: PILCO
#
# This notebook runs the examples of
# [Chapter 19](https://thduynguyen.github.io/gtsam/chapter19): a small version
# of PILCO. The dynamics factor is a Gaussian process, the forward message is
# kept Gaussian by moment matching, and the expected return is then a
# closed-form function of the policy parameter. The Gaussian process and the
# moment matching are in numpy; the exact reference they are tested against,
# on linear dynamics, is an elimination with the `gtsam/semiring` module.

# %%
import numpy as np
from gtsam import HessianFactor, JacobianFactor, Ordering, noiseModel
from gtsam import SemiringFactorGraph, SemiringGaussianFactor
from gtsam.symbol_shorthand import U, X
from scipy.optimize import minimize, minimize_scalar

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## The robot on a hill (Section 1)
#
# The origin is the top of a hill. The slope pushes the robot away from it:
# $x' = x + u + \sin x + w$, with $w \sim N(0, 0.1)$. The robot starts at
# $x_0 \sim N(2, 1)$, makes four moves with the policy $u = -K x$, and is
# rewarded as on the line: $-(x^2 + u^2)$ per move and $-x_4^2$ at the end.

# %%
MOVES, NOISE = 4, 0.1
START_MEAN, START_VARIANCE = 2.0, 1.0


def hill(x, u):
    """The mean of the next position."""
    return x + u + np.sin(x)


def collect(rng, episodes, gain=None, action_variance=4.0):
    """Run episodes on the real hill; return inputs (x, u) and changes x' - x.

    With gain=None the actions are random; otherwise u = -gain * x.
    """
    inputs, changes = [], []
    for _ in range(episodes):
        x = START_MEAN + np.sqrt(START_VARIANCE) * rng.normal()
        for t in range(MOVES):
            u = (np.sqrt(action_variance) * rng.normal() if gain is None
                 else -gain * x)
            x_next = hill(x, u) + np.sqrt(NOISE) * rng.normal()
            inputs.append([x, u]), changes.append(x_next - x)
            x = x_next
    return np.array(inputs), np.array(changes)


def true_rollout(gain, samples=200000, seed=100):
    """Monte Carlo on the real hill: J and the moments of x_t."""
    rng = np.random.default_rng(seed)
    x = START_MEAN + np.sqrt(START_VARIANCE) * rng.normal(size=samples)
    total = np.zeros(samples)
    moments = [(x.mean(), x.var())]
    for t in range(MOVES):
        u = -gain * x
        total -= x ** 2 + u ** 2
        x = hill(x, u) + np.sqrt(NOISE) * rng.normal(size=samples)
        moments.append((x.mean(), x.var()))
    return (total - x ** 2).mean(), np.array(moments)


for gain in [0.5, 1.0]:
    J, moments = true_rollout(gain)
    print(f"K = {gain}: true J = {J:.3f}, mean of x_t = {moments[:, 0]}")

# %% [markdown]
# ## A Gaussian-process dynamics factor (Section 2)
#
# The model predicts the change $x' - x$ from the input $z = (x, u)$, with a
# squared-exponential kernel.


# %%
class GaussianProcess:
    """GP regression with a squared-exponential kernel and fixed settings."""

    def __init__(self, inputs, targets, lengths, signal, noise):
        self.inputs, self.signal, self.noise = inputs, signal, noise
        self.inverse_lengths = 1 / np.asarray(lengths) ** 2  # Lambda^-1
        gram = self.kernel(inputs, inputs)
        self.solve = np.linalg.inv(gram + noise * np.eye(len(inputs)))
        self.weights = self.solve @ targets  # xi

    def kernel(self, a, b):
        d = a[:, None, :] - b[None, :, :]
        return self.signal * np.exp(
            -0.5 * np.einsum("ijk,k,ijk->ij", d, self.inverse_lengths, d))

    def predict(self, queries):
        """Mean and variance of the function at certain inputs."""
        k = self.kernel(queries, self.inputs)
        variance = self.signal - np.einsum("ij,jk,ik->i", k, self.solve, k)
        return k @ self.weights, variance

    def moments(self, mean, covariance):
        """Moment matching: the input is uncertain, z ~ N(mean, covariance).

        Returns the mean and variance of the predicted change (including the
        noise of the dynamics) and its covariance with the input.
        """
        dimension = len(mean)
        lengths = np.diag(1 / self.inverse_lengths)  # Lambda
        d = self.inputs - mean
        inverse = np.linalg.inv(covariance + lengths)
        # k_bar_i = E[k(z, z_i)]
        k_bar = self.signal / np.sqrt(np.linalg.det(
            covariance * self.inverse_lengths[None, :] + np.eye(dimension))
        ) * np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, inverse, d))
        predicted_mean = self.weights @ k_bar
        cross = covariance @ inverse @ (d.T @ (self.weights * k_bar))
        # Xi_ij = E[k(z, z_i) k(z, z_j)]
        centre = 0.5 * (self.inputs[:, None, :] + self.inputs[None, :, :]) - mean
        apart = self.inputs[:, None, :] - self.inputs[None, :, :]
        inverse2 = np.linalg.inv(covariance + 0.5 * lengths)
        second = self.signal ** 2 / np.sqrt(np.linalg.det(
            2 * covariance * self.inverse_lengths[None, :] + np.eye(dimension))
        ) * np.exp(-0.25 * np.einsum(
            "ijk,k,ijk->ij", apart, self.inverse_lengths, apart)
        ) * np.exp(-0.5 * np.einsum("ijk,kl,ijl->ij", centre, inverse2, centre))
        variance = (self.signal - np.trace(self.solve @ second) +
                    self.weights @ second @ self.weights -
                    predicted_mean ** 2 + self.noise)
        return predicted_mean, variance, cross


LENGTHS, SIGNAL = [1.5, 3.0], 4.0
rng = np.random.default_rng(0)
inputs, changes = collect(rng, 5)
model = GaussianProcess(inputs, changes, LENGTHS, SIGNAL, NOISE)
print(len(changes), "transitions with random actions")
for x in [0.0, 1.0, 2.0, 3.0]:
    mean, variance = model.predict(np.array([[x, -x]]))
    print(f"at x = {x}, u = {-x}: model {mean[0]:6.3f} +- "
          f"{np.sqrt(variance[0]):.3f},  true {hill(x, -x) - x:6.3f}")

# %% [markdown]
# ## Moment matching, checked by sampling (Section 3)
#
# For a Gaussian input, the mean and variance of the prediction have closed
# forms. They agree with the moments of many sampled inputs pushed through the
# model.

# %%
direction = np.array([1.0, -0.5])  # z = (x, u) = x * (1, -K) with K = 0.5
mean, covariance = 2.0 * direction, 1.0 * np.outer(direction, direction)
matched = model.moments(mean, covariance)

rng = np.random.default_rng(1)
x = 2.0 + rng.normal(size=400000)
sampled_mean, sampled_variance = model.predict(np.outer(x, direction))
sampled = (sampled_mean.mean(),
           sampled_variance.mean() + sampled_mean.var() + NOISE,
           np.cov(x, sampled_mean)[0, 1])
print("moment matching: mean %.4f  variance %.4f  cov(x, change) %.4f"
      % (matched[0], matched[1], matched[2][0]))
print("sampling:        mean %.4f  variance %.4f  cov(x, change) %.4f"
      % sampled)
assert np.allclose([matched[0], matched[1], matched[2][0]], sampled, atol=5e-3)

# %% [markdown]
# ## Stage 1: the forward message, step by step (Section 3)
#
# The state stays Gaussian, $x_t \sim N(\mu_t, \Sigma_t)$, and the expected
# return is a sum of closed-form terms.


# %%
def stage1(moments_of_change, gain, moves=MOVES, mean=START_MEAN,
           variance=START_VARIANCE):
    """Propagate N(mean, variance) and add up the expected rewards.

    moments_of_change(mean_z, covariance_z) returns the mean and variance of
    x' - x and its covariance with z = (x, u), for a Gaussian z.
    """
    direction = np.array([1.0, -gain])  # z = x * (1, -K)
    total, messages = 0.0, [(mean, variance)]
    for t in range(moves):
        total -= (1 + gain ** 2) * (mean ** 2 + variance)  # E[x^2 + u^2]
        change_mean, change_variance, cross = moments_of_change(
            mean * direction, variance * np.outer(direction, direction))
        mean = mean + change_mean
        variance = variance + change_variance + 2 * cross[0]
        messages.append((mean, variance))
    return total - (mean ** 2 + variance), np.array(messages)


# %% [markdown]
# ## An exact special case: the line (Section 5)
#
# For linear dynamics the next state of a Gaussian state is exactly Gaussian,
# so moment matching is exact. With the true linear model of the line,
# $x' = x + u + w$, $w \sim N(0, 0.5)$, two moves and $u = -0.5 x$, it must
# return the $J = -9.375$ of Chapter 1.


# %%
def linear_moments(F, B, noise):
    """Moments of the change x' - x = (F - 1) x + B u + w."""
    coefficients = np.array([F - 1.0, B])

    def moments(mean, covariance):
        return (coefficients @ mean,
                coefficients @ covariance @ coefficients + noise,
                covariance @ coefficients)

    return moments


J_line, messages_line = stage1(linear_moments(1.0, 1.0, 0.5), 0.5, moves=2)
print("the line, linear model:  J =", J_line)
print("forward messages (mean, variance):\n", messages_line)
assert np.isclose(J_line, -9.375)

# %% [markdown]
# The reference is the same line as a semiring factor graph, eliminated by
# the module. Its expectation is the exact return. The exact forward message
# on $x_t$ is read from the same graph without its rewards: the mean and the
# second moment of $x_t$ are the expectations of the "rewards" $x_t$ and
# $x_t^2$.

# %%
I = np.eye(1)
zero = np.zeros(1)


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def reward(key, G, g):
    """Lift the quadratic 0.5 G z^2 - g z on one variable to (1, value)."""
    return SemiringGaussianFactor.Reward(
        HessianFactor(key, G * I, np.array([g]), 0.0))


def line(gain, rewards):
    """The two-move line under u = -gain * x, with the given reward factors."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([START_MEAN]),
                             noiseModel.Isotropic.Variance(1, START_VARIANCE)))
    for t in range(2):
        graph.push_back(gaussian(U(t), I, X(t), gain * I, zero,
                                 noiseModel.Constrained.All(1)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 noiseModel.Isotropic.Variance(1, 0.5)))
    for factor in rewards:
        graph.push_back(factor)
    return graph


backward = Ordering()
for key in [X(2), U(1), X(1), U(0), X(0)]:
    backward.push_back(key)

# The return: the rewards -(x^2 + u^2) per move and -x^2 at the end.
penalties = [reward(key, -2.0, 0.0) for key in [X(0), U(0), X(1), U(1), X(2)]]
J_module = line(0.5, penalties).expectation(backward)
print("the line, by elimination:  J =", J_module)
assert np.isclose(J_module, -9.375) and np.isclose(J_line, J_module)

# The forward messages: E[x_t] and E[x_t^2] as expectations of rewards.
messages_module = []
for t in range(3):
    mean = line(0.5, [reward(X(t), 0.0, -1.0)]).expectation(backward)
    second = line(0.5, [reward(X(t), 2.0, 0.0)]).expectation(backward)
    messages_module.append((mean, second - mean ** 2))
print("forward messages by elimination (mean, variance):\n",
      np.array(messages_module))
assert np.allclose(messages_module, [[2, 1], [1, 0.75], [0.5, 0.6875]])
assert np.allclose(messages_line, messages_module)

# The same with a Gaussian process learned from 300 transitions of the line.
# This is no longer exact: the model itself has an error.
rng = np.random.default_rng(2)
line_x = 2.0 + rng.normal(size=300) * 1.5
line_u = 2.0 * rng.normal(size=300)
line_change = line_u + np.sqrt(0.5) * rng.normal(size=300)
line_model = GaussianProcess(np.stack([line_x, line_u], axis=1), line_change,
                             [6.0, 6.0], 9.0, 0.5)
J_line_gp, _ = stage1(line_model.moments, 0.5, moves=2)
print("the line, Gaussian-process model:  J =", J_line_gp)
assert abs(J_line_gp + 9.375) < 0.3

# %% [markdown]
# ## Where moment matching errs (Section 6)
#
# To separate the two sources of error, first use the *true* hill in place of
# the model. The moments of $\sin x$ under a Gaussian have closed forms, so
# the matched moments of one step are exact. The error that remains comes
# from assuming that the state is still Gaussian after a nonlinear step.


# %%
def hill_moments(mean, covariance):
    """Exact moments of the change u + sin(x) + w for a Gaussian z = (x, u)."""
    mu, var = mean[0], covariance[0, 0]
    damp = np.exp(-var / 2)
    sine = damp * np.sin(mu)  # E[sin x]
    sine_squared = 0.5 * (1 - np.exp(-2 * var) * np.cos(2 * mu))  # E[sin^2 x]
    cov_x_sine = var * damp * np.cos(mu)  # Cov[x, sin x]
    cov_u_sine = covariance[0, 1] / var * cov_x_sine if var > 0 else 0.0
    change_mean = mean[1] + sine
    change_variance = (covariance[1, 1] + sine_squared - sine ** 2 +
                       2 * cov_u_sine + NOISE)
    cross = covariance[:, 1] + np.array([cov_x_sine, cov_u_sine])
    return change_mean, change_variance, cross


for gain in [0.5, 1.0]:
    J_matched, matched = stage1(hill_moments, gain)
    J_true, true = true_rollout(gain)
    print(f"K = {gain}:  J matched {J_matched:.3f}   J true {J_true:.3f}")
    print("   step:             ", np.arange(MOVES + 1))
    print("   mean, matched:    ", matched[:, 0])
    print("   mean, true:       ", true[:, 0])
    print("   variance, matched:", matched[:, 1])
    print("   variance, true:   ", true[:, 1])
    # The first step is exact; later steps are not.
    assert np.allclose(matched[1], true[1], atol=0.01)
J_weak, matched_weak = stage1(hill_moments, 0.5)
_, true_weak = true_rollout(0.5)
assert true_weak[4, 1] > 2 * matched_weak[4, 1]

# %% [markdown]
# ## The loop: collect, learn, Stage 1, Stage 2 (Sections 4 and 7)
#
# Stage 2 maximizes the closed-form $J(K)$ of the model with a quasi-Newton
# optimizer. Each round then runs the new policy on the real hill for three
# episodes and refits the model.

# %%
best = minimize_scalar(lambda gain: -true_rollout(gain)[0],
                       bounds=(0, 2), method="bounded")
print(f"best gain on the real hill: K = {best.x:.3f}, J = {-best.fun:.3f}")

rng = np.random.default_rng(0)
inputs, changes = collect(rng, 5)
gain, history = 0.0, []
for round_ in range(5):
    model = GaussianProcess(inputs, changes, LENGTHS, SIGNAL, NOISE)  # learn
    result = minimize(lambda g: -stage1(model.moments, g[0])[0], [gain],
                      method="BFGS")  # stage 2, around stage 1
    gain, predicted = float(result.x[0]), -result.fun
    true = true_rollout(gain)[0]
    history.append((len(changes), gain, predicted, true))
    print(f"round {round_}: {len(changes):2d} transitions,  K = {gain:.3f},  "
          f"predicted J = {predicted:7.3f},  true J = {true:7.3f}")
    new_inputs, new_changes = collect(rng, 3, gain=gain)  # collect
    inputs = np.vstack([inputs, new_inputs])
    changes = np.concatenate([changes, new_changes])
assert abs(history[-1][3] - (-best.fun)) < 0.1
assert history[-1][3] > history[0][3]

# %% [markdown]
# The forward message of the final model, against the real hill.

# %%
J_model, matched = stage1(model.moments, gain)
J_true, true = true_rollout(gain)
print(f"K = {gain:.3f}")
print("mean, model:    ", matched[:, 0])
print("mean, true:     ", true[:, 0])
print("variance, model:", matched[:, 1])
print("variance, true: ", true[:, 1])
