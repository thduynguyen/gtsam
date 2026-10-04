# %% [markdown]
# # Chapter 8 examples: softness and risk
#
# This notebook runs the examples of
# [Chapter 8](https://thduynguyen.github.io/gtsam/chapter08) with the
# `gtsam/semiring` module: the tilted sum of Chapter 2 applied to the actions
# (a soft maximum), to the states (an attitude toward risk), and to both
# (control as inference), on the track and on the line of Chapter 1. Each case
# is one factor graph and a rule for summing out each variable.

# %%
import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, GaussianFactorGraph
from gtsam import HessianFactor, JacobianFactor, Ordering, noiseModel
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam import SemiringGaussianFactor, SemiringRules, SemiringSum
from gtsam.symbol_shorthand import A, S, U, X

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track of Chapter 1 (Section 1)

# %%
L, R = 0, 1
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
base = np.full((3, 2), 0.5)  # the base policy pi(a | s): a coin flip
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
move_reward = np.array([[0.0, -1.0]] * 3)  # r(s, a)
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)

state = lambda t: (S(t), 3)  # (key, cardinality)
action = lambda t: (A(t), 2)


def probability(keys, table):
    """Lift a probability table to (p, 0)."""
    return SemiringDiscreteFactor(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def value(keys, table):
    """Lift a reward table to (1, r)."""
    return SemiringDiscreteFactor.Reward(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def ordering(*keys):
    """An ordering of the given keys."""
    result = Ordering()
    for key in keys:
        result.push_back(key)
    return result


def table(factor, keys):
    """Read a DecisionTreeFactor into an array indexed in the order of keys."""
    result = np.zeros([cardinality for _, cardinality in keys])
    for index in np.ndindex(*result.shape):
        values = DiscreteValues()
        for (key, _), index_of_key in zip(keys, index):
            values[key] = index_of_key
        result[index] = factor(values)
    return result


def track(policies=(base, base), move_rewards=(move_reward, move_reward),
          final=final_reward):
    """The factor graph of the track for one policy table per move."""
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    for t in range(2):
        keys = [state(t), action(t)]
        graph.push_back(probability(keys, policies[t]))
        graph.push_back(probability(keys + [state(t + 1)], dynamics))
        graph.push_back(value(keys, move_rewards[t]))
    graph.push_back(value([state(2)], final))
    return graph


states, actions = [S(0), S(1), S(2)], [A(0), A(1)]
backward = ordering(S(2), A(1), S(1), A(0), S(0))  # backward in time


def rules(at_actions, at_states=SemiringSum.Average()):
    """One rule for the actions and one for the states."""
    result = SemiringRules()
    result.setAll(actions, at_actions)
    result.setAll(states, at_states)
    return result


def evaluate(policies):
    """Plain expected return of a policy given as one table per move."""
    return track(policies).expectation()


# %% [markdown]
# ## Soft maximum over the actions, average over the states (Section 2)
#
# The backward pass of Chapter 4 with the maximum replaced by the soft
# maximum of Chapter 2: the actions get the rule `SoftMaximum(eta)` and the
# states keep the average. The conditional of each action, tilted, is the soft
# policy $q$.


# %%
def tilted_policies(bayes_net, eta):
    """The tilted conditionals of the two actions, as tables q_t(a | s)."""
    return [table(bayes_net.at(position).tilted(1 / eta),
                  [state(t), action(t)])
            for t, position in [(0, 3), (1, 1)]]


def soft_control(eta):
    """Soft value at the root, and the soft policy of each move."""
    soft = rules(SemiringSum.SoftMaximum(eta))
    graph = track()
    bayes_net = graph.eliminateSequential(backward, soft)
    return graph.expectation(backward, soft), tilted_policies(bayes_net, eta)


def expected_kl(policies):
    """Sum over the moves of the expected KL(q || base) along the policy.

    The term log(q / base) is a reward factor on (s, a), so the total is the
    expectation of a graph that has these rewards and the policy q.
    """
    log_ratios = [np.log(np.where(q > 0, q / base, 1.0)) for q in policies]
    return track(policies, log_ratios, np.zeros(3)).expectation()


soft_value, soft_policies = soft_control(1.0)
print("eta = 1: soft value =", soft_value)
print("soft policy of the first move =\n", soft_policies[0])
print("soft policy of the last move =\n", soft_policies[1])
true_return = evaluate(soft_policies)
kl = expected_kl(soft_policies)
print("its plain return:", true_return, " its KL from the coin flip:", kl)
print("return - eta * KL:", true_return - 1.0 * kl)
assert np.isclose(soft_value, 4.7536, atol=1e-4)
assert np.isclose(true_return, 6.0284, atol=1e-4)
assert np.isclose(kl, 1.2748, atol=1e-4)
assert np.isclose(true_return - 1.0 * kl, soft_value)
assert np.allclose(soft_policies[1][0], [0.7311, 0.2689], atol=1e-4)

# %% [markdown]
# ## The tilt on every variable: control as inference (Section 3)
#
# Give every variable, the states included, the same tilted rule. With
# $\kappa = 1 / \eta$ this is inference in the graph whose rewards have been
# turned into the factors $e^{r / \eta}$.


# %%
def inference_value(eta, order=backward):
    tilt = SemiringSum.Tilted(1 / eta)
    return track().expectation(order, rules(tilt, tilt))


print("eta = 2:  ", inference_value(2.0))
print("eta = 0.5:", inference_value(0.5))
assert np.isclose(inference_value(2.0), 5.4251, atol=1e-4)  # Chapter 2
assert np.isclose(inference_value(0.5), 7.6490, atol=1e-4)
# One sum for all variables: any elimination order gives the same result.
other_order = ordering(A(0), S(0), S(1), S(2), A(1))
assert np.isclose(inference_value(2.0, other_order), inference_value(2.0))

# %% [markdown]
# The Bayes net of the same elimination gives the policy that inference
# produces: the tilted conditional of each action.


# %%
def inference_control(eta):
    """Tilt at the states too: the claimed value and the policies."""
    tilt = SemiringSum.Tilted(1 / eta)
    bayes_net = track().eliminateSequential(backward, rules(tilt, tilt))
    return inference_value(eta), tilted_policies(bayes_net, eta), bayes_net


claimed, inference_policies, inference_net = inference_control(2.0)

# %% [markdown]
# Inference tilts the dynamics as well. At the last move, in cell 2 after
# Left, the robot really slips and stays with probability 0.2. The conditional
# that inference leaves on the next state, tilted, says otherwise:

# %%
eta = 2.0
last_state = inference_net.at(0)  # the conditional on s2 given s1 and a1
keys = [state(1), action(1), state(2)]
true_dynamics = table(last_state.probability(), keys)[2, L]
tilted_dynamics = table(last_state.tilted(1 / eta), keys)[2, L]
print("true dynamics p(s' | 2, L):     ", true_dynamics)
print("as inference sees them, eta = 2:", tilted_dynamics)
assert np.allclose(true_dynamics, dynamics[2, L])
assert np.allclose(tilted_dynamics, [0, 0.0262, 0.9738], atol=1e-4)

# %% [markdown]
# ## Three numbers per temperature (Section 3)
#
# For each temperature: the soft value with the true dynamics and the plain
# return of its policy; the value that inference claims and the plain return
# that its policy really collects. The best any policy can do is $J^* = 6.1$.

# %%
J_star = track().expectation(backward, rules(SemiringSum.Maximum()))
assert np.isclose(J_star, 6.1)  # Chapter 4
results = {}
for eta in [10.0, 2.0, 1.0, 0.5, 0.2, 0.05]:
    soft_value, soft_policies = soft_control(eta)
    claimed, inference_policies, _ = inference_control(eta)
    results[eta] = (soft_value, evaluate(soft_policies), claimed,
                    evaluate(inference_policies))
    print(f"eta = {eta:5.2f}: soft value {results[eta][0]:7.4f}, its policy "
          f"really collects {results[eta][1]:7.4f} | inference claims "
          f"{results[eta][2]:7.4f}, its policy really collects "
          f"{results[eta][3]:7.4f}")
    assert results[eta][1] <= J_star + 1e-9
    assert results[eta][3] <= J_star + 1e-9
    assert results[eta][0] <= results[eta][1] + 1e-9  # the KL is a penalty
    assert results[eta][0] <= results[eta][2] + 1e-9  # a bound on the claim
expected = {10.0: (1.864, 2.353, 2.237, 2.374),
            2.0: (3.672, 5.390, 5.425, 4.716),
            1.0: (4.754, 6.028, 6.825, 4.556),
            0.5: (5.413, 6.088, 7.649, 3.893),
            0.2: (5.823, 6.099, 8.363, 3.198),
            0.05: (6.031, 6.100, 8.839, 3.150)}
for eta, row in expected.items():
    assert np.allclose(results[eta], row, atol=1e-3)
# As eta -> 0 the soft policy becomes the best one. Its soft value is below
# J* by at most eta times the largest possible KL, 2 log 2.
assert np.isclose(results[0.05][1], 6.1, atol=1e-3)
assert 6.1 - 0.05 * 2 * np.log(2) - 1e-6 <= results[0.05][0] <= 6.1
assert results[0.05][2] > 8.8  # inference -> the best trajectory, 9
assert results[0.05][3] < results[0.05][1] - 1.0  # and its policy is worse
print("last move of the inference policy at eta = 0.05 =\n",
      inference_control(0.05)[1][1])

# %% [markdown]
# ## Linear in the tilted value (Section 3)
#
# With the tilt on every variable, the backup is a matrix product on
# $e^{V / \eta}$. This cell is plain linear algebra, as an independent check
# of the number the module gave above.

# %%
eta = 2.0
Lambda = np.einsum("sa,sa,sat->st", base, np.exp(move_reward / eta),
                   dynamics)
z = np.exp(final_reward / eta)  # e^{V_2 / eta}
z = Lambda @ (Lambda @ z)  # two backups
print("matrix Lambda =\n", Lambda)
print("eta * log(prior . z) =", eta * np.log(prior @ z))
assert np.isclose(eta * np.log(prior @ z), inference_value(2.0))

# %% [markdown]
# ## Risk on the line: LEQG (Section 4)
#
# $x' = x + u + w$ with $w \sim N(0, 0.5)$, reward $-(x^2 + u^2)$ per move and
# $-x_2^2$ at the end, $x_0 \sim N(2, 1)$. There is no policy factor: the
# actions are summed out by maximum and the states by the tilted mean.

# %%
Sigma_w, mu_0, Sigma_0 = 0.5, 2.0, 1.0
I, zero = np.eye(1), np.zeros(1)
line_backward = ordering(X(2), U(1), X(1), U(0), X(0))


def gaussian(*args):
    """Lift a Gaussian factor to (p, 0)."""
    return SemiringGaussianFactor(JacobianFactor(*args))


def penalty(key):
    """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
    return SemiringGaussianFactor.Cost(HessianFactor(key, 2 * I, zero, 0.0))


def line(gains=None):
    """The factor graph of the line; with gains, the policy u = -K_t x."""
    graph = SemiringFactorGraph()
    graph.push_back(gaussian(X(0), I, np.array([mu_0]),
                             noiseModel.Isotropic.Variance(1, Sigma_0)))
    for t in range(2):
        if gains is not None:  # the hard constraint u + K x = 0
            graph.push_back(gaussian(U(t), I, X(t), gains[t] * I, zero,
                                     noiseModel.Constrained.All(1)))
        graph.push_back(gaussian(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 noiseModel.Isotropic.Variance(1, Sigma_w)))
        graph.push_back(penalty(X(t)))
        graph.push_back(penalty(U(t)))
    graph.push_back(penalty(X(2)))
    return graph


def line_rules(kappa, at_actions=SemiringSum.Maximum()):
    """A tilt at the states and a rule, by default the maximum, at the actions."""
    result = SemiringRules()
    result.setAll([U(0), U(1)], at_actions)
    result.setAll([X(0), X(1), X(2)], SemiringSum.Tilted(kappa))
    return result


def leqg(kappa):
    """Gains, and the tilted value at the root, for a tilt kappa.

    Returns None where the tilted mean is infinite: the module then raises.
    """
    try:
        bayes_net = line().eliminateSequential(line_backward,
                                               line_rules(kappa))
        value = line().expectation(line_backward, line_rules(kappa))
    except ValueError:
        return None
    # The conditional of an action is u + K x = 0: the gain is its S block.
    K = {t: bayes_net.at(position).conditional().S()[0, 0]
         for t, position in [(1, 1), (0, 3)]}
    return K, value


def tilted_value(K, kappa):
    """(1 / kappa) log E[exp(kappa R)] for the policy u = -K_t x."""
    try:
        return line(K).expectation(
            line_backward, line_rules(kappa, SemiringSum.Average()))
    except ValueError:
        return -np.inf if kappa < 0 else np.inf


def plain_return(K):
    """Expected return of the policy u = -K_t x on the real system."""
    return line(K).expectation(line_backward)


# %% [markdown]
# Two independent checks of the module's numbers, in numpy: the recursion of
# the chapter, and the tilted value of a linear policy as one Gaussian
# integral over the three random numbers $(x_0, w_0, w_1)$.


# %%
def leqg_recursion(kappa):
    """The risk-sensitive Riccati recursion of Section 4."""
    P, beta, K = 1.0, 0.0, {}
    for t in [1, 0]:
        P_tilted = P / (1 + 2 * kappa * Sigma_w * P)  # the tilted matrix
        beta += (np.log(1 + 2 * kappa * Sigma_w * P) / (2 * kappa)
                 if kappa != 0 else P * Sigma_w)
        K[t] = P_tilted / (1 + P_tilted)
        P = 1 + P_tilted - P_tilted ** 2 / (1 + P_tilted)
    P_tilted = P / (1 + 2 * kappa * Sigma_0 * P)
    beta += (np.log(1 + 2 * kappa * Sigma_0 * P) / (2 * kappa)
             if kappa != 0 else P * Sigma_0)
    return K, -(P_tilted * mu_0 ** 2 + beta)


def exact_tilted_value(K, kappa):
    """The same as tilted_value, by a Gaussian integral over (x0, w0, w1)."""
    # x0, x1, x2 and u0, u1 as linear functions of z, and R = -z' M z.
    x0 = np.array([1.0, 0, 0])
    x1 = (1 - K[0]) * x0 + np.array([0, 1.0, 0])
    x2 = (1 - K[1]) * x1 + np.array([0, 0, 1.0])
    M = sum(np.outer(row, row) for row in
            [x0, K[0] * x0, x1, K[1] * x1, x2])
    covariance = np.diag([Sigma_0, Sigma_w, Sigma_w])
    m = np.array([mu_0, 0, 0])
    if kappa == 0:
        return -(m @ M @ m + np.trace(M @ covariance))
    N = np.eye(3) + 2 * kappa * covariance @ M
    if np.linalg.eigvals(N).real.min() <= 0:
        return -np.inf if kappa < 0 else np.inf
    log_expectation = (-0.5 * np.log(np.linalg.det(N))
                       - kappa * m @ M @ np.linalg.solve(N, m))
    return log_expectation / kappa


expected = {-0.25: (0.7213, 0.5714, -54.93, -9.443),
            -0.2: (0.6931, 0.5556, -25.30, -9.364),
            -0.1: (0.6430, 0.5263, -13.14, -9.275),
            0.0: (0.6, 0.5, -9.25, -9.25),
            0.5: (0.4516, 0.4, -4.20, -9.565),
            1.0: (0.3636, 0.3333, -2.89, -10.089),
            2.0: (0.2632, 0.25, -1.87, -11.070)}
for kappa, (K0, K1, J_kappa, J_plain) in expected.items():
    K, value = leqg(kappa)
    print(f"kappa = {kappa:5.2f}: K0 = {K[0]:.4f}, K1 = {K[1]:.4f}, tilted "
          f"value {value:8.4f}, plain return {plain_return(K):.4f}")
    assert np.allclose([K[0], K[1]], [K0, K1], atol=1e-4)
    assert np.isclose(value, J_kappa, atol=6e-3)
    assert np.isclose(plain_return(K), J_plain, atol=1e-3)
    # The module agrees with the recursion and with the Gaussian integral.
    K_recursion, value_recursion = leqg_recursion(kappa)
    assert np.allclose([K[0], K[1]], [K_recursion[0], K_recursion[1]])
    assert np.isclose(value, value_recursion)
    assert np.isclose(value, exact_tilted_value(K, kappa))
    assert np.isclose(value, tilted_value(K, kappa))
    assert np.isclose(plain_return(K), exact_tilted_value(K, 0))
    # The gains maximize the tilted value: any change lowers it.
    for dK0, dK1 in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
        changed = {0: K[0] + dK0, 1: K[1] + dK1}
        assert tilted_value(changed, kappa) < value
        assert np.isclose(tilted_value(changed, kappa),
                          exact_tilted_value(changed, kappa))

K, value = leqg(0.0)
assert np.isclose(K[0], 0.6) and np.isclose(K[1], 0.5)
assert np.isclose(value, -9.25)  # Chapter 6
K, value = leqg(1e-6)
assert np.isclose(K[0], 0.6, atol=1e-5) and np.isclose(value, -9.25, atol=1e-4)

# %% [markdown]
# The breakdown: below a critical tilt the tilted mean is infinite, and the
# module raises an exception in place of a value.

# %%
kappas = np.linspace(-0.6, 0, 6001)
# The recursion exists at -0.25 and not at -0.3; search between them.
candidates = [kappa for kappa in kappas if -0.3 <= kappa <= -0.25]
critical = min(kappa for kappa in candidates if leqg(kappa) is not None)
print("the recursion exists down to kappa =", round(critical, 4))
K, value = leqg(critical)
print(f"there: K0 = {K[0]:.4f}, K1 = {K[1]:.4f}, tilted value {value:.2f}")
assert leqg(-0.3) is None and leqg(-0.25) is not None
assert np.isclose(critical, -0.287, atol=1e-3) and value < -30000
try:
    line().expectation(line_backward, line_rules(-0.3))
except ValueError as error:
    print("at kappa = -0.3 the module says:", str(error).split(": ", 1)[-1])
# The LQR gains themselves have no finite tilted value at kappa = -0.4.
print("LQR gains at kappa = -0.4:", tilted_value({0: 0.6, 1: 0.5}, -0.4))
assert tilted_value({0: 0.6, 1: 0.5}, -0.4) == -np.inf
assert exact_tilted_value({0: 0.6, 1: 0.5}, -0.4) == -np.inf

# %% [markdown]
# ## Rewards as Gaussian factors on the line (Section 5)
#
# Control as inference with GTSAM's ordinary Gaussian factors: each reward
# $-z^2$ becomes the factor $e^{-z^2 / \eta}$. Eliminating backward leaves a
# conditional on each action given its state; its coefficient is a gain.


# %%
def inference_gains(eta):
    graph = GaussianFactorGraph()
    graph.add(JacobianFactor(X(0), I, np.array([mu_0]),
                             noiseModel.Isotropic.Variance(1, Sigma_0)))
    for t in range(2):
        graph.add(JacobianFactor(X(t + 1), I, X(t), -I, U(t), -I, zero,
                                 noiseModel.Isotropic.Variance(1, Sigma_w)))
        for key in [X(t), U(t)]:  # the factor exp(-z^2 / eta): error z^2 / eta
            graph.add(HessianFactor(key, 2 / eta * I, zero, 0.0))
    graph.add(HessianFactor(X(2), 2 / eta * I, zero, 0.0))
    bayes_net = graph.eliminateSequential(line_backward)
    gains = {}
    for t, position in [(1, 1), (0, 3)]:
        conditional = bayes_net.at(position)  # R u + S x = d
        gains[t] = (conditional.S() / conditional.R()).item()
    return gains


for eta in [1.0, 0.5, 2.0]:
    gains = inference_gains(eta)
    K, _ = leqg(1 / eta)
    print(f"eta = {eta}: gains from Gaussian inference {gains[0]:.4f}, "
          f"{gains[1]:.4f}; risk-seeking LEQG with kappa = 1/eta: "
          f"{K[0]:.4f}, {K[1]:.4f}; plain return {plain_return(gains):.4f}")
    assert np.isclose(gains[0], K[0]) and np.isclose(gains[1], K[1])
assert np.allclose(list(inference_gains(1.0).values()), [1 / 3, 4 / 11])
assert plain_return(inference_gains(1.0)) < -9.25

# %% [markdown]
# With the dynamics kept fixed, the soft maximum over a continuous action
# keeps the Riccati gain and adds Gaussian noise of covariance
# $\tfrac{\eta}{2} H_{uu}^{-1}$. The module gives both pieces at the last
# move: eliminating $u_1$ by maximum leaves the gain in the conditional and the
# regret $-H_{uu}\,(u + K x)^2$ in its value channel, and the soft policy is
# proportional to $e^{\text{regret} / \eta}$.

# %%
eta, x = 1.0, 1.3
last_action = line().eliminateSequential(line_backward, line_rules(0.0)).at(1)
K_1 = last_action.conditional().S()[0, 0]
regret = last_action.surprise()  # a quadratic on (u1, x1): 0.5 z' G z
G_uu = regret.information()[0, 0]  # -2 H_uu, with u1 the first key
H_uu = -G_uu / 2
print(f"gain K_1 = {K_1}, H_uu = {H_uu}")
assert list(regret.keys())[0] == U(1)
assert np.isclose(K_1, 0.5) and np.isclose(H_uu, 2.0)

# An independent check by numerical integration of exp(Q / eta), where
# Q_1(x, u) = -(x^2 + u^2 + (x + u)^2 + 0.5).
u = np.linspace(-12, 12, 240001)
Q = -(x ** 2 + u ** 2 + (x + u) ** 2 + 0.5)
weights = np.exp(Q / eta)
weights /= weights.sum()
mean = weights @ u
var = weights @ (u - mean) ** 2
print(f"soft policy at x = {x}: mean {mean:.4f} (= -K_1 x), variance "
      f"{var:.4f} (= eta / (2 H_uu) = {eta / (2 * H_uu)})")
assert np.isclose(mean, -K_1 * x)
assert np.isclose(var, eta / (2 * H_uu), atol=1e-6)
