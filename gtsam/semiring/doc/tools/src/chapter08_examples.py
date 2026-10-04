# %% [markdown]
# # Chapter 8 examples: softness and risk
#
# This notebook runs the examples of
# [Chapter 8](https://thduynguyen.github.io/gtsam/chapter08): the tilted sum of
# Chapter 2 applied to the actions (a soft maximum), to the states (an
# attitude toward risk), and to both (control as inference), on the track and
# on the line of Chapter 1.

# %%
import numpy as np
from gtsam import GaussianFactorGraph, HessianFactor, JacobianFactor, Ordering
from gtsam import noiseModel
from gtsam.symbol_shorthand import U, X

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


def evaluate(policies):
    """Plain expected return of a policy given as one table per move."""
    V = final_reward
    for policy in reversed(policies):
        V = (policy * (move_reward + dynamics @ V)).sum(axis=1)
    return prior @ V


# %% [markdown]
# ## Soft maximum over the actions, average over the states (Section 2)
#
# The backward pass of Chapter 4 with the maximum replaced by the soft
# maximum of Chapter 2. The conditional of each action, tilted, is the soft
# policy $q$.


# %%
def soft_control(eta):
    """Soft value at the root, and the soft policy of each move."""
    V, policies = final_reward, []
    for t in [1, 0]:
        Q = move_reward + dynamics @ V  # average over the next state
        V = eta * np.log((base * np.exp(Q / eta)).sum(axis=1))  # soft max
        policies.insert(0, base * np.exp((Q - V[:, None]) / eta))
    return prior @ V, policies


def expected_kl(policies):
    """Sum over the moves of the expected KL(q || base) along the policy."""
    d, total = prior, 0.0
    for q in policies:
        kl = (q * np.log(np.where(q > 0, q / base, 1.0))).sum(axis=1)
        total += d @ kl
        d = np.einsum("s,sa,sat->t", d, q, dynamics)
    return total


soft_value, soft_policies = soft_control(1.0)
print("eta = 1: soft value =", soft_value)
print("soft policy of the first move =\n", soft_policies[0])
print("soft policy of the last move =\n", soft_policies[1])
true_return = evaluate(soft_policies)
kl = expected_kl(soft_policies)
print("its plain return:", true_return, " its KL from the coin flip:", kl)
print("return - eta * KL:", true_return - 1.0 * kl)
assert np.isclose(true_return - 1.0 * kl, soft_value)
assert np.allclose(soft_policies[1][0], [0.7311, 0.2689], atol=1e-4)

# %% [markdown]
# ## The tilt on every variable: control as inference (Section 3)
#
# Chapter 2's tilted semiring, in stored form $(p, m)$ with
# $m = p\, e^{\kappa v}$, and its generic elimination routine. With
# $\kappa = 1 / \eta$ this is inference in the graph whose rewards have been
# turned into the factors $e^{r / \eta}$.


# %%
class Tilted:
    """Probability and tilted value (p, m), m = p exp(kappa v)."""

    def __init__(self, kappa):
        self.kappa = kappa

    def probability(self, p):
        return (p, p)

    def reward(self, r):
        return (np.ones_like(r), np.exp(self.kappa * r))

    def times(self, a, b):
        return (a[0] * b[0], a[1] * b[1])

    def plus(self, a, axis):
        return (a[0].sum(axis), a[1].sum(axis))

    def read(self, a):
        return np.log(a[1] / a[0]) / self.kappa


def expand(factor, variables):
    """Reshape the channels of a factor so its axes follow `variables`."""
    names, channels = factor
    index = [names.index(v) for v in variables if v in names]
    return tuple(np.transpose(c, index).reshape(
        [c.shape[names.index(v)] if v in names else 1 for v in variables])
        for c in channels)


def multiply(semiring, factors):
    """The product of several factors, on the union of their variables."""
    variables = tuple(dict.fromkeys(v for names, _ in factors for v in names))
    product = expand(factors[0], variables)
    for factor in factors[1:]:
        product = semiring.times(product, expand(factor, variables))
    shape = np.broadcast_shapes(*(c.shape for c in product))
    return variables, tuple(np.broadcast_to(c, shape) for c in product)


def eliminate(semiring, terms, order):
    """Variable elimination. Returns what is left when no variable remains."""
    factors = [(names, getattr(semiring, kind)(np.asarray(table, float)))
               for kind, names, table in terms]
    for variable in order:
        bucket = [f for f in factors if variable in f[0]]
        factors = [f for f in factors if variable not in f[0]]
        names, product = multiply(semiring, bucket)
        separator = tuple(v for v in names if v != variable)
        factors.append(
            (separator, semiring.plus(product, names.index(variable))))
    return multiply(semiring, factors)[1]


terms = [
    ("probability", ("s0",), prior),
    ("probability", ("s0", "a0"), base),
    ("reward", ("s0", "a0"), move_reward),
    ("probability", ("s0", "a0", "s1"), dynamics),
    ("probability", ("s1", "a1"), base),
    ("reward", ("s1", "a1"), move_reward),
    ("probability", ("s1", "a1", "s2"), dynamics),
    ("reward", ("s2",), final_reward),
]


def inference_value(eta, order=("s2", "a1", "s1", "a0", "s0")):
    semiring = Tilted(1 / eta)
    return float(semiring.read(eliminate(semiring, terms, list(order))))


print("eta = 2:  ", inference_value(2.0))
print("eta = 0.5:", inference_value(0.5))
assert np.isclose(inference_value(2.0), 5.4251, atol=1e-4)  # Chapter 2
assert np.isclose(inference_value(0.5), 7.6490, atol=1e-4)
# One sum for all variables: any elimination order gives the same result.
assert np.isclose(inference_value(2.0, ("a0", "s0", "s1", "s2", "a1")),
                  inference_value(2.0))

# %% [markdown]
# The same value by a backward pass, which also gives the policy that
# inference produces: the tilted conditional of each action.


# %%
def inference_control(eta):
    """Tilt at the states too: the claimed value and the policies."""
    V, policies = final_reward, []
    for t in [1, 0]:
        Q = move_reward + eta * np.log(dynamics @ np.exp(V / eta))  # tilted
        V = eta * np.log((base * np.exp(Q / eta)).sum(axis=1))
        policies.insert(0, base * np.exp((Q - V[:, None]) / eta))
    return eta * np.log(prior @ np.exp(V / eta)), policies


claimed, inference_policies = inference_control(2.0)
assert np.isclose(claimed, inference_value(2.0))

# %% [markdown]
# Inference tilts the dynamics as well. At the last move, in cell 2 after
# Left, the robot really slips and stays with probability 0.2. The conditional
# that inference leaves on the next state says otherwise:

# %%
eta = 2.0
true_dynamics = dynamics[2, L]
tilted_dynamics = true_dynamics * np.exp(final_reward / eta)
tilted_dynamics /= tilted_dynamics.sum()
print("true dynamics p(s' | 2, L):     ", true_dynamics)
print("as inference sees them, eta = 2:", tilted_dynamics)
assert np.allclose(tilted_dynamics, [0, 0.0262, 0.9738], atol=1e-4)

# %% [markdown]
# ## Three numbers per temperature (Section 3)
#
# For each temperature: the soft value with the true dynamics and the plain
# return of its policy; the value that inference claims and the plain return
# that its policy really collects. The best any policy can do is $J^* = 6.1$.

# %%
J_star = 6.1
table = {}
for eta in [10.0, 2.0, 1.0, 0.5, 0.2, 0.05]:
    soft_value, soft_policies = soft_control(eta)
    claimed, inference_policies = inference_control(eta)
    table[eta] = (soft_value, evaluate(soft_policies), claimed,
                  evaluate(inference_policies))
    print(f"eta = {eta:5.2f}: soft value {table[eta][0]:7.4f}, its policy "
          f"really collects {table[eta][1]:7.4f} | inference claims "
          f"{table[eta][2]:7.4f}, its policy really collects "
          f"{table[eta][3]:7.4f}")
    assert table[eta][1] <= J_star + 1e-9 and table[eta][3] <= J_star + 1e-9
    assert table[eta][0] <= table[eta][1] + 1e-9  # the KL term is a penalty
    assert table[eta][0] <= table[eta][2] + 1e-9  # a bound on the claim
# As eta -> 0 the soft policy becomes the best one. Its soft value is below
# J* by at most eta times the largest possible KL, 2 log 2.
assert np.isclose(table[0.05][1], 6.1, atol=1e-3)
assert 6.1 - 0.05 * 2 * np.log(2) - 1e-6 <= table[0.05][0] <= 6.1
assert table[0.05][2] > 8.8  # inference -> the best trajectory, 9
assert table[0.05][3] < table[0.05][1] - 1.0  # and its policy is worse
print("last move of the inference policy at eta = 0.05 =\n",
      inference_control(0.05)[1][1])

# %% [markdown]
# ## Linear in the tilted value (Section 3)
#
# With the tilt on every variable, the backup is a matrix product on
# $e^{V / \eta}$.

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
# $-x_2^2$ at the end, $x_0 \sim N(2, 1)$. The states are summed out with the
# tilted mean, the actions by maximum.

# %%
Sigma_w, mu_0, Sigma_0 = 0.5, 2.0, 1.0


def leqg(kappa):
    """Gains, and the tilted value at the root, for a tilt kappa."""
    P, beta, K = 1.0, 0.0, {}
    for t in [1, 0]:
        if 1 + 2 * kappa * Sigma_w * P <= 0:
            return None  # the tilted mean over the next state is infinite
        P_tilted = P / (1 + 2 * kappa * Sigma_w * P)
        constant = (np.log(1 + 2 * kappa * Sigma_w * P) / (2 * kappa)
                    if kappa != 0 else P * Sigma_w)
        K[t] = P_tilted / (1 + P_tilted)
        P = 1 + P_tilted - P_tilted ** 2 / (1 + P_tilted)
        beta = beta + constant
    if 1 + 2 * kappa * Sigma_0 * P <= 0:
        return None  # the tilted mean over the first state is infinite
    P_tilted = P / (1 + 2 * kappa * Sigma_0 * P)
    constant = (np.log(1 + 2 * kappa * Sigma_0 * P) / (2 * kappa)
                if kappa != 0 else P * Sigma_0)
    return K, -(P_tilted * mu_0 ** 2 + constant + beta)


def exact_tilted_value(K, kappa):
    """(1 / kappa) log E[exp(kappa R)] for u = -K_t x, by a Gaussian integral
    over z = (x0, w0, w1). Returns -inf or +inf if it does not exist."""
    # x0, x1, x2 and u0, u1 as linear functions of z, and R = -z' M z.
    x0 = np.array([1.0, 0, 0])
    x1 = (1 - K[0]) * x0 + np.array([0, 1.0, 0])
    x2 = (1 - K[1]) * x1 + np.array([0, 0, 1.0])
    M = sum(np.outer(row, row) for row in
            [x0, K[0] * x0, x1, K[1] * x1, x2])
    S = np.diag([Sigma_0, Sigma_w, Sigma_w])
    m = np.array([mu_0, 0, 0])
    if kappa == 0:
        return -(m @ M @ m + np.trace(M @ S))
    A = np.eye(3) + 2 * kappa * S @ M
    if np.linalg.eigvals(A).real.min() <= 0:
        return -np.inf if kappa < 0 else np.inf
    log_expectation = (-0.5 * np.log(np.linalg.det(A))
                       - kappa * m @ M @ np.linalg.solve(A, m))
    return log_expectation / kappa


def plain_return(K):
    return exact_tilted_value(K, 0)


for kappa in [-0.25, -0.2, -0.1, 0.0, 0.5, 1.0, 2.0]:
    K, value = leqg(kappa)
    check = exact_tilted_value(K, kappa)
    print(f"kappa = {kappa:5.2f}: K0 = {K[0]:.4f}, K1 = {K[1]:.4f}, tilted "
          f"value {value:8.4f} (integral {check:8.4f}), plain return "
          f"{plain_return(K):.4f}")
    assert np.isclose(value, check)
    # The gains maximize the tilted value: any change lowers it.
    for dK0, dK1 in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
        changed = {0: K[0] + dK0, 1: K[1] + dK1}
        assert exact_tilted_value(changed, kappa) < value

K, value = leqg(0.0)
assert np.isclose(K[0], 0.6) and np.isclose(K[1], 0.5)
assert np.isclose(value, -9.25)  # Chapter 6
K, value = leqg(1e-6)
assert np.isclose(K[0], 0.6, atol=1e-5) and np.isclose(value, -9.25, atol=1e-4)

# %% [markdown]
# The breakdown: below a critical tilt the tilted mean is infinite.

# %%
kappas = np.linspace(-0.6, 0, 6001)
critical = min(kappa for kappa in kappas if leqg(kappa) is not None)
print("the recursion exists down to kappa =", round(critical, 4))
K, value = leqg(critical)
print(f"there: K0 = {K[0]:.4f}, K1 = {K[1]:.4f}, tilted value {value:.2f}")
assert leqg(-0.3) is None and leqg(-0.25) is not None
# The LQR gains themselves have no finite tilted value at kappa = -0.4.
print("LQR gains at kappa = -0.4:", exact_tilted_value({0: 0.6, 1: 0.5}, -0.4))

# %% [markdown]
# ## Rewards as Gaussian factors on the line (Section 5)
#
# Control as inference with GTSAM's ordinary Gaussian factors: each reward
# $-z^2$ becomes the factor $e^{-z^2 / \eta}$. Eliminating backward leaves a
# conditional on each action given its state; its coefficient is a gain.

# %%
I, zero = np.eye(1), np.zeros(1)


def ordering(*keys):
    result = Ordering()
    for key in keys:
        result.push_back(key)
    return result


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
    bayes_net = graph.eliminateSequential(
        ordering(X(2), U(1), X(1), U(0), X(0)))
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
# $\tfrac{\eta}{2} H_{uu}^{-1}$. A numerical check at the last move, where
# $Q_1(x, u) = -(x^2 + u^2 + (x + u)^2 + 0.5)$ and $H_{uu} = 2$:

# %%
eta, x = 1.0, 1.3
u = np.linspace(-12, 12, 240001)
Q = -(x ** 2 + u ** 2 + (x + u) ** 2 + 0.5)
weights = np.exp(Q / eta)
weights /= weights.sum()
mean = weights @ u
var = weights @ (u - mean) ** 2
print(f"soft policy at x = {x}: mean {mean:.4f} (= -0.5 x), variance "
      f"{var:.4f} (= eta / (2 H_uu) = {eta / 4})")
assert np.isclose(mean, -0.5 * x) and np.isclose(var, eta / 4, atol=1e-6)
