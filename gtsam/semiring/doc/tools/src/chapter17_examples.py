# %% [markdown]
# # Chapter 17 examples: soft and EM methods
#
# This notebook runs the examples of
# [Chapter 17](https://thduynguyen.github.io/gtsam/chapter17) on the endless
# track of Chapter 3: the tilted conditional as an improved policy, the
# E-step and M-step loop of MPO, REPS and AWR, and the soft fixed point that
# SAC computes. Every exact quantity is computed by elimination with the
# `gtsam/semiring` module: the soft maximum is the rule `SoftMaximum` for
# summing out the action, and the tilted policy is the conditional it leaves.
# The sampling and the fitting of a parametric policy are plain numpy.

# %%
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import SemiringDiscreteFactor, SemiringSum
from gtsam.symbol_shorthand import A, S

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track and exact policy evaluation (Section 1)

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
coin_flip = np.full((3, 2), 0.5)

# %% [markdown]
# One step of the endless chain, as semiring factors. The discount is a
# termination outcome (Chapter 3): a fourth state, "ended", reached with
# probability $1 - \gamma$ after every move.

# %%
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


ENDED = 3
ended_dynamics = np.zeros((4, 2, 4))
ended_dynamics[:3, :, :3] = gamma * dynamics  # continue
ended_dynamics[:3, :, ENDED] = 1 - gamma  # end
ended_dynamics[ENDED, :, ENDED] = 1  # stay ended
ended_prior = np.append(prior, 0.0)

now, move, later = (S(0), 4), (A(0), 2), (S(1), 4)
transition = probability([now, move, later], ended_dynamics)
move_reward = value([now, move], np.vstack([reward, [0.0, 0.0]]))
average = SemiringSum.Average()


def policy_factor(policy):
    """A policy table on the three cells, as a factor on (s, a)."""
    return probability([now, move], np.vstack([policy, [0.5, 0.5]]))


def action_values(V, extra=None):
    """The bucket of the action without a policy factor: (1, Q).

    V is the value of the next state, an array over the four states. The
    next state is eliminated by average, and the reward of the move is added,
    together with an optional extra value factor on (s, a).
    """
    bucket = move_reward * (transition * value([later], V)).sum(ordering(S(1)))
    return bucket if extra is None else extra * bucket


def read_Q(bucket):
    """The table Q(s, a) on the three cells, from the bucket of the action."""
    return table(bucket.value(), [now, move])[:3]


def fixed_point(backup):
    """Repeat one step of elimination until the value stops changing."""
    V = np.zeros(4)
    for _ in range(5000):
        new_V = backup(V)
        if np.abs(new_V - V).max() < 1e-13:
            break
        V = new_V
    return new_V


def evaluate(policy, extra=None):
    """Exact V and Q of a policy (Chapter 3): eliminate the next state and
    the action by average until the value stops changing."""
    V = fixed_point(lambda V: table(
        (policy_factor(policy) * action_values(V, extra)).sum(
            ordering(A(0)), average).value(), [now]))
    return V[:3], read_Q(action_values(V, extra))


def visitation(policy):
    """The discounted visitation d of a policy, by eliminating forward.

    One step sums out the current state and action and leaves the marginal of
    the next state. The mass still in the three cells after t moves is
    gamma^t d_t, and d is its sum over t.
    """
    marginal, d = ended_prior, np.zeros(3)
    while marginal[:3].sum() > 1e-13:
        d += marginal[:3]
        joint = probability([now], marginal) * policy_factor(policy) * transition
        marginal = table(joint.sum(ordering(S(0), A(0))).probability(), [later])
    return d


V, Q = evaluate(coin_flip)
d = visitation(coin_flip)
print("coin flip: V =", V, " J =", prior @ V)
print("Q =\n", Q)
print("d =", d)
assert np.isclose(prior @ V, 0.4385, atol=1e-4)
assert np.allclose(d, [3.8062, 3.4746, 2.7192], atol=1e-4)

# %% [markdown]
# ## Stage 1, the E-step: the tilted conditional (Section 2)
#
# Eliminate the action from the bucket $(\pi(a \mid s), Q(s, a))$ with the
# soft maximum at temperature $\eta$. The conditional, reweighted by its own
# surprise, is a new policy $q$.


# %%
def soft_eliminate(policy, Q, eta):
    """Eliminate the action from the bucket (pi, Q) by the soft maximum.

    Returns the conditional on the action and the value of the new factor,
    the soft maximum eta log sum_a pi(a|s) exp(Q(s,a) / eta).
    """
    bucket = SemiringDiscreteFactor(
        DecisionTreeFactor([now, move], np.vstack([policy, [0.5, 0.5]]).ravel()),
        DecisionTreeFactor([now, move], np.vstack([Q, [0.0, 0.0]]).ravel()))
    conditional, new_factor = bucket.eliminate(
        ordering(A(0)), SemiringSum.SoftMaximum(eta))
    return conditional, table(new_factor.value(), [now])[:3]


def soft_max(policy, Q, eta):
    """The soft maximum over the actions, for each cell."""
    return soft_eliminate(policy, Q, eta)[1]


def tilt(policy, Q, eta):
    """The tilted policy q = pi exp((Q - soft max) / eta): the conditional
    that the soft maximum leaves on the action, reweighted by its surprise."""
    conditional, _ = soft_eliminate(policy, Q, eta)
    return table(conditional.tilted(1 / eta), [now, move])[:3]


eta = 1.0
soft_V = soft_max(coin_flip, Q, eta)
q = tilt(coin_flip, Q, eta)
conditional, _ = soft_eliminate(coin_flip, Q, eta)
soft_advantage = table(conditional.surprise(), [now, move])[:3]
print("soft maximum =", soft_V, " average V =", V)
print("soft advantage =\n", soft_advantage)
print("tilted policy q =\n", q)
print("rows of q sum to", q.sum(axis=1))
assert np.allclose(q.sum(axis=1), 1)  # the normalization invariant
assert np.all(soft_V >= V)
assert np.allclose(soft_advantage, Q - soft_V[:, None])
assert np.allclose(q[:, R], [0.489, 0.894, 0.764], atol=1e-3)
# The module's soft maximum against the formula, and q against its definition.
assert np.allclose(soft_V, eta * logsumexp(Q / eta, b=coin_flip, axis=1))
assert np.allclose(q, coin_flip * np.exp(soft_advantage / eta))

# The same q from the ordinary advantage: q is proportional to pi exp(A / eta).
advantage = Q - V[:, None]
proportional = coin_flip * np.exp(advantage / eta)
assert np.allclose(proportional / proportional.sum(axis=1, keepdims=True), q)

# The tilted policy is the best trade-off between value and staying close.
rng = np.random.default_rng(0)


def kl(q, p):
    """KL(q || p) per state, with 0 log 0 = 0."""
    ratio = np.where(q > 0, q / p, 1.0)
    return (q * np.log(ratio)).sum(axis=1)


best = (q * Q).sum(axis=1) - eta * kl(q, coin_flip)
print("value minus eta * KL at q:", best)
assert np.allclose(best, soft_V)
for _ in range(1000):
    right = rng.random(3)
    other = np.stack([1 - right, right], axis=1)
    assert np.all((other * Q).sum(axis=1) - eta * kl(other, coin_flip)
                  <= soft_V + 1e-12)

# %% [markdown]
# ## The E-step and M-step loop with a free table (Sections 3 and 6)
#
# Stage 1 evaluates the current policy exactly and tilts it. With a table as
# the policy, the M-step fit is exact: the new policy is $q$.


# %%
def em(eta, iterations):
    policy = coin_flip.copy()
    history = []
    for _ in range(iterations):
        V, Q = evaluate(policy)  # stage 1: evaluate ...
        history.append(prior @ V)
        policy = tilt(policy, Q, eta)  # ... and tilt; stage 2: policy = q
    return np.array(history), policy


for eta in [1.0, 0.3]:
    history, policy = em(eta, 80)
    print(f"eta = {eta}: J =", history[[0, 1, 2, 3, 5, 10, 20, 79]])
    print("   pi(R | s) =", policy[:, R])
    assert np.all(np.diff(history) >= -1e-10)  # never worse
    assert np.isclose(history[-1], 6.4902, atol=1e-4)
    if eta == 1.0:
        assert np.allclose(history[:4], [0.4385, 4.3914, 5.6221, 6.0322],
                           atol=1e-4)
    else:
        assert np.isclose(history[10], 6.4902, atol=1e-3)

# A tiny temperature is the greedy step of policy iteration (Chapter 4).
history, policy = em(1e-3, 4)
print("eta = 0.001: J =", history)
assert np.allclose(history[:3], [0.4385, 3.7805, 6.4902], atol=1e-4)

# %% [markdown]
# ## The temperature from a bound on the KL divergence (Section 4)
#
# REPS and MPO do not choose $\eta$. They choose how far the tilted policy
# may move, as a bound $D_{\max}$ on its average KL divergence from the
# current policy, and solve a one-dimensional problem for $\eta$.

# %%
V, Q = evaluate(coin_flip)
d = visitation(coin_flip)
weights = d / d.sum()  # how often each state occurs
D_max = 0.1


def dual(eta):
    return eta * D_max + weights @ soft_max(coin_flip, Q, eta)


solution = minimize_scalar(dual, bounds=(1e-3, 100), method="bounded",
                           options={"xatol": 1e-10})
eta_star = solution.x
q_star = tilt(coin_flip, Q, eta_star)
print("eta =", eta_star)
print("average KL of the tilted policy:", weights @ kl(q_star, coin_flip))
print("q(R | s) =", q_star[:, R])
assert np.isclose(weights @ kl(q_star, coin_flip), D_max, atol=1e-5)
assert np.isclose(eta_star, 1.396, atol=1e-3)
for eta in [0.5, 1.0, 2.0, 5.0]:
    print(f"eta = {eta}: average KL =",
          weights @ kl(tilt(coin_flip, Q, eta), coin_flip))

# %% [markdown]
# ## The M-step with a policy that has shared parameters (Section 3)
#
# The policy $\pi_\theta(R \mid s) = \sigma(\theta_0 + \theta_1 s)$ has two
# parameters for three cells. The M-step fits it to $q$ by weighted maximum
# likelihood, with a few Newton steps.


# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


design = np.stack([np.ones(3), np.arange(3.0)], axis=1)  # rows (1, s)


def table_of(theta):
    right = sigmoid(design @ theta)
    return np.stack([1 - right, right], axis=1)


def m_step(theta, q, weights, steps=20):
    """Maximize sum_s weights(s) sum_a q(a|s) log pi_theta(a|s)."""
    for _ in range(steps):
        right = sigmoid(design @ theta)
        gradient = design.T @ (weights * (q[:, R] - right))
        hessian = (design * (weights * right * (1 - right))[:, None]).T @ design
        theta = theta + np.linalg.solve(hessian + 1e-6 * np.eye(2), gradient)
    return theta


theta = np.zeros(2)  # the coin flip
shared = []
for k in range(30):
    policy = table_of(theta)
    V, Q = evaluate(policy)
    d = visitation(policy)
    shared.append(prior @ V)
    q = tilt(policy, Q, eta=1.0)  # E-step
    theta = m_step(theta, q, d / d.sum())  # M-step
    if k == 0:
        print("first E-step q(R | s) =", q[:, R])
        print("first M-step pi_theta(R | s) =", table_of(theta)[:, R])
shared = np.array(shared)
print("J =", shared[[0, 1, 2, 3, 5, 10, 29]])
print("theta =", theta, " pi_theta(R | s) =", table_of(theta)[:, R])
assert shared[-1] > 6.4 and np.all(np.diff(shared) > -1e-9)
assert np.allclose(shared[:3], [0.4385, 4.1051, 5.8665], atol=1e-4)

# %% [markdown]
# ## Both steps from samples (Section 4)
#
# Now the dynamics table is only sampled. Each iteration collects 5,000
# transitions with the current policy, learns $\hat Q$ from them by the
# fitted evaluation of Chapter 15, and re-estimates the policy from the
# actions actually taken, each weighted by $e^{\hat A / \eta}$: this is
# advantage-weighted regression.


# %%
def collect(rng, policy, count):
    """Transitions (s, a, r, s'); an episode ends with probability 1 - gamma."""
    data = np.zeros((count, 4), dtype=int)
    s = rng.choice(3, p=prior)
    for i in range(count):
        a = rng.choice(2, p=policy[s])
        s_next = rng.choice(3, p=dynamics[s, a])
        data[i] = s, a, 0, s_next
        s = s_next if rng.random() < gamma else rng.choice(3, p=prior)
    return data


def learn_Q(data, policy, sweeps=200):
    """Fitted evaluation with a table: the average under the policy at s'."""
    s, a, _, s_next = data.T
    Q = np.zeros((3, 2))
    for _ in range(sweeps):
        targets = reward[s, a] + gamma * (policy[s_next] * Q[s_next]).sum(axis=1)
        total, count = np.zeros((3, 2)), np.zeros((3, 2))
        np.add.at(total, (s, a), targets)
        np.add.at(count, (s, a), 1)
        Q = np.where(count > 0, total / np.maximum(count, 1), Q)
    return Q


rng = np.random.default_rng(0)
policy = coin_flip.copy()
sampled = []
for k in range(15):
    sampled.append(prior @ evaluate(policy)[0])  # exact J, for the record
    data = collect(rng, policy, 5000)
    Q_hat = learn_Q(data, policy)  # learned backward message
    A_hat = Q_hat - (policy * Q_hat).sum(axis=1, keepdims=True)
    s, a = data[:, 0], data[:, 1]
    weighted = np.zeros((3, 2))
    np.add.at(weighted, (s, a), np.exp(A_hat[s, a] / 1.0))  # weights exp(A/eta)
    policy = weighted / weighted.sum(axis=1, keepdims=True)  # weighted counts
sampled = np.array(sampled)
print("J =", sampled[[0, 1, 2, 3, 5, 10, 14]])
print("pi(R | s) =", policy[:, R])
assert sampled[-1] > 6.4
assert np.allclose(sampled[:3], [0.4385, 4.4525, 5.6238], atol=1e-4)

# %% [markdown]
# ## SAC: the soft fixed point (Section 5)
#
# Keep the reference policy fixed (the coin flip) and the temperature fixed,
# and iterate the soft maximum to its fixed point: soft value iteration.


# %%
def soft_value_iteration(eta, reference=coin_flip, extra=None):
    """Repeat one step of elimination, the next state by average and the
    action by soft maximum, until the value stops changing."""
    soft = SemiringSum.SoftMaximum(eta)

    def bucket(V):
        return policy_factor(reference) * action_values(V, extra)

    V = fixed_point(lambda V: table(
        bucket(V).sum(ordering(A(0)), soft).value(), [now]))
    conditional, _ = bucket(V).eliminate(ordering(A(0)), soft)
    policy = table(conditional.tilted(1 / eta), [now, move])[:3]
    return V[:3], read_Q(action_values(V, extra)), policy


V_star = np.array([5.4194, 7.5610, 10.0])
for eta in [5.0, 1.0, 0.5, 0.1, 0.01, 0.001]:
    V_soft, Q_soft, policy = soft_value_iteration(eta)
    J_true = prior @ evaluate(policy)[0]
    print(f"eta = {eta:5.3f}: V = {V_soft}, pi(R|s) = {policy[:, R]}, "
          f"J of that policy = {J_true:.4f}")
    assert np.allclose(policy.sum(axis=1), 1)
    gap = V_star - V_soft
    assert np.all(gap > -1e-3)
    assert np.all(gap <= eta * np.log(2) / (1 - gamma) + 1e-3)
assert np.allclose(soft_value_iteration(0.001)[0], V_star, atol=0.01)

V_soft, Q_soft, soft_policy = soft_value_iteration(1.0)
assert np.allclose(V_soft, [1.7619, 3.6034, 6.3205], atol=1e-4)
assert np.allclose(soft_policy[:, R], [0.581, 0.907, 0.722], atol=1e-3)
assert np.isclose(prior @ evaluate(soft_policy)[0], 4.360, atol=1e-3)

# %% [markdown]
# The same fixed point from the two alternating steps of SAC, done exactly:
# evaluate the soft value of the current policy, then replace the policy by
# the tilted reference.


# %%
def soft_evaluate(policy, eta, reference=coin_flip):
    """V of a policy when every step also pays -eta KL(pi || reference).

    The price is one more value factor on (s, a), -eta log(pi / reference):
    its average under the policy is -eta times the KL divergence.
    """
    ratio = np.where(policy > 0, policy / reference, 1.0)
    price = value([now, move], np.vstack([-eta * np.log(ratio), [0.0, 0.0]]))
    V = fixed_point(lambda V: table(
        (policy_factor(policy) * action_values(V, price)).sum(
            ordering(A(0)), average).value(), [now]))
    return V[:3], read_Q(action_values(V))


policy = coin_flip.copy()
for k in range(200):
    V, Q = soft_evaluate(policy, eta=1.0)  # stage 1
    policy = tilt(coin_flip, Q, eta=1.0)  # stage 2
print("soft policy iteration: V =", V, " pi(R | s) =", policy[:, R])
assert np.allclose(V, V_soft, atol=1e-6) and np.allclose(policy, soft_policy)

# SAC's own convention uses the entropy instead of the KL to the coin flip.
# The two differ by the constant eta log 2 per step: one more value factor.
bonus = value([now], [np.log(2)] * 3 + [0.0])
entropy_V, _, _ = soft_value_iteration(1.0, extra=bonus)
print("entropy convention: V =", entropy_V)
assert np.allclose(entropy_V - V_soft, np.log(2) / (1 - gamma))
