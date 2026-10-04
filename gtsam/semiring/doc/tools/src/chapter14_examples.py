# %% [markdown]
# # Chapter 14 examples: stale messages and trust regions
#
# This notebook runs the examples of
# [Chapter 14](https://thduynguyen.github.io/gtsam/chapter14): the performance
# difference lemma, checked to machine precision; the surrogate objective,
# which reuses the old forward message; and the three ways to stay close to
# the old policy, the natural gradient, TRPO and PPO.

# %%
import itertools

import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues, Ordering
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam.symbol_shorthand import U, X
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The endless track (Chapter 3) and its exact messages
#
# The exact messages come from the module: the track as semiring factors with
# the termination outcome of Chapter 3. The value $V$ of a policy is obtained
# by composing moves, $Q$ is the bucket of the action without a policy factor,
# and the forward message $d(s)$ is the expected return when each step spent
# in the cell pays 1.

# %%
L, R = 0, 1
gamma = 0.9
prior = np.array([0.5, 0.5, 0.0])  # p(s0)
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])  # r(s, a)
cumulative = dynamics.cumsum(axis=2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def policy_table(theta):
    """pi_theta(a | s) with pi_theta(R | s) = sigmoid(theta_s)."""
    right = sigmoid(np.asarray(theta, float))
    return np.stack([1 - right, right], axis=1)


ENDED = 3
now, move, later = (X(0), 4), (U(0), 2), (X(1), 4)  # (key, cardinality)


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


def with_ended(cells, fill=0.0):
    """Add the row of the state "ended" to a table over the three cells."""
    cells = np.asarray(cells, float)
    return np.concatenate([cells, np.full((1,) + cells.shape[1:], fill)])


def step_factor(discount=gamma):
    """The dynamics of one move; the episode continues with prob. `discount`."""
    ended = np.zeros((4, 2, 4))
    ended[:3, :, :3] = discount * dynamics  # continue
    ended[:3, :, ENDED] = 1 - discount  # end
    ended[ENDED, :, ENDED] = 1  # stay ended
    return probability([now, move, later], ended)


def one_move(policy, rewards=reward, discount=gamma):
    """One move under a policy, as a factor on (s, s'): the action summed out."""
    bucket = (probability([now, move], with_ended(policy, 0.5)) *
              value([now, move], with_ended(rewards)) * step_factor(discount))
    return bucket.sum(ordering(U(0)))


def relabel(factor, old_keys, new_keys):
    """The same factor on other variables."""
    return SemiringDiscreteFactor.FromChannels(
        DecisionTreeFactor(
            new_keys, table(factor.probability(), old_keys).ravel().tolist()),
        DecisionTreeFactor(
            new_keys, table(factor.weightedValue(), old_keys).ravel().tolist()))


def compose(moves, doublings):
    """Compose a factor on (X(0), X(1)) with itself, `doublings` times.

    Each doubling multiplies the factor with a copy of itself on the
    following states and sums out the state in between, which doubles the
    number of moves. Returns the factor and the key of its last state.
    """
    for i in range(1, doublings + 1):
        middle, end = (X(i), 4), (X(i + 1), 4)
        copy = relabel(moves, [now, middle], [middle, end])
        moves = (moves * copy).sum(ordering(X(i)))  # now on (X(0), X(i + 1))
    return moves, (X(doublings + 1), 4)


def evaluate(policy, rewards=reward, discount=gamma):
    """V of the endless chain under a policy: 1024 moves, by ten doublings."""
    moves, end = compose(one_move(policy, rewards, discount), 10)
    return table(moves.sum(ordering(end[0])).value(), [now])[:3]


def action_values(V, rewards=reward, discount=gamma, baseline=None):
    """Q(s, a) = r(s, a) + discount * E[V(s')] (minus a baseline on s, if any).

    It is the value of the bucket of the action, without a policy factor.
    """
    bucket = value([now, move], with_ended(rewards)) * (
        step_factor(discount) * value([later], with_ended(V))).sum(
            ordering(X(1)))
    if baseline is not None:
        bucket = value([now], with_ended(-np.asarray(baseline))) * bucket
    return table(bucket.value(), [now, move])[:3]


def visitation(policy):
    """d(s): the expected number of steps spent in each cell per episode.

    It is the expected return of a reward that pays 1 per step in the cell.
    """
    visits = np.zeros(3)
    for cell in range(3):
        pays_one = np.zeros((3, 2))
        pays_one[cell] = 1.0
        visits[cell] = prior @ evaluate(policy, pays_one)
    return visits


def expected_return(policy):
    """J of a policy table, by elimination."""
    return prior @ evaluate(policy)


def advantages(policy):
    """J, V and the advantage A = Q - V of a policy table."""
    V = evaluate(policy)
    return prior @ V, V, action_values(V) - V[:, None]


def stage1(policy):
    """Exact J, V, Q, A and discounted visitation d of a policy table."""
    V = evaluate(policy)
    Q = action_values(V)
    return prior @ V, V, Q, Q - V[:, None], visitation(policy)


old = policy_table(np.zeros(3))  # the coin flip
J_old, V_old, Q_old, A_old, d_old = stage1(old)
print("J_old =", J_old, " d_old =", d_old)
print("A_old =\n", A_old)

# %% [markdown]
# ## The performance difference lemma (Section 2)
#
# $J_{\text{new}} - J_{\text{old}} = \sum_s d_{\text{new}}(s) \sum_a
# \pi_{\text{new}}(a \mid s)\, A_{\text{old}}(s, a)$: the new forward message
# times the old advantages. It is exact, for any pair of policies.

# %%
candidates = {
    "Right everywhere": np.eye(2)[[R, R, R]],
    "greedy on A_old: L, R, R": np.eye(2)[[L, R, R]],
    "theta = (1, -2, 0.5)": policy_table([1.0, -2.0, 0.5]),
    "Left everywhere": np.eye(2)[[L, L, L]],
}
for name, new in candidates.items():
    J_new, d_new = expected_return(new), visitation(new)
    lemma = (d_new[:, None] * new * A_old).sum()
    surrogate = (d_old[:, None] * new * A_old).sum()
    print(f"{name:26s} J_new - J_old = {J_new - J_old:8.4f}   "
          f"lemma = {lemma:8.4f}   with the old forward message = "
          f"{surrogate:8.4f}")
    assert np.isclose(J_new - J_old, lemma, atol=1e-12)

rng = np.random.default_rng(0)
worst = 0.0
for _ in range(1000):
    a_policy = rng.dirichlet([1, 1], size=3)
    b_policy = rng.dirichlet([1, 1], size=3)
    J_a, _, A_a = advantages(a_policy)
    J_b, d_b = expected_return(b_policy), visitation(b_policy)
    worst = max(worst, abs(J_b - J_a - (d_b[:, None] * b_policy * A_a).sum()))
print("largest violation over 1000 random pairs of policies:", worst)
assert worst < 1e-10

# %% [markdown]
# ## The surrogate: accurate only near the old policy (Section 3)
#
# The surrogate keeps the old forward message. Its maximum over all policies
# is the greedy policy on the old advantages: Left in cell 0, Right in cells 1
# and 2. Move the parameters from the coin flip toward that policy, along
# $\theta = \text{step} \cdot (-1, 1, 1)$, and compare the surrogate with the
# true return.

# %%
slope = 0.25  # sigma'(0)
gradient = d_old * slope * (A_old[:, R] - A_old[:, L])
fisher = np.diag(d_old * slope)
natural = np.linalg.solve(fisher, gradient)
print("gradient =", gradient, " natural gradient =", natural)


def kl(theta_from, theta_to):
    """KL between trajectory distributions: sum_s d(s) KL(pi(.|s) || pi'(.|s))."""
    p, q = policy_table(theta_from), policy_table(theta_to)
    d = visitation(p)
    return (d[:, None] * p * np.log(p / q)).sum()


def surrogate(theta_new):
    """J_old + sum_s d_old(s) sum_a pi_new(a|s) A_old(s, a)."""
    return J_old + (d_old[:, None] * policy_table(theta_new) * A_old).sum()


toward_greedy = np.sign(A_old[:, R] - A_old[:, L])
print("direction:", toward_greedy)
rows = {}
for step in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    theta_new = step * toward_greedy
    J_new = expected_return(policy_table(theta_new))
    rows[step] = (J_new, surrogate(theta_new), kl(np.zeros(3), theta_new),
                  0.5 * theta_new @ fisher @ theta_new)
    print(f"step {step:4.2f}: J = {J_new:.4f}  surrogate = {rows[step][1]:.4f}  "
          f"difference = {rows[step][1] - J_new:+.4f}  KL = {rows[step][2]:.4f}"
          f"  quadratic KL = {rows[step][3]:.4f}")
assert abs(rows[0.05][1] - rows[0.05][0]) < 0.01
assert abs(rows[8.0][1] - rows[8.0][0]) > 1.0

# The surrogate has the same gradient as J at the old policy.
h = 1e-6
numeric = np.array([(surrogate(h * e) - surrogate(-h * e)) / (2 * h)
                    for e in np.eye(3)])
print("gradient of the surrogate at the old policy:", numeric)
assert np.allclose(numeric, gradient)

# %% [markdown]
# ## A guaranteed, but very cautious, bound (Section 3)
#
# Schulman et al. (2015) prove $J_{\text{new}} \ge L - C \max_s
# \mathrm{KL}(\pi_{\text{old}}(\cdot \mid s) \,\|\, \pi_{\text{new}}(\cdot
# \mid s))$ with $C = 4 \gamma \max |A_{\text{old}}| / (1 - \gamma)^2$.

# %%
C = 4 * gamma * np.abs(A_old).max() / (1 - gamma) ** 2
print("C =", C)
rng = np.random.default_rng(1)
smallest_slack = np.inf
for _ in range(2000):
    new = policy_table(rng.normal(scale=rng.choice([0.01, 0.1, 1.0]), size=3))
    J_new = expected_return(new)
    L_new = J_old + (d_old[:, None] * new * A_old).sum()
    max_kl = (old * np.log(old / new)).sum(axis=1).max()
    smallest_slack = min(smallest_slack, J_new - (L_new - C * max_kl))
print("smallest slack of the bound over 2000 random policies:", smallest_slack)
assert smallest_slack >= 0

# %% [markdown]
# ## A TRPO step on the track of Chapter 1 (Sections 4 and 5)
#
# On the two-move track with the logistic policy at $\theta = 0$, Chapter 5
# found the gradient $(0.15, 1, 0.35)$ and the Fisher matrix
# $\operatorname{diag}(0.25, 0.2, 0.05)$. The TRPO step maximizes the surrogate
# inside the region $\mathrm{KL} \le D_{\max}$. For a small region it is the
# natural-gradient step with a particular step size.

# %%
track_prior = np.array([0.5, 0.5, 0.0])
move_reward = np.array([[0.0, -1.0]] * 3)
final_reward = np.array([0.0, 0.0, 10.0])


cell = lambda t: (X(t), 3)  # the keys of the two-move track
choice = lambda t: (U(t), 2)


def track(theta):
    """The factor graph of the two-move track under the logistic policy."""
    graph = SemiringFactorGraph()
    graph.push_back(probability([cell(0)], track_prior))
    for t in range(2):
        keys = [cell(t), choice(t)]
        graph.push_back(probability(keys, policy_table(theta)))
        graph.push_back(probability(keys + [cell(t + 1)], dynamics))
        graph.push_back(value(keys, move_reward))
    graph.push_back(value([cell(2)], final_reward))
    return graph


def track_messages(theta):
    """J, A_t and d_t of the two-move track, by elimination (Chapter 5)."""
    graph = track(theta)
    bayes_net = graph.eliminateSequential(
        ordering(X(2), U(1), X(1), U(0), X(0)))
    # The conditionals of a_1 and a_0 are at positions 1 and 3.
    A = {1: table(bayes_net.at(1).surprise(), [cell(1), choice(1)]),
         0: table(bayes_net.at(3).surprise(), [cell(0), choice(0)])}
    bayes_tree = graph.eliminateMultifrontal()
    d = {t: table(bayes_tree.marginalFactor(X(t)).probability(), [cell(t)])
         for t in range(2)}
    return graph.expectation(), A, d


J0, A0, d0 = track_messages(np.zeros(3))
pi0 = policy_table(np.zeros(3))
track_gradient = sum(d0[t] * 0.25 * (A0[t][:, R] - A0[t][:, L])
                     for t in range(2))
track_fisher = np.diag(0.25 * (d0[0] + d0[1]))
track_natural = np.linalg.solve(track_fisher, track_gradient)
print("gradient", track_gradient, " Fisher", np.diag(track_fisher),
      " natural gradient", track_natural)
assert np.allclose(track_gradient, [0.15, 1.0, 0.35])
assert np.allclose(track_natural, [0.6, 5.0, 7.0])


def track_surrogate(theta):
    new = policy_table(theta)
    return J0 + sum((d0[t][:, None] * new * A0[t]).sum() for t in range(2))


def track_kl(theta):
    """sum_t sum_s d_t(s) KL(pi_old(.|s) || pi_new(.|s))."""
    new = policy_table(theta)
    return ((d0[0] + d0[1])[:, None] * pi0 * np.log(pi0 / new)).sum()


# The trajectory KL, by enumerating all trajectories of the two-move track.
# Their probabilities are the probability channel of the product of all
# factors of the graph.
theta_test = np.array([0.3, -0.4, 0.8])
new = policy_table(theta_test)
trajectory_probability = table(
    track(np.zeros(3)).product().probability(),
    [cell(0), choice(0), cell(1), choice(1), cell(2)])
brute = 0.0
for s0, a0, s1, a1, s2 in itertools.product(
        range(3), range(2), range(3), range(2), range(3)):
    p = trajectory_probability[s0, a0, s1, a1, s2]
    if p > 0:
        brute += p * np.log(pi0[s0, a0] * pi0[s1, a1] /
                            (new[s0, a0] * new[s1, a1]))
print("trajectory KL by enumeration:", brute,
      " by forward messages:", track_kl(theta_test))
assert np.isclose(brute, track_kl(theta_test))

quadratic = track_gradient @ track_natural  # gradient^T Fisher^-1 gradient
print("gradient^T Fisher^-1 gradient =", quadratic)
trpo = {}
for D_max in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    formula = np.sqrt(2 * D_max / quadratic) * track_natural
    solution = minimize(
        lambda theta: -track_surrogate(theta), formula, method="SLSQP",
        constraints=[{"type": "ineq",
                      "fun": lambda theta: D_max - track_kl(theta)}],
        options={"ftol": 1e-14, "maxiter": 500})
    exact = solution.x
    relative = np.linalg.norm(exact - formula) / np.linalg.norm(formula)
    J_step = track_messages(formula)[0]
    trpo[D_max] = (formula, exact, relative, J_step)
    print(f"D_max = {D_max:6.4f}: natural-gradient step {formula}  "
          f"constrained maximum {exact}  relative difference {relative:.4f}  "
          f"J after the step {J_step:.4f}")
assert trpo[0.0001][2] < 0.02 and trpo[0.5][2] > trpo[0.0001][2]
assert np.isclose(quadratic, 7.54)

# %% [markdown]
# ## PPO: reusing one batch for several updates (Section 5)
#
# Back on the endless track, with sampled messages as in Chapter 13: each
# iteration runs 5 episodes, updates a TD(0) critic, and then updates the
# policy from that one batch in one of three ways. The expected return of
# each iterate is evaluated exactly, by elimination, for reporting only.


# %%
def episodes(theta, M, rng):
    """Run M episodes that end with probability 1 - gamma after each move."""
    right = sigmoid(theta)
    s = (rng.random(M)[:, None] > prior.cumsum()).sum(axis=1)
    batch = []
    while len(s):
        a = (rng.random(len(s)) < right[s]).astype(int)
        s_next = (rng.random(len(s))[:, None] > cumulative[s, a]).sum(axis=1)
        batch.append((s, a, reward[s, a], s_next))
        s = s_next[rng.random(len(s)) < gamma]
    return [np.concatenate(column) for column in zip(*batch)]


def train(rule, seed, M=5, iterations=60, alpha=0.2, epochs=20, clip=0.2,
          alpha_V=0.05):
    """rule: 'one step', 'unclipped' or 'clipped' reuse of each batch."""
    rng = np.random.default_rng(seed)
    theta, V_hat = np.zeros(3), np.zeros(3)
    history = []
    for iteration in range(iterations):
        history.append(expected_return(policy_table(theta)))
        # Stage 1, at theta_old: sampled forward message, learned critic.
        s, a, r, s_next = episodes(theta, M, rng)
        for i in rng.permutation(len(s)):
            V_hat[s[i]] += alpha_V * (r[i] + gamma * V_hat[s_next[i]]
                                      - V_hat[s[i]])
        advantage = r + gamma * V_hat[s_next] - V_hat[s]
        pi_old = policy_table(theta)[s, a]
        # Stage 2: one or several updates from the same messages.
        for epoch in range(1 if rule == "one step" else epochs):
            rho = policy_table(theta)[s, a] / pi_old  # importance weight
            if rule == "clipped":
                active = ~(((advantage > 0) & (rho > 1 + clip)) |
                           ((advantage < 0) & (rho < 1 - clip)))
            else:
                active = np.ones(len(s), dtype=bool)
            g = a - sigmoid(theta[s])  # score term
            theta = theta + alpha * np.bincount(
                s, weights=active * rho * advantage * g, minlength=3) / M
    history.append(expected_return(policy_table(theta)))
    return np.array(history)


summary = {}
for rule in ["one step", "unclipped", "clipped"]:
    histories = np.array([train(rule, seed) for seed in range(40)])
    final = histories[:, -1]
    summary[rule] = (histories[:, 10].mean(), histories[:, 30].mean(),
                     final.mean(), final.min(), (final < 5).mean(),
                     np.diff(histories, axis=1).min())
    print(f"{rule:10s} mean J after 10: {summary[rule][0]:.2f}  after 30: "
          f"{summary[rule][1]:.2f}  after 60: {summary[rule][2]:.2f}  "
          f"worst run: {summary[rule][3]:.2f}  runs ending below 5: "
          f"{summary[rule][4]:.0%}  largest drop in one iteration: "
          f"{summary[rule][5]:.2f}")
assert summary["clipped"][3] > 6.0
assert summary["unclipped"][3] < 1.0
assert summary["clipped"][0] > summary["one step"][0]
