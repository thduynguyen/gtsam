# %% [markdown]
# # Chapter 22 examples: partial observability
#
# This notebook runs the examples of
# [Chapter 22](https://thduynguyen.github.io/gtsam/chapter22): a robot on the
# track that cannot see its cell and only has a noisy sensor. It computes the
# belief by forward elimination, the best policy that uses only the sensor,
# and the QMDP shortcut, and it revisits the separation principle on the line.

# %%
import itertools

import numpy as np
from gtsam import DecisionTreeFactor, DiscreteValues
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph
from gtsam.symbol_shorthand import S

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## The track and a noisy cell sensor (Section 1)
#
# The dynamics are those of Chapter 1. The sensor reports the true cell with
# probability 0.7 and a neighbouring cell otherwise.

# %%
L, R = 0, 1
dynamics = np.zeros((3, 2, 3))  # p(s' | s, a)
dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
sensor = np.array([[0.7, 0.3, 0.0],  # p(y | s): rows s, columns y
                   [0.15, 0.7, 0.15],
                   [0.0, 0.3, 0.7]])
final_reward = np.array([0.0, 0.0, 10.0])  # r(s2)

# The track of Chapter 1, and the "lost robot" variant of this chapter.
track = dict(prior=np.array([0.5, 0.5, 0.0]),
             move_reward=np.array([[0.0, -1.0]] * 3))
lost = dict(prior=np.array([0.5, 0.0, 0.5]),
            move_reward=np.array([[0.0, -3.0]] * 3))


# %% [markdown]
# ## Three levels of knowledge (Sections 1 and 4)
#
# `fully_observed` is the backward pass of Chapter 4. `blind` tries the four
# fixed pairs of moves. `belief_pass` is the exact solution with the sensor:
# a backward pass in which the state is replaced by the belief.


# %%
def fully_observed(prior, move_reward):
    """The pass of Chapter 4: returns Q*_0, Q*_1 and J*."""
    Q1 = move_reward + dynamics @ final_reward
    Q0 = move_reward + dynamics @ Q1.max(axis=1)
    return Q0, Q1, prior @ Q0.max(axis=1)


def blind(prior, move_reward):
    """The best pair of moves chosen without any observation."""
    best = None
    for a0, a1 in itertools.product([L, R], repeat=2):
        last = move_reward[:, a1] + dynamics[:, a1, :] @ final_reward
        J = prior @ (move_reward[:, a0] + dynamics[:, a0, :] @ last)
        if best is None or J > best[0]:
            best = (J, (a0, a1))
    return best


def update(predicted, y):
    """One step of the filter: multiply by the sensor factor, normalize.

    Returns the probability of the reading and the belief after it.
    """
    joint = predicted * sensor[:, y]
    total = joint.sum()
    return total, joint / total if total > 0 else joint


def belief_pass(prior, move_reward, first_move="best"):
    """Expected return of a policy that sees only the sensor readings.

    The last move is always the best one for the belief. The first move is
    the best one ("best") or the one QMDP picks ("qmdp").
    """
    Q0, Q1, _ = fully_observed(prior, move_reward)
    J, moves = 0.0, {}
    for y0 in range(3):
        p_y0, belief0 = update(prior, y0)
        if p_y0 == 0:
            continue
        value = []
        for a0 in [L, R]:
            total = belief0 @ move_reward[:, a0]
            predicted = belief0 @ dynamics[:, a0, :]
            for y1 in range(3):
                p_y1, belief1 = update(predicted, y1)
                if p_y1 > 0:
                    total += p_y1 * (belief1 @ Q1).max()
            value.append(total)
        qmdp_value = belief0 @ Q0  # pretends the cell is revealed afterwards
        a0 = int(np.argmax(value if first_move == "best" else qmdp_value))
        moves[y0] = dict(belief=belief0, value=np.array(value),
                         qmdp=qmdp_value, move=a0)
        J += p_y0 * value[a0]
    return J, moves


for name, problem in [("track", track), ("lost robot", lost)]:
    _, _, J_full = fully_observed(**problem)
    J_blind, moves_blind = blind(**problem)
    J_sensor, _ = belief_pass(**problem)
    J_qmdp, _ = belief_pass(**problem, first_move="qmdp")
    print(f"{name:10s}: blind {J_blind:.4f} {moves_blind}, QMDP {J_qmdp:.4f}, "
          f"sensor {J_sensor:.4f}, fully observed {J_full:.4f}")

assert np.isclose(fully_observed(**track)[2], 6.1)
assert np.isclose(blind(**track)[0], 6.0)
assert np.isclose(belief_pass(**track)[0], 6.0)
assert np.isclose(fully_observed(**lost)[2], 3.2)
assert np.isclose(blind(**lost)[0], 2.2)
assert np.isclose(belief_pass(**lost)[0], 2.69)
assert np.isclose(belief_pass(**lost, first_move="qmdp")[0], 2.615)

# %% [markdown]
# ## The belief: a forward message given the readings (Section 2)
#
# For the lost robot, the belief after the first reading, the predicted belief
# after a move Right, and the belief after the second reading.

# %%
prior = lost["prior"]
for y0 in range(3):
    p_y0, belief0 = update(prior, y0)
    print(f"y0 = {y0}: probability {p_y0:.3f}, belief {belief0}")

p_y0, belief0 = update(prior, 1)  # the first reading says "cell 1"
predicted = belief0 @ dynamics[:, R, :]  # then the robot moves Right
print("after y0 = 1:", belief0)
print("predicted after moving Right:", predicted)
for y1 in range(3):
    p_y1, belief1 = update(predicted, y1)
    print(f"  y1 = {y1}: probability {p_y1:.4f}, belief {belief1}")
assert np.allclose(belief0, [0.5, 0, 0.5])
assert np.allclose(predicted, [0.1, 0.4, 0.5])

# %% [markdown]
# The same belief with the module. The readings are fixed, so each sensor
# factor is a table on its state alone, and the action is fixed, so the
# dynamics factor is a table on two states. The marginal of $s_1$ is the
# belief.

# %%
state = lambda t: (S(t), 3)


def probability(keys, table):
    """Lift a probability table to (p, 0)."""
    return SemiringDiscreteFactor(
        DecisionTreeFactor(keys, np.ravel(table).tolist()))


def table(factor, keys):
    """Read a DecisionTreeFactor into an array indexed in the order of keys."""
    result = np.zeros([cardinality for _, cardinality in keys])
    for index in np.ndindex(*result.shape):
        values = DiscreteValues()
        for (key, _), index_of_key in zip(keys, index):
            values[key] = index_of_key
        result[index] = factor(values)
    return result


def module_belief(prior, y0, a0, y1):
    graph = SemiringFactorGraph()
    graph.push_back(probability([state(0)], prior))
    graph.push_back(probability([state(0)], sensor[:, y0]))
    graph.push_back(probability([state(0), state(1)], dynamics[:, a0, :]))
    graph.push_back(probability([state(1)], sensor[:, y1]))
    marginal = graph.eliminateMultifrontal().marginalFactor(S(1))
    belief = table(marginal.probability(), [state(1)])
    return belief / belief.sum()


for y1 in range(3):
    by_module = module_belief(prior, 1, R, y1)
    _, by_numpy = update(predicted, y1)
    print(f"y1 = {y1}: module {by_module}, numpy {by_numpy}")
    assert np.allclose(by_module, by_numpy)

# %% [markdown]
# ## The best policy that sees only the sensor (Section 3)
#
# The first move for each first reading, with its value, and the last move as
# a function of the belief.

# %%
J_sensor, moves = belief_pass(**lost)
for y0, entry in moves.items():
    print(f"y0 = {y0}: belief {entry['belief']}, value of L, R = "
          f"{entry['value']}, best first move {'LR'[entry['move']]}")

Q0, Q1, _ = fully_observed(**lost)
print("Q*_1 =\n", Q1)
print("Q*_0 =\n", Q0)
# Right is the better last move when the belief b satisfies b @ Q1[:, R] >
# b @ Q1[:, L].
print("difference Q*_1(s, R) - Q*_1(s, L):", Q1[:, R] - Q1[:, L])

# %% [markdown]
# A check by brute force: all deterministic policies that map the readings to
# moves. The first move depends on $y_0$ (8 choices), the last move on
# $(y_0, y_1)$ (512 choices).


# %%
def evaluate(prior, move_reward, first, last):
    """Expected return of a policy given as tables first[y0], last[y0][y1]."""
    J = 0.0
    for s0, y0 in itertools.product(range(3), repeat=2):
        p0 = prior[s0] * sensor[s0, y0]
        if p0 == 0:
            continue
        a0 = first[y0]
        for s1, y1 in itertools.product(range(3), repeat=2):
            p1 = p0 * dynamics[s0, a0, s1] * sensor[s1, y1]
            if p1 == 0:
                continue
            a1 = last[y0][y1]
            J += p1 * (move_reward[s0, a0] + move_reward[s1, a1] +
                       dynamics[s1, a1] @ final_reward)
    return J


best = -np.inf
for first in itertools.product([L, R], repeat=3):
    for flat in itertools.product([L, R], repeat=9):
        last = np.reshape(flat, (3, 3))
        best = max(best, evaluate(**lost, first=first, last=last))
print("best of 4096 policies:", best)
assert np.isclose(best, J_sensor)

# %% [markdown]
# ## QMDP and where it errs (Section 5)
#
# QMDP weights the fully observed action values by the belief. That assumes
# the cell will be known exactly after this move.

# %%
J_qmdp, moves_qmdp = belief_pass(**lost, first_move="qmdp")
for y0, entry in moves_qmdp.items():
    print(f"y0 = {y0}: QMDP values of L, R = {entry['qmdp']}, true values = "
          f"{entry['value']}, QMDP move {'LR'[entry['move']]}, "
          f"best move {'LR'[moves[y0]['move']]}")
print("QMDP:", J_qmdp, " best:", J_sensor)
assert J_qmdp < J_sensor

# %% [markdown]
# ## The value of the sensor on the original track (Section 1)
#
# On the track of Chapter 1 the sensor changes nothing: no reading makes the
# belief in cell 0 large enough to prefer Left at the last move.

# %%
Q0_track, Q1_track, _ = fully_observed(**track)
gap = Q1_track[:, R] - Q1_track[:, L]
print("Q*_1(s, R) - Q*_1(s, L) on the track:", gap)
# Left is better only if b0 * 1 > b1 * 7 + b2 * 7, i.e. b0 > 7/8.
largest = 0.0
for y0, a0, y1 in itertools.product(range(3), [L, R], range(3)):
    p_y0, belief0 = update(track["prior"], y0)
    if p_y0 == 0:
        continue
    p_y1, belief1 = update(belief0 @ dynamics[:, a0, :], y1)
    if p_y1 > 0:
        largest = max(largest, belief1[0]) if a0 == R else largest
print("largest belief in cell 0 after a move Right:", largest)
assert largest < 7 / 8

# %% [markdown]
# ## The separation principle on the line (Section 6)
#
# The line of Chapter 1 with a noisy position sensor,
# $y_t = x_t + n_t$, $n_t \sim N(0, 0.5)$. The best policy applies the Riccati
# gains of the fully observed problem to the mean of the belief, which is the
# Kalman filter. The expected return is the fully observed one minus a price
# for the estimation error.

# %%
sigma_w, sigma_y = 0.5, 0.5
gains = [0.6, 0.5]  # the Riccati gains of the fully observed problem
H_uu = [1 + 1.5, 1 + 1.0]  # C_u + B' P_{t+1} B
J_full = -9.25

# Variance of the belief after each reading (the Kalman filter).
variance = []
predicted_variance = 1.0  # the prior variance of x0
for t in range(2):
    posterior = predicted_variance * sigma_y / (predicted_variance + sigma_y)
    variance.append(posterior)
    predicted_variance = posterior + sigma_w
print("variance of the belief after y0 and y1:", variance)
price = sum(H_uu[t] * gains[t] ** 2 * variance[t] for t in range(2))
J_lqg = J_full - price
print("price of not seeing the state:", price, " J =", J_lqg)
assert np.isclose(J_lqg, -9.70625)

# A check by simulation, with a fixed seed.
rng = np.random.default_rng(0)
samples = 400_000
x = 2 + rng.normal(size=samples)
mean, var, total = np.full(samples, 2.0), 1.0, np.zeros(samples)
for t in range(2):
    y = x + np.sqrt(sigma_y) * rng.normal(size=samples)
    kalman = var / (var + sigma_y)
    mean, var = mean + kalman * (y - mean), (1 - kalman) * var
    u = -gains[t] * mean
    total -= x ** 2 + u ** 2
    x = x + u + np.sqrt(sigma_w) * rng.normal(size=samples)
    mean, var = mean + u, var + sigma_w
total -= x ** 2
print("simulated J:", total.mean(), "+-", total.std() / np.sqrt(samples))
assert abs(total.mean() - J_lqg) < 0.05
