"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Unit tests for semiring factors, conditionals and factor graphs, which compute
expected rewards by variable elimination.
Author: Duy-Nguyen Ta
"""

# pylint: disable=no-name-in-module, invalid-name

import unittest

import numpy as np
from gtsam.symbol_shorthand import U, X
from gtsam.utils.test_case import GtsamTestCase

import gtsam
from gtsam import (
    DecisionTreeFactor,
    DiscreteConditional,
    DiscreteValues,
    GaussianConditional,
    HessianFactor,
    JacobianFactor,
    Ordering,
    SemiringDiscreteConditional,
    SemiringDiscreteFactor,
    SemiringFactorGraph,
    SemiringGaussianConditional,
    SemiringGaussianFactor,
    VectorValues,
    noiseModel,
)


def make_ordering(*keys):
    """Create an ordering from keys."""
    ordering = Ordering()
    for key in keys:
        ordering.push_back(key)
    return ordering


def assignment(**pairs):
    """Create discrete values from (discrete key, value) pairs."""
    values = DiscreteValues()
    for (key, _), value in pairs.values():
        values[key] = value
    return values


def table(factor, keys):
    """Read a DecisionTreeFactor into an array indexed in the order of keys."""
    result = np.zeros([cardinality for _, cardinality in keys])
    for index in np.ndindex(*result.shape):
        values = DiscreteValues()
        for (key, _), value in zip(keys, index):
            values[key] = value
        result[index] = factor(values)
    return result


class TestSemiringTwoActions(GtsamTestCase):
    """One decision: action L or R, then a good or bad next state.

    Action L costs 1 and the good state pays 10, so Q(L) = -1 + 0.9 * 10 = 8
    and Q(R) = 0.2 * 10 = 2. Under the policy (0.6, 0.4), V = 5.6.
    """

    def setUp(self):
        self.A = (0, 2)
        self.S = (1, 2)
        self.policy = SemiringDiscreteFactor(
            DecisionTreeFactor(self.A, "0.6 0.4"))
        self.actionReward = SemiringDiscreteFactor.Reward(
            DecisionTreeFactor(self.A, "-1 0"))
        self.transition = SemiringDiscreteFactor(
            DecisionTreeFactor([self.A, self.S], "0.9 0.1 0.2 0.8"))
        self.stateReward = SemiringDiscreteFactor.Reward(
            DecisionTreeFactor(self.S, "10 0"))

    def values(self, action, state):
        """Assignment to the action and the next state."""
        return assignment(a=(self.A, action), s=(self.S, state))

    def graph(self):
        """The factor graph of the whole problem."""
        graph = SemiringFactorGraph()
        for factor in [self.policy, self.actionReward, self.transition,
                       self.stateReward]:
            graph.push_back(factor)
        return graph

    def test_channels(self):
        """Probability factors lift to (f, 0), rewards to (1, r)."""
        self.assertEqual(self.transition.evaluate(self.values(0, 1)),
                         (0.1, 0.0))
        self.assertEqual(self.stateReward.evaluate(self.values(1, 0)),
                         (1.0, 10.0))
        self.assertEqual(self.transition.keys(), [0, 1])
        self.assertIsInstance(self.transition.probability(),
                              DecisionTreeFactor)

    def test_accepts_conditional(self):
        """A DiscreteConditional can be lifted directly."""
        conditional = DiscreteConditional(self.S, [self.A], "9/1 2/8")
        factor = SemiringDiscreteFactor(conditional)
        self.assertTrue(factor.equals(self.transition, 1e-9))

    def test_product(self):
        """The product multiplies probabilities and adds values."""
        product = (self.policy * self.actionReward * self.transition *
                   self.stateReward)
        probability, value = product.evaluate(self.values(0, 0))
        self.assertAlmostEqual(probability, 0.54)
        self.assertAlmostEqual(value, 9.0)
        self.assertAlmostEqual(product.expectation(), 5.6)

        # The virtual product returns the derived type.
        virtual = self.transition.multiply(self.stateReward)
        self.assertIsInstance(virtual, SemiringDiscreteFactor)
        self.assertTrue(
            virtual.equals(self.transition * self.stateReward, 1e-9))

    def test_eliminate_factor(self):
        """Eliminating the next state gives its conditional and Q(a) - r(a)."""
        joint = self.transition * self.stateReward
        conditional, separator = joint.eliminate(make_ordering(self.S[0]))

        self.assertIsInstance(separator, SemiringDiscreteFactor)
        np.testing.assert_allclose(table(separator.value(), [self.A]), [9, 2])
        np.testing.assert_allclose(
            table(separator.probability(), [self.A]), [1, 1])

        self.assertIsInstance(conditional, SemiringDiscreteConditional)
        self.assertEqual(conditional.keys(), [1, 0])
        self.assertEqual(conditional.nrFrontals(), 1)
        self.assertEqual(conditional.firstFrontalKey(), 1)
        self.assertIsInstance(conditional.probability(), DiscreteConditional)
        np.testing.assert_allclose(
            table(conditional.surprise(), [self.A, self.S]),
            [[1, -9], [8, -2]])

        # The sum alone is the same separator factor.
        self.assertTrue(
            joint.sum(make_ordering(self.S[0])).equals(separator, 1e-9))

    def test_expectation(self):
        """The expected total reward does not depend on the ordering."""
        graph = self.graph()
        self.assertEqual(graph.size(), 4)
        self.assertAlmostEqual(graph.expectation(), 5.6)
        self.assertAlmostEqual(
            graph.expectation(make_ordering(self.S[0], self.A[0])), 5.6)
        self.assertAlmostEqual(
            graph.expectation(make_ordering(self.A[0], self.S[0])), 5.6)
        self.assertAlmostEqual(graph.product().expectation(), 5.6)

    def test_eliminate_sequential(self):
        """The Bayes net holds the policy with the advantage function."""
        bayesNet = self.graph().eliminateSequential(
            make_ordering(self.S[0], self.A[0]))
        self.assertEqual(bayesNet.size(), 2)

        # Retrieve repeatedly: both wrappers must refer to the same result.
        for _ in range(2):
            action = bayesNet.at(1)
            self.assertIsInstance(action, SemiringDiscreteConditional)
            self.assertEqual(action.keys(), [0])
            np.testing.assert_allclose(
                table(action.surprise(), [self.A]), [2.4, -3.6])

        # p(a) A(a) is the policy gradient for softmax logits.
        gradient = bayesNet.at(1).table().weightedValue()
        np.testing.assert_allclose(table(gradient, [self.A]), [1.44, -1.44])

    def test_eliminate_partial(self):
        """Partial elimination leaves the action values in the graph."""
        graph = SemiringFactorGraph()
        for factor in [self.actionReward, self.transition, self.stateReward]:
            graph.push_back(factor)
        bayesNet, remaining = graph.eliminatePartialSequential(
            make_ordering(self.S[0]))
        self.assertEqual(bayesNet.size(), 1)
        actionValue = remaining.product()
        np.testing.assert_allclose(table(actionValue.value(), [self.A]), [8, 2])

    def test_custom_eliminate(self):
        """A Python function can serve as the elimination function."""
        calls = []

        def eliminate(factors, keys):
            calls.append(factors.size())
            return gtsam.EliminateSemiring(factors, keys)

        ordering = make_ordering(self.S[0], self.A[0])
        bayesNet = self.graph().eliminateSequential(ordering, eliminate)
        self.assertEqual(calls, [2, 3])
        self.assertTrue(
            bayesNet.equals(self.graph().eliminateSequential(ordering), 1e-9))

    def test_multifrontal(self):
        """Multifrontal elimination and marginals from the Bayes tree."""
        bayesTree = self.graph().eliminateMultifrontal(
            make_ordering(self.S[0], self.A[0]))
        self.assertEqual(bayesTree.size(), 1)
        joint = bayesTree[self.A[0]].conditional()
        self.assertIsInstance(joint, SemiringDiscreteConditional)
        self.assertEqual(joint.nrFrontals(), 2)

        marginal = bayesTree.marginalFactor(self.A[0])
        probability, value = marginal.evaluate(self.values(0, 0))
        self.assertAlmostEqual(probability, 0.6)
        self.assertAlmostEqual(value, 8.0 - 5.6)

    def test_bayes_net_as_graph(self):
        """Conditionals are factors; their surprises average to zero."""
        bayesNet = self.graph().eliminateSequential()
        graph = SemiringFactorGraph(bayesNet)
        self.assertEqual(graph.size(), 2)
        self.assertAlmostEqual(graph.expectation(), 0.0)


class TestSemiringTrackExample(GtsamTestCase):
    """The discrete worked example of gtsam/semiring/doc/chapter01.md.

    A robot on a track of three cells makes two moves, Left or Right, each
    with probability 0.5. A move succeeds with probability 0.8, else the robot
    stays. Moving right costs 1, and ending in cell 2 pays 10.
    """

    def setUp(self):
        self.prior = [0.5, 0.5, 0.0]
        self.policy = [[0.5, 0.5]] * 3
        self.dynamics = [[[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]],
                         [[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]],
                         [[0.0, 0.8, 0.2], [0.0, 0.0, 1.0]]]
        self.moveReward = [[0.0, -1.0]] * 3
        self.finalReward = [0.0, 0.0, 10.0]

    @staticmethod
    def state(t):
        """Discrete key of the cell after t moves."""
        return (X(t), 3)

    @staticmethod
    def action(t):
        """Discrete key of move t."""
        return (U(t), 2)

    @staticmethod
    def probability(keys, values):
        """Lift a probability table to (p, 0)."""
        return SemiringDiscreteFactor(
            DecisionTreeFactor(keys, np.ravel(values).tolist()))

    @staticmethod
    def reward(keys, values):
        """Lift a reward table to (1, r)."""
        return SemiringDiscreteFactor.Reward(
            DecisionTreeFactor(keys, np.ravel(values).tolist()))

    def graph(self):
        """The eight factors of the example."""
        graph = SemiringFactorGraph()
        graph.push_back(self.probability([self.state(0)], self.prior))
        for t in range(2):
            keys = [self.state(t), self.action(t)]
            graph.push_back(self.probability(keys, self.policy))
            graph.push_back(
                self.probability(keys + [self.state(t + 1)], self.dynamics))
            graph.push_back(self.reward(keys, self.moveReward))
        graph.push_back(self.reward([self.state(2)], self.finalReward))
        return graph

    def test_expected_return(self):
        """The coin-flipping robot collects 1.4 on average."""
        self.assertEqual(self.graph().size(), 8)
        self.assertAlmostEqual(self.graph().expectation(), 1.4)

    def test_elimination_steps(self):
        """Every table of the chapter, from eliminating backward in time."""
        bayesNet = self.graph().eliminateSequential(
            make_ordering(X(2), U(1), X(1), U(0), X(0)))
        lastState, lastAction, middleState, firstAction, firstState = [
            bayesNet.at(i) for i in range(5)
        ]

        # Step 1: surprises of the last move's outcomes, cell 1 and Right.
        keys = [self.state(1), self.action(1), self.state(2)]
        np.testing.assert_allclose(
            table(lastState.surprise(), keys)[1, 1], [0, -8, 2])

        # Step 2: A1 = Q1 - V1.
        keys = [self.state(1), self.action(1)]
        np.testing.assert_allclose(
            table(lastAction.surprise(), keys),
            [[0.5, -0.5], [-3.5, 3.5], [-3.5, 3.5]])

        # Step 3: surprises of the first move's outcomes, cell 1 and Right.
        keys = [self.state(0), self.action(0), self.state(1)]
        np.testing.assert_allclose(
            table(middleState.surprise(), keys)[1, 1], [0, -1.6, 0.4])

        # Step 4: A0 = Q0 - V0.
        keys = [self.state(0), self.action(0)]
        np.testing.assert_allclose(
            table(firstAction.surprise(), keys),
            [[-1.1, 1.1], [-1.9, 1.9], [-0.3, 0.3]])
        np.testing.assert_allclose(
            table(firstAction.probability(), keys), self.policy)

        # Step 5: V0 - J for the two possible starting cells.
        np.testing.assert_allclose(
            table(firstState.surprise(), [self.state(0)])[:2], [-0.8, 0.8])

    def test_value_functions(self):
        """Q and V at both moves, by eliminating one bucket at a time."""
        expectedQ = [[[-0.5, 1.7], [0.3, 4.1], [3.9, 4.5]],
                     [[0, -1], [0, 7], [2, 9]]]
        expectedV = [[0.6, 2.2, 4.2], [-0.5, 3.5, 5.5]]
        value = self.reward([self.state(2)], self.finalReward)
        for t in reversed(range(2)):
            keys = [self.state(t), self.action(t)]
            dynamics = self.probability(keys + [self.state(t + 1)],
                                        self.dynamics)
            bucket = (self.probability(keys, self.policy) *
                      self.reward(keys, self.moveReward) *
                      (dynamics * value).sum(make_ordering(X(t + 1))))
            np.testing.assert_allclose(table(bucket.value(), keys),
                                       expectedQ[t])
            value = bucket.sum(make_ordering(U(t)))
            np.testing.assert_allclose(
                table(value.value(), [self.state(t)]), expectedV[t])

    def test_best_policy(self):
        """Maximizing over the action is dynamic programming (chapter 2)."""
        expectedQ = [[[0, 4.6], [1.4, 7.6], [7.4, 8]],
                     [[0, -1], [0, 7], [2, 9]]]
        expectedV = [[4.6, 7.6, 8], [0, 7, 9]]
        expectedMove = [[1, 1, 1], [0, 1, 1]]  # 0 = Left, 1 = Right
        value = self.reward([self.state(2)], self.finalReward)
        for t in reversed(range(2)):
            keys = [self.state(t), self.action(t)]
            dynamics = self.probability(keys + [self.state(t + 1)],
                                        self.dynamics)
            # Eliminate the next state by expectation; no policy factor.
            bucket = self.reward(keys, self.moveReward) * (
                dynamics * value).sum(make_ordering(X(t + 1)))
            Q = table(bucket.value(), keys)
            np.testing.assert_allclose(Q, expectedQ[t])
            np.testing.assert_array_equal(Q.argmax(axis=1), expectedMove[t])
            # Eliminate the action by max: the best move, as a factor.
            best = self.probability(keys, np.eye(2)[Q.argmax(axis=1)])
            value = (best * bucket).sum(make_ordering(U(t)))
            np.testing.assert_allclose(
                table(value.value(), [self.state(t)]), expectedV[t])
        prior = self.probability([self.state(0)], self.prior)
        self.assertAlmostEqual((prior * value).expectation(), 6.1)

    def test_trajectory_surprises(self):
        """Surprises along one trajectory add up to its return minus J."""
        # Start in cell 1, move Right to cell 2, move Right and stay.
        values = DiscreteValues()
        for key, value in [(X(0), 1), (U(0), 1), (X(1), 2), (U(1), 1),
                           (X(2), 2)]:
            values[key] = value
        bayesNet = self.graph().eliminateSequential(
            make_ordering(X(2), U(1), X(1), U(0), X(0)))
        surprises = [bayesNet.at(i).surprise()(values) for i in range(5)]
        np.testing.assert_allclose(surprises, [0, 3.5, 0.4, 1.9, 0.8])
        self.assertAlmostEqual(sum(surprises), (-1 - 1 + 10) - 1.4)


class TestSemiringMarkovDecisionProcess(GtsamTestCase):
    """A small tabular MDP, checked against dynamic programming in numpy."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.nrStates, self.nrActions, self.horizon = 3, 2, 4
        # dynamics[s, a, s'], reward[s, a], terminalReward[s], prior[s]
        self.dynamics = rng.dirichlet(np.ones(self.nrStates),
                                      size=(self.nrStates, self.nrActions))
        self.reward = rng.normal(size=(self.nrStates, self.nrActions))
        self.terminalReward = rng.normal(size=self.nrStates)
        self.prior = rng.dirichlet(np.ones(self.nrStates))
        # policy[s, a], the same at every step
        self.policy = rng.dirichlet(np.ones(self.nrActions),
                                    size=self.nrStates)

    def state(self, t):
        """Discrete key of the state at step t, a 64-bit symbol key."""
        return (X(t), self.nrStates)

    def action(self, t):
        """Discrete key of the action at step t."""
        return (U(t), self.nrActions)

    def stage_factors(self, t, policy):
        """The policy, dynamics and reward factors of step t."""
        keys = [self.state(t), self.action(t)]
        return (SemiringDiscreteFactor(
            DecisionTreeFactor(keys, policy.flatten().tolist())),
                SemiringDiscreteFactor(
                    DecisionTreeFactor(keys + [self.state(t + 1)],
                                       self.dynamics.flatten().tolist())),
                SemiringDiscreteFactor.Reward(
                    DecisionTreeFactor(keys, self.reward.flatten().tolist())))

    def terminal_factor(self):
        """The reward factor on the final state."""
        return SemiringDiscreteFactor.Reward(
            DecisionTreeFactor(self.state(self.horizon),
                               self.terminalReward.tolist()))

    def graph(self, withPrior=True):
        """The factor graph of the MDP under the fixed policy."""
        graph = SemiringFactorGraph()
        if withPrior:
            graph.push_back(
                SemiringDiscreteFactor(
                    DecisionTreeFactor(self.state(0), self.prior.tolist())))
        for t in range(self.horizon):
            for factor in self.stage_factors(t, self.policy):
                graph.push_back(factor)
        graph.push_back(self.terminal_factor())
        return graph

    def dynamic_programming(self, greedy=False):
        """Backward recursion in numpy: Q and V tables for every step."""
        V = self.terminalReward
        Qs, Vs = [], [V]
        for _ in range(self.horizon):
            Q = self.reward + self.dynamics @ V
            V = Q.max(axis=1) if greedy else (self.policy * Q).sum(axis=1)
            Qs.insert(0, Q)
            Vs.insert(0, V)
        return Qs, Vs

    def test_policy_evaluation(self):
        """Elimination evaluates the policy: expected total reward and V."""
        _, Vs = self.dynamic_programming()
        self.assertAlmostEqual(self.graph().expectation(), self.prior @ Vs[0])

        # Eliminate backward in time down to the first state.
        ordering = Ordering()
        for t in reversed(range(self.horizon)):
            ordering.push_back(X(t + 1))
            ordering.push_back(U(t))
        bayesNet, remaining = self.graph(False).eliminatePartialSequential(
            ordering)
        self.assertEqual(bayesNet.size(), 2 * self.horizon)
        valueFunction = remaining.product()
        self.assertEqual(valueFunction.keys(), [X(0)])
        np.testing.assert_allclose(
            table(valueFunction.value(), [self.state(0)]), Vs[0])

    def test_bellman_backups(self):
        """Eliminate by hand, one Bellman backup per step."""
        Qs, Vs = self.dynamic_programming()
        value = self.terminal_factor()
        for t in reversed(range(self.horizon)):
            policy, dynamics, reward = self.stage_factors(t, self.policy)
            keys = [self.state(t), self.action(t)]

            # Summing out the next state gives the action values Q.
            actionValue = (dynamics * reward * value).sum(
                make_ordering(X(t + 1)))
            np.testing.assert_allclose(table(actionValue.value(), keys), Qs[t])

            # Summing out the action gives V, and the advantage Q - V.
            conditional, value = (policy * actionValue).eliminate(
                make_ordering(U(t)))
            np.testing.assert_allclose(
                table(value.value(), [self.state(t)]), Vs[t])
            np.testing.assert_allclose(
                table(conditional.surprise(), keys), Qs[t] - Vs[t][:, None])
            np.testing.assert_allclose(
                table(conditional.probability(), keys), self.policy)

    def test_greedy_backward_induction(self):
        """Acting greedily on Q at every step yields the optimal values."""
        _, optimal = self.dynamic_programming(greedy=True)
        value = self.terminal_factor()
        for t in reversed(range(self.horizon)):
            _, dynamics, reward = self.stage_factors(t, self.policy)
            keys = [self.state(t), self.action(t)]
            actionValue = (dynamics * reward * value).sum(
                make_ordering(X(t + 1)))

            # A deterministic policy has zero-probability entries.
            Q = table(actionValue.value(), keys)
            greedy = np.eye(self.nrActions)[Q.argmax(axis=1)]
            policy = SemiringDiscreteFactor(
                DecisionTreeFactor(keys, greedy.flatten().tolist()))
            value = (policy * actionValue).sum(make_ordering(U(t)))
            np.testing.assert_allclose(
                table(value.value(), [self.state(t)]), optimal[t])

        # The greedy policy is at least as good as the fixed one.
        _, Vs = self.dynamic_programming()
        self.assertTrue(np.all(optimal[0] >= Vs[0] - 1e-12))


class TestSemiringLineExample(GtsamTestCase):
    """The continuous worked example of gtsam/semiring/doc/chapter01.md.

    A robot on a line makes two moves: x' = x + u + w with w ~ N(0, 0.5),
    under the policy u = -0.5 x + e with e ~ N(0, 0.1). It pays x^2 + u^2 at
    each move and x^2 at the end, and starts at x0 ~ N(2, 1).
    """

    @staticmethod
    def gaussian(*args):
        """Lift a Gaussian factor to (p, 0)."""
        return SemiringGaussianFactor(JacobianFactor(*args))

    @staticmethod
    def penalty(key):
        """Lift the penalty z^2 on one variable to the reward (1, -z^2)."""
        return SemiringGaussianFactor.Cost(
            HessianFactor(key, 2.0 * np.eye(1), np.zeros(1), 0.0))

    @staticmethod
    def values(**scalars):
        """VectorValues from (key, value) pairs."""
        result = VectorValues()
        for key, value in scalars.values():
            result.insert(key, np.array([value]))
        return result

    def dynamics(self, t):
        """The dynamics factor of move t."""
        I = np.eye(1)
        return self.gaussian(X(t + 1), I, X(t), -I, U(t), -I, np.zeros(1),
                             noiseModel.Isotropic.Variance(1, 0.5))

    def policy(self, t):
        """The policy factor of move t."""
        I = np.eye(1)
        return self.gaussian(U(t), I, X(t), 0.5 * I, np.zeros(1),
                             noiseModel.Isotropic.Variance(1, 0.1))

    def graph(self):
        """The factor graph of the example."""
        graph = SemiringFactorGraph()
        graph.push_back(
            self.gaussian(X(0), np.eye(1), np.array([2.0]),
                          noiseModel.Isotropic.Variance(1, 1.0)))
        for t in range(2):
            graph.push_back(self.policy(t))
            graph.push_back(self.dynamics(t))
            graph.push_back(self.penalty(X(t)))
            graph.push_back(self.penalty(U(t)))
        graph.push_back(self.penalty(X(2)))
        return graph

    def bayes_net(self):
        """Eliminate backward in time."""
        return self.graph().eliminateSequential(
            make_ordering(X(2), U(1), X(1), U(0), X(0)))

    def test_expected_return(self):
        """J = -(1.625 * E[x0^2] + 1.7) with E[x0^2] = 5."""
        self.assertAlmostEqual(self.graph().expectation(), -9.825)

    def test_value_functions(self):
        """V1 = -(1.5 x^2 + 0.7) and V0 = -(1.625 x^2 + 1.7)."""
        expected = [lambda x: -(1.625 * x**2 + 1.7),
                    lambda x: -(1.5 * x**2 + 0.7)]
        expectedQ = [
            lambda x, u: -(x**2 + u**2 + 1.5 * (x + u)**2 + 1.45),
            lambda x, u: -(x**2 + u**2 + (x + u)**2 + 0.5)
        ]
        value = self.penalty(X(2))
        for t in reversed(range(2)):
            # phi(x, u) = E[value of the next state | x, u]
            phi = self.dynamics(t).multiply(value).sum(make_ordering(X(t + 1)))
            bucket = self.policy(t).multiply(self.penalty(X(t))).multiply(
                self.penalty(U(t))).multiply(phi)
            for x, u in [(2.0, -1.0), (1.0, -0.5), (-0.7, 0.3)]:
                self.assertAlmostEqual(
                    bucket.value(self.values(x=(X(t), x), u=(U(t), u))),
                    expectedQ[t](x, u))
            value = bucket.sum(make_ordering(U(t)))
            for x in [-1.0, 0.0, 2.0]:
                self.assertAlmostEqual(value.value(self.values(x=(X(t), x))),
                                       expected[t](x))

    def test_advantages(self):
        """A1 peaks at the policy mean, A0 at u = -0.6 x."""
        bayesNet = self.bayes_net()
        for x, u in [(2.0, -1.0), (2.0, -1.2), (1.0, -0.5), (-1.0, 3.0)]:
            self.assertAlmostEqual(
                bayesNet.at(1).surprise(
                    self.values(x=(X(1), x), u=(U(1), u))),
                0.2 - 2.0 * (u + 0.5 * x)**2)
            self.assertAlmostEqual(
                bayesNet.at(3).surprise(
                    self.values(x=(X(0), x), u=(U(0), u))),
                0.25 + 0.025 * x**2 - 2.5 * (u + 0.6 * x)**2)

    def deterministic_policy(self, t, gain):
        """The policy u = -gain * x as a hard constraint."""
        I = np.eye(1)
        return self.gaussian(U(t), I, X(t), gain * I, np.zeros(1),
                             noiseModel.Constrained.All(1))

    def test_best_policy_is_riccati(self):
        """Maximizing over the action is the Riccati recursion (chapter 2)."""
        # P and beta of V_t(x) = -(P x^2 + beta), and the gains, for t = 0, 1.
        expectedP, expectedBeta = [1.6, 1.5], [1.25, 0.5]
        expectedGain = [0.6, 0.5]
        value = self.penalty(X(2))
        for t in reversed(range(2)):
            # Eliminate the next state by expectation; no policy factor.
            phi = self.dynamics(t).multiply(value).sum(make_ordering(X(t + 1)))
            bucket = self.penalty(X(t)).multiply(self.penalty(U(t))).multiply(
                phi)
            # Eliminate the action by max: solve H_uu u = -H_ux x.
            Q = bucket.value()
            keys, H = list(Q.keys()), Q.information()
            u, x = keys.index(U(t)), keys.index(X(t))
            gain = H[u, x] / H[u, u]
            self.assertAlmostEqual(gain, expectedGain[t])
            value = self.deterministic_policy(t, gain).multiply(bucket).sum(
                make_ordering(U(t)))
            for position in [-1.0, 0.0, 2.0]:
                self.assertAlmostEqual(
                    value.value(self.values(x=(X(t), position))),
                    -(expectedP[t] * position**2 + expectedBeta[t]))
        prior = self.gaussian(X(0), np.eye(1), np.array([2.0]),
                              noiseModel.Isotropic.Variance(1, 1.0))
        self.assertAlmostEqual(prior.multiply(value).expectation(), -9.25)

    def test_deterministic_policy(self):
        """A policy without jitter is a constrained factor."""
        graph = SemiringFactorGraph()
        graph.push_back(
            self.gaussian(X(0), np.eye(1), np.array([2.0]),
                          noiseModel.Isotropic.Variance(1, 1.0)))
        for t in range(2):
            graph.push_back(self.deterministic_policy(t, 0.5))
            graph.push_back(self.dynamics(t))
            graph.push_back(self.penalty(X(t)))
            graph.push_back(self.penalty(U(t)))
        graph.push_back(self.penalty(X(2)))
        ordering = make_ordering(X(2), U(1), X(1), U(0), X(0))
        self.assertAlmostEqual(graph.expectation(ordering), -9.375)

    def test_trajectory_surprises(self):
        """Surprises along one trajectory add up to its return minus J."""
        trajectory = self.values(a=(X(0), 2.0), b=(U(0), -1.0), c=(X(1), 1.0),
                                 d=(U(1), -0.5), e=(X(2), 0.5))
        bayesNet = self.bayes_net()
        surprises = [bayesNet.at(i).surprise(trajectory) for i in range(5)]
        np.testing.assert_allclose(surprises, [0.5, 0.2, 0.75, 0.25, 1.625])
        self.assertAlmostEqual(sum(surprises), -6.5 - (-9.825))


class TestSemiringLinearQuadratic(GtsamTestCase):
    """A scalar linear-quadratic problem under a linear-Gaussian policy.

    x' = a x + b u + w,  u = k x + e,  cost q x^2 + r u^2, terminal qT x^2.
    """

    def setUp(self):
        self.horizon = 4
        self.a, self.b, self.k = 0.9, 0.5, -0.6
        self.q, self.r, self.qT = 1.0, 0.2, 3.0
        self.sigmaW, self.sigmaE = 0.3, 0.1
        self.mean0, self.sigma0 = 2.0, 0.5

    @staticmethod
    def cost(key, weight):
        """The cost factor weight * x^2, a negative reward."""
        return SemiringGaussianFactor.Cost(
            HessianFactor(key, 2.0 * weight * np.eye(1), np.zeros(1), 0.0))

    def graph(self, withPrior=True):
        """The factor graph of the problem under the fixed policy."""
        I = np.eye(1)
        graph = SemiringFactorGraph()
        if withPrior:
            graph.push_back(
                SemiringGaussianFactor(
                    JacobianFactor(X(0), I, np.array([self.mean0]),
                                   noiseModel.Isotropic.Sigma(1, self.sigma0))))
        for t in range(self.horizon):
            graph.push_back(
                SemiringGaussianFactor(
                    JacobianFactor(U(t), I, X(t), -self.k * I, np.zeros(1),
                                   noiseModel.Isotropic.Sigma(1, self.sigmaE))))
            graph.push_back(
                SemiringGaussianFactor(
                    JacobianFactor(X(t + 1), I, X(t), -self.a * I, U(t),
                                   -self.b * I, np.zeros(1),
                                   noiseModel.Isotropic.Sigma(1, self.sigmaW))))
            graph.push_back(self.cost(X(t), self.q))
            graph.push_back(self.cost(U(t), self.r))
        graph.push_back(self.cost(X(self.horizon), self.qT))
        return graph

    def backward_ordering(self):
        """Eliminate from the last state back to the first action."""
        ordering = Ordering()
        for t in reversed(range(self.horizon)):
            ordering.push_back(X(t + 1))
            ordering.push_back(U(t))
        return ordering

    def cost_to_go(self):
        """Lyapunov recursion: cost-to-go p[t] x^2 + c[t] of the policy."""
        p, c = [self.qT], [0.0]
        closedLoop = self.a + self.b * self.k
        for _ in range(self.horizon):
            c.insert(
                0, self.r * self.sigmaE**2 + p[0] *
                (self.b**2 * self.sigmaE**2 + self.sigmaW**2) + c[0])
            p.insert(0, self.q + self.r * self.k**2 + p[0] * closedLoop**2)
        return p, c

    @staticmethod
    def values(**scalars):
        """VectorValues from key=value pairs given as (key, value) tuples."""
        result = VectorValues()
        for key, value in scalars.values():
            result.insert(key, np.array([value]))
        return result

    def test_channels(self):
        """A lifted Gaussian has no value; a cost has no probability."""
        density = self.graph().at(0)
        self.assertIsInstance(density, SemiringGaussianFactor)
        self.assertIsNone(density.value())
        self.assertEqual(density.gaussian().size(), 1)
        self.assertAlmostEqual(density.error(self.values(x=(X(0), 3.0))), 2.0)

        cost = self.cost(X(0), 2.0)
        self.assertEqual(cost.gaussian().size(), 0)
        self.assertIsInstance(cost.value(), HessianFactor)
        self.assertAlmostEqual(cost.value(self.values(x=(X(0), 3.0))), -18.0)

    def test_expectation(self):
        """The expected total reward matches the Lyapunov recursion."""
        p, c = self.cost_to_go()
        expected = -(p[0] * (self.mean0**2 + self.sigma0**2) + c[0])
        graph = self.graph()
        self.assertAlmostEqual(graph.expectation(), expected)
        ordering = self.backward_ordering()
        ordering.push_back(X(0))
        self.assertAlmostEqual(graph.expectation(ordering), expected)

    def test_value_function(self):
        """Partial elimination leaves the quadratic value function V(x0)."""
        p, c = self.cost_to_go()
        bayesNet, remaining = self.graph(False).eliminatePartialSequential(
            self.backward_ordering())
        valueFunction = remaining.product()
        self.assertIsInstance(valueFunction, SemiringGaussianFactor)
        for x0 in [-1.0, 0.0, 1.5]:
            self.assertAlmostEqual(
                valueFunction.value(self.values(x=(X(0), x0))),
                -(p[0] * x0**2 + c[0]))

        # The last conditional is the policy with the advantage Q - V.
        policy = bayesNet.at(bayesNet.size() - 1)
        self.assertIsInstance(policy, SemiringGaussianConditional)
        self.assertEqual(policy.firstFrontalKey(), U(0))
        self.assertIsInstance(policy.conditional(), GaussianConditional)
        self.assertIsInstance(policy.surprise(), HessianFactor)

        x, u = 1.5, -0.4
        nextState = self.a * x + self.b * u
        Q = -(self.q * x**2 + self.r * u**2 + p[1] *
              (nextState**2 + self.sigmaW**2) + c[1])
        V = -(p[0] * x**2 + c[0])
        self.assertAlmostEqual(
            policy.surprise(self.values(x=(X(0), x), u=(U(0), u))), Q - V)

    def test_eliminate_factor(self):
        """Eliminate one bucket by hand with the factor interface."""
        I = np.eye(1)
        dynamics = SemiringGaussianFactor(
            JacobianFactor(X(1), I, X(0), -0.5 * I, np.zeros(1),
                           noiseModel.Isotropic.Sigma(1, 0.3)))
        bucket = dynamics.multiply(self.cost(X(1), 1.0))
        conditional, separator = bucket.eliminate(make_ordering(X(1)))
        self.assertIsInstance(conditional, SemiringGaussianConditional)
        self.assertEqual(separator.keys(), [X(0)])
        # E[-x1^2 | x0] = -(0.25 x0^2 + 0.09)
        self.assertAlmostEqual(separator.value(self.values(x=(X(0), 2.0))),
                               -1.09)

    def test_multifrontal(self):
        """Multifrontal elimination runs on Gaussian factors too."""
        bayesTree = self.graph().eliminateMultifrontal()
        self.assertGreater(bayesTree.size(), 0)
        clique = bayesTree[X(0)]
        self.assertIsInstance(clique.conditional(),
                              SemiringGaussianConditional)

    def test_mixed_families(self):
        """Discrete and Gaussian factors cannot be combined."""
        discrete = SemiringDiscreteFactor(DecisionTreeFactor((X(0), 2), "1 1"))
        self.assertRaises(ValueError, discrete.multiply,
                          self.cost(X(0), 1.0))


if __name__ == "__main__":
    unittest.main()
