/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testSemiringFactorGraph.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/semiring/SemiringBayesNet.h>
#include <gtsam/semiring/SemiringBayesTree.h>
#include <gtsam/semiring/SemiringDiscreteConditional.h>
#include <gtsam/semiring/SemiringFactorGraph.h>
#include <gtsam/semiring/SemiringGaussianConditional.h>
#include <gtsam/semiring/tests/TwoActionExample.h>

#include <map>
#include <vector>

using namespace gtsam;

GTSAM_CONCEPT_TESTABLE_INST(SemiringFactorGraph)
GTSAM_CONCEPT_TESTABLE_INST(SemiringBayesNet)

/* ************************************************************************* */
namespace one_decision {

using namespace two_action;

/// The graph of the two-action example.
SemiringFactorGraph createGraph() {
  SemiringFactorGraph graph;
  graph.emplace_shared<SemiringDiscreteFactor>(policy());
  graph.emplace_shared<SemiringDiscreteFactor>(actionReward());
  graph.emplace_shared<SemiringDiscreteFactor>(transition());
  graph.emplace_shared<SemiringDiscreteFactor>(stateReward());
  return graph;
}

// The product of all factors is the joint over trajectories, with the total
// reward of each trajectory as its value.
TEST(SemiringFactorGraph, Product) {
  const auto product = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      createGraph().product());
  CHECK(product);
  const auto [probability, value] = product->evaluate(assignment(L, bad));
  EXPECT_DOUBLES_EQUAL(0.6 * 0.1, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(-1.0, value, 1e-9);
  EXPECT(!SemiringFactorGraph().product());
}

// Eliminating backward in time is a Bellman backup: the conditional of the
// next state holds the surprise of each outcome, that of the action the
// advantage Q - V.
TEST(SemiringFactorGraph, EliminateSequential) {
  const Ordering ordering(KeyVector{S.first, A.first});
  const auto bayesNet = createGraph().eliminateSequential(ordering);
  EXPECT_LONGS_EQUAL(2, bayesNet->size());

  const auto state =
      std::dynamic_pointer_cast<SemiringDiscreteConditional>(bayesNet->at(0));
  CHECK(state);
  EXPECT((KeyVector{S.first, A.first}) == state->keys());
  const auto outcome = state->evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.9, outcome.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(10.0 - 9.0, outcome.second, 1e-9);

  const auto action =
      std::dynamic_pointer_cast<SemiringDiscreteConditional>(bayesNet->at(1));
  CHECK(action);
  EXPECT((KeyVector{A.first}) == action->keys());
  const auto left = action->evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.6, left.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(8.0 - 5.6, left.second, 1e-9);
}

// The expected total reward does not depend on the elimination order.
TEST(SemiringFactorGraph, Expectation) {
  const SemiringFactorGraph graph = createGraph();
  EXPECT_DOUBLES_EQUAL(
      5.6, graph.expectation(Ordering(KeyVector{S.first, A.first})), 1e-9);
  EXPECT_DOUBLES_EQUAL(
      5.6, graph.expectation(Ordering(KeyVector{A.first, S.first})), 1e-9);
  EXPECT_DOUBLES_EQUAL(5.6, graph.expectation(), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0, SemiringFactorGraph().expectation(), 1e-9);
}

// Partial elimination leaves the action values Q(a) in the remaining graph.
TEST(SemiringFactorGraph, EliminatePartialSequential) {
  SemiringFactorGraph graph;
  graph.emplace_shared<SemiringDiscreteFactor>(actionReward());
  graph.emplace_shared<SemiringDiscreteFactor>(transition());
  graph.emplace_shared<SemiringDiscreteFactor>(stateReward());

  const auto [bayesNet, remaining] =
      graph.eliminatePartialSequential(Ordering(KeyVector{S.first}));
  EXPECT_LONGS_EQUAL(1, bayesNet->size());
  const auto actionValue = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      remaining->product());
  CHECK(actionValue);
  EXPECT_DOUBLES_EQUAL(8.0, actionValue->evaluate(assignment(L, good)).second,
                       1e-9);
  EXPECT_DOUBLES_EQUAL(2.0, actionValue->evaluate(assignment(R, good)).second,
                       1e-9);
}

// The conditionals of a Bayes net are factors too, and since surprises
// average out, their product has expectation zero.
TEST(SemiringFactorGraph, BayesNetIsCentered) {
  const auto bayesNet = createGraph().eliminateSequential();
  const SemiringFactorGraph conditionals(*bayesNet);
  EXPECT_DOUBLES_EQUAL(0.0, conditionals.expectation(), 1e-9);
}

// Multifrontal elimination puts both variables in one clique, whose
// conditional is the joint with value R - E[R].
TEST(SemiringFactorGraph, EliminateMultifrontal) {
  const Ordering ordering(KeyVector{S.first, A.first});
  const auto bayesTree = createGraph().eliminateMultifrontal(ordering);
  EXPECT_LONGS_EQUAL(1, bayesTree->size());

  const auto joint = std::dynamic_pointer_cast<SemiringDiscreteConditional>(
      bayesTree->roots().front()->conditional());
  CHECK(joint);
  EXPECT_LONGS_EQUAL(2, joint->nrFrontals());
  const auto [probability, value] = joint->evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.6 * 0.9, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(9.0 - 5.6, value, 1e-9);
}

// A marginal from the Bayes tree holds the expected total reward given the
// variable, relative to the overall expectation.
TEST(SemiringFactorGraph, MarginalFactor) {
  const auto bayesTree = createGraph().eliminateMultifrontal();
  const auto marginal = std::dynamic_pointer_cast<SemiringDiscreteConditional>(
      bayesTree->marginalFactor(A.first));
  CHECK(marginal);
  const auto left = marginal->evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.6, left.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(8.0 - 5.6, left.second, 1e-9);
}

// Graphs compare equal if their factors do.
TEST(SemiringFactorGraph, Equals) {
  SemiringFactorGraph other = createGraph();
  EXPECT(assert_equal(createGraph(), other));
  other.emplace_shared<SemiringDiscreteFactor>(stateReward());
  EXPECT(!createGraph().equals(other));
}

}  // namespace one_decision
/* ************************************************************************* */
namespace markov_decision_process {

const size_t horizon = 3, nrStates = 3, nrActions = 2;

DiscreteKey stateKey(size_t t) { return {symbol_shorthand::X(t), nrStates}; }
DiscreteKey actionKey(size_t t) { return {symbol_shorthand::U(t), nrActions}; }

/// The tables of a small Markov decision process with a fixed policy.
struct Tables {
  std::vector<DecisionTreeFactor> probabilities;  ///< prior, policy, dynamics
  std::vector<DecisionTreeFactor> rewards;        ///< per step, and terminal

  Tables() {
    probabilities.emplace_back(stateKey(0), "0.5 0.3 0.2");
    for (size_t t = 0; t < horizon; t++) {
      // DiscreteConditional normalizes these tables over the frontal variable.
      probabilities.push_back(DiscreteConditional(
          1, DecisionTreeFactor(actionKey(t) & stateKey(t), "1 2 3 3 2 1")));
      probabilities.push_back(DiscreteConditional(
          1, DecisionTreeFactor(stateKey(t + 1) & stateKey(t) & actionKey(t),
                                "5 1 1 2 3 3   1 4 2 2 1 3   1 1 6 2 4 1")));
      rewards.emplace_back(stateKey(t) & actionKey(t), "1 0 -2 3 0.5 -1");
    }
    rewards.emplace_back(stateKey(horizon), "0 4 -3");
  }

  /// The semiring factor graph, optionally without the prior on the state.
  SemiringFactorGraph graph(bool withPrior = true) const {
    SemiringFactorGraph result;
    for (size_t i = withPrior ? 0 : 1; i < probabilities.size(); i++) {
      result.emplace_shared<SemiringDiscreteFactor>(probabilities[i]);
    }
    for (const DecisionTreeFactor& reward : rewards) {
      result.emplace_shared<SemiringDiscreteFactor>(
          SemiringDiscreteFactor::Reward(reward));
    }
    return result;
  }

  /// All variables, as discrete keys.
  DiscreteKeys allKeys() const {
    DiscreteKeys keys;
    for (size_t t = 0; t < horizon; t++) {
      keys.push_back(stateKey(t));
      keys.push_back(actionKey(t));
    }
    keys.push_back(stateKey(horizon));
    return keys;
  }

  /**
   * Enumerate all trajectories to get the probability of each initial state
   * and the expected total reward when starting from it.
   */
  std::pair<Vector, Vector> bruteForce() const {
    Vector probability = Vector::Zero(nrStates);
    Vector weighted = Vector::Zero(nrStates);
    for (const auto& values : DiscreteValues::CartesianProduct(allKeys())) {
      double p = 1.0, totalReward = 0.0;
      for (const auto& factor : probabilities) p *= factor(values);
      for (const auto& reward : rewards) totalReward += reward(values);
      const size_t initialState = values.at(stateKey(0).first);
      probability(initialState) += p;
      weighted(initialState) += p * totalReward;
    }
    return {probability, weighted.cwiseQuotient(probability)};
  }
};

// The expected total reward from elimination matches the enumeration of all
// trajectories, for any elimination order.
TEST(SemiringFactorGraph, ExpectedTotalReward) {
  const Tables tables;
  const auto [probability, value] = tables.bruteForce();
  const double expected = probability.dot(value);

  const SemiringFactorGraph graph = tables.graph();
  EXPECT_DOUBLES_EQUAL(expected, graph.expectation(), 1e-9);
  EXPECT_DOUBLES_EQUAL(
      expected, graph.expectation(Ordering::Natural(graph)), 1e-9);
  EXPECT_DOUBLES_EQUAL(expected, graph.product()->expectation(), 1e-9);
}

// Eliminating everything but the initial state leaves the value function
// V(s0), the expected total reward when starting from each state.
TEST(SemiringFactorGraph, ValueFunction) {
  const Tables tables;
  const Vector expected = tables.bruteForce().second;

  const SemiringFactorGraph graph = tables.graph(false);
  Ordering ordering;
  for (size_t t = horizon; t > 0; t--) {
    ordering.push_back(stateKey(t).first);
    ordering.push_back(actionKey(t - 1).first);
  }
  const auto remaining = graph.eliminatePartialSequential(ordering).second;
  const auto valueFunction =
      std::dynamic_pointer_cast<SemiringDiscreteFactor>(remaining->product());
  CHECK(valueFunction);
  EXPECT((KeyVector{stateKey(0).first}) == valueFunction->keys());
  for (size_t state = 0; state < nrStates; state++) {
    const auto [probability, value] =
        valueFunction->evaluate({{stateKey(0).first, state}});
    EXPECT_DOUBLES_EQUAL(1.0, probability, 1e-9);
    EXPECT_DOUBLES_EQUAL(expected(state), value, 1e-9);
  }
}

// Sequential and multifrontal elimination agree on the marginal of the
// initial state and on its value relative to the expected total reward.
TEST(SemiringFactorGraph, MultifrontalMarginal) {
  const Tables tables;
  const auto [probability, value] = tables.bruteForce();
  const double total = probability.dot(value);

  const auto bayesTree = tables.graph().eliminateMultifrontal();
  const auto marginal = std::dynamic_pointer_cast<SemiringDiscreteConditional>(
      bayesTree->marginalFactor(stateKey(0).first));
  CHECK(marginal);
  for (size_t state = 0; state < nrStates; state++) {
    const auto actual = marginal->evaluate({{stateKey(0).first, state}});
    EXPECT_DOUBLES_EQUAL(probability(state), actual.first, 1e-9);
    EXPECT_DOUBLES_EQUAL(value(state) - total, actual.second, 1e-9);
  }
}

}  // namespace markov_decision_process
/* ************************************************************************* */
namespace linear_quadratic {

using symbol_shorthand::U;
using symbol_shorthand::X;

/**
 * A scalar linear-quadratic problem under a fixed linear-Gaussian policy:
 *   x' = a x + b u + w,  u = k x + e,  cost q x^2 + r u^2, terminal qT x^2.
 */
struct Problem {
  const size_t horizon = 4;
  const double a = 0.9, b = 0.5, k = -0.6;
  const double q = 1.0, r = 0.2, qT = 3.0;
  const double sigmaW = 0.3, sigmaE = 0.1;
  const double mean0 = 2.0, sigma0 = 0.5;

  static GaussianFactor::shared_ptr quadraticCost(Key key, double weight) {
    return std::make_shared<HessianFactor>(key, 2.0 * weight * I_1x1,
                                           Vector1(0.0), 0.0);
  }

  /// The semiring factor graph, optionally without the prior on the state.
  SemiringFactorGraph graph(bool withPrior = true) const {
    SemiringFactorGraph result;
    if (withPrior) {
      result.emplace_shared<SemiringGaussianFactor>(
          std::make_shared<JacobianFactor>(
              X(0), I_1x1, Vector1(mean0),
              noiseModel::Isotropic::Sigma(1, sigma0)));
    }
    for (size_t t = 0; t < horizon; t++) {
      result.emplace_shared<SemiringGaussianFactor>(
          std::make_shared<JacobianFactor>(
              U(t), I_1x1, X(t), -k * I_1x1, Vector1(0.0),
              noiseModel::Isotropic::Sigma(1, sigmaE)));
      result.emplace_shared<SemiringGaussianFactor>(
          std::make_shared<JacobianFactor>(
              X(t + 1), I_1x1, X(t), -a * I_1x1, U(t), -b * I_1x1, Vector1(0.0),
              noiseModel::Isotropic::Sigma(1, sigmaW)));
      result.emplace_shared<SemiringGaussianFactor>(
          SemiringGaussianFactor::Cost(*quadraticCost(X(t), q)));
      result.emplace_shared<SemiringGaussianFactor>(
          SemiringGaussianFactor::Cost(*quadraticCost(U(t), r)));
    }
    result.emplace_shared<SemiringGaussianFactor>(
        SemiringGaussianFactor::Cost(*quadraticCost(X(horizon), qT)));
    return result;
  }

  /// Eliminate backward in time, from the last state to the first action.
  Ordering backwardOrdering() const {
    Ordering ordering;
    for (size_t t = horizon; t > 0; t--) {
      ordering.push_back(X(t));
      ordering.push_back(U(t - 1));
    }
    return ordering;
  }

  /**
   * The cost-to-go p[t] x^2 + c[t] of the policy at every step, by the
   * Lyapunov recursion.
   */
  std::pair<std::vector<double>, std::vector<double>> costToGo() const {
    std::vector<double> p(horizon + 1), c(horizon + 1);
    p[horizon] = qT;
    c[horizon] = 0.0;
    for (size_t t = horizon; t-- > 0;) {
      const double closedLoop = a + b * k;
      p[t] = q + r * k * k + p[t + 1] * closedLoop * closedLoop;
      c[t] = r * sigmaE * sigmaE +
             p[t + 1] * (b * b * sigmaE * sigmaE + sigmaW * sigmaW) + c[t + 1];
    }
    return {p, c};
  }
};

/// A value for a single scalar variable.
VectorValues scalar(Key key, double value) {
  VectorValues values;
  values.insert(key, Vector1(value));
  return values;
}

// The expected total reward matches the Lyapunov recursion, for any
// elimination order.
TEST(SemiringFactorGraph, ExpectedTotalCost) {
  const Problem problem;
  const auto [p, c] = problem.costToGo();
  const double expected =
      -(p[0] * (problem.mean0 * problem.mean0 +
                problem.sigma0 * problem.sigma0) +
        c[0]);

  const SemiringFactorGraph graph = problem.graph();
  Ordering backward = problem.backwardOrdering();
  backward.push_back(X(0));
  EXPECT_DOUBLES_EQUAL(expected, graph.expectation(backward), 1e-9);
  EXPECT_DOUBLES_EQUAL(expected, graph.expectation(), 1e-9);
  EXPECT_DOUBLES_EQUAL(
      expected, graph.expectation(Ordering::Natural(graph)), 1e-9);
}

// Eliminating everything but the initial state leaves the quadratic value
// function V(x0) = -(p0 x0^2 + c0).
TEST(SemiringFactorGraph, QuadraticValueFunction) {
  const Problem problem;
  const auto [p, c] = problem.costToGo();

  const auto remaining = problem.graph(false)
                             .eliminatePartialSequential(
                                 problem.backwardOrdering())
                             .second;
  const auto valueFunction =
      std::dynamic_pointer_cast<SemiringGaussianFactor>(remaining->product());
  CHECK(valueFunction);
  for (double x0 : {-1.0, 0.0, 1.5}) {
    EXPECT_DOUBLES_EQUAL(-(p[0] * x0 * x0 + c[0]),
                         valueFunction->value(scalar(X(0), x0)), 1e-9);
  }
}

// The conditional of the first action is the policy, with the advantage
// function A(x, u) = Q(x, u) - V(x).
TEST(SemiringFactorGraph, QuadraticAdvantage) {
  const Problem problem;
  const auto [p, c] = problem.costToGo();

  const auto bayesNet = problem.graph(false)
                            .eliminatePartialSequential(
                                problem.backwardOrdering())
                            .first;
  const auto policy = std::dynamic_pointer_cast<SemiringGaussianConditional>(
      bayesNet->back());
  CHECK(policy);
  EXPECT_LONGS_EQUAL(U(0), policy->firstFrontalKey());

  const double x = 1.5, u = -0.4;
  const double next = problem.a * x + problem.b * u;
  const double Q =
      -(problem.q * x * x + problem.r * u * u +
        p[1] * (next * next + problem.sigmaW * problem.sigmaW) + c[1]);
  const double V = -(p[0] * x * x + c[0]);
  VectorValues values = scalar(X(0), x);
  values.insert(U(0), Vector1(u));
  EXPECT_DOUBLES_EQUAL(Q - V, policy->surprise(values), 1e-9);
}

// Multifrontal elimination runs on Gaussian factors as well.
TEST(SemiringFactorGraph, GaussianMultifrontal) {
  const Problem problem;
  const auto bayesTree = problem.graph().eliminateMultifrontal();
  size_t nrFrontals = 0;
  for (const auto& [key, clique] : bayesTree->nodes()) {
    if (clique->conditional()->firstFrontalKey() == key) {
      nrFrontals += clique->conditional()->nrFrontals();
    }
  }
  EXPECT_LONGS_EQUAL(2 * problem.horizon + 1, nrFrontals);
}

}  // namespace linear_quadratic
/* ************************************************************************* */
namespace mixed_families {

// Discrete and Gaussian factors cannot share a graph.
TEST(SemiringFactorGraph, MixedFamilies) {
  SemiringFactorGraph graph;
  graph.emplace_shared<SemiringDiscreteFactor>(two_action::policy());
  graph.emplace_shared<SemiringGaussianFactor>(
      std::make_shared<JacobianFactor>(two_action::A.first, I_1x1,
                                       Vector1(0.0),
                                       noiseModel::Unit::Create(1)));
  CHECK_EXCEPTION(graph.product(), std::invalid_argument);
  CHECK_EXCEPTION(graph.eliminateSequential(), std::invalid_argument);
}

}  // namespace mixed_families
/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
