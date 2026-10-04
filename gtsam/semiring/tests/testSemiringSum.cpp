/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testSemiringSum.cpp
 * @brief   Unit tests for the rules by which a variable is summed out: the
 *          average, the maximum and the tilted mean
 * @author  Duy Ta
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/semiring/SemiringBayesNet.h>
#include <gtsam/semiring/SemiringDiscreteConditional.h>
#include <gtsam/semiring/SemiringFactorGraph.h>
#include <gtsam/semiring/SemiringGaussianConditional.h>
#include <gtsam/semiring/SemiringSum.h>
#include <gtsam/semiring/tests/TwoActionExample.h>

#include <cmath>
#include <vector>

using namespace gtsam;
using symbol_shorthand::U;
using symbol_shorthand::X;

/* ************************************************************************* */
namespace rules {

// The named constructors give the three kinds of rule; a zero tilt is the
// average and the soft maximum is a tilt of one over the temperature.
TEST(SemiringSum, Constructors) {
  EXPECT(SemiringSum().isAverage());
  EXPECT(SemiringSum::Average().isAverage());
  EXPECT(SemiringSum::Maximum().isMaximum());
  EXPECT(SemiringSum::Tilted(0.5).isTilted());
  EXPECT(SemiringSum::Tilted(0.0).isAverage());
  EXPECT_DOUBLES_EQUAL(0.5, SemiringSum::SoftMaximum(2.0).tilt(), 1e-12);
  EXPECT(assert_equal(SemiringSum::Tilted(0.5), SemiringSum::SoftMaximum(2.0)));
  EXPECT(!SemiringSum::Maximum().equals(SemiringSum::Average()));
  CHECK_EXCEPTION(SemiringSum::SoftMaximum(0.0), std::invalid_argument);
}

// A variable without a rule is averaged, and variables eliminated together
// must share one rule.
TEST(SemiringRules, Common) {
  SemiringRules rules;
  EXPECT(rules.at(X(0)).isAverage());
  rules.set(U(0), SemiringSum::Maximum());
  rules.setAll({U(1), U(2)}, SemiringSum::Tilted(2.0));
  EXPECT_LONGS_EQUAL(3, rules.size());
  EXPECT(rules.at(U(0)).isMaximum());
  EXPECT(rules.common(Ordering(KeyVector{U(1), U(2)})).isTilted());
  EXPECT(rules.common(Ordering(KeyVector{X(0), X(1)})).isAverage());
  CHECK_EXCEPTION(rules.common(Ordering(KeyVector{U(0), X(0)})),
                  std::invalid_argument);
}

}  // namespace rules
/* ************************************************************************* */
namespace one_decision {

using namespace two_action;

const Ordering actionOrdering(KeyVector{A.first});

/// The bucket of the action without a policy: (1, Q), with Q = (8, 2).
SemiringDiscreteFactor actionValues() {
  const auto future = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      (transition() * stateReward()).sum(Ordering(KeyVector{S.first})));
  return actionReward() * *future;
}

/// Two numbers as a vector, to compare tables on the action.
Vector pair(double left, double right) {
  return (Vector(2) << left, right).finished();
}

/// The table of a factor or conditional on the action alone.
Vector onAction(const DecisionTreeFactor& table) {
  return pair(table(DiscreteValues{{A.first, L}}),
              table(DiscreteValues{{A.first, R}}));
}

// The maximum keeps the better action. The probabilities are added, and the
// conditional holds the regret of each action.
TEST(SemiringDiscreteFactor, Maximum) {
  const auto [conditional, marginal] =
      actionValues().eliminate(actionOrdering, SemiringSum::Maximum());
  const auto best = std::dynamic_pointer_cast<SemiringDiscreteFactor>(marginal);
  CHECK(best);
  const auto [probability, value] = best->evaluate(DiscreteValues());
  EXPECT_DOUBLES_EQUAL(2.0, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(8.0, value, 1e-9);
  EXPECT_DOUBLES_EQUAL(8.0, marginal->expectation(), 1e-9);

  const auto discrete =
      std::dynamic_pointer_cast<SemiringDiscreteConditional>(conditional);
  CHECK(discrete);
  EXPECT(assert_equal(pair(0.5, 0.5), onAction(discrete->probability())));
  EXPECT(assert_equal(pair(0.0, -6.0), onAction(discrete->surprise())));
  EXPECT(assert_equal(pair(1.0, 0.0), onAction(discrete->greedy())));
}

// With a policy factor in the bucket the maximum is unchanged: the policy
// only fills the probability channel.
TEST(SemiringDiscreteFactor, MaximumIgnoresPolicy) {
  const SemiringDiscreteFactor bucket = policy() * actionValues();
  const auto [conditional, marginal] =
      bucket.eliminate(actionOrdering, SemiringSum::Maximum());
  EXPECT_DOUBLES_EQUAL(8.0, marginal->expectation(), 1e-9);
  const auto discrete =
      std::dynamic_pointer_cast<SemiringDiscreteConditional>(conditional);
  EXPECT(assert_equal(pair(0.6, 0.4), onAction(discrete->probability())));
  EXPECT(assert_equal(pair(0.0, -6.0), onAction(discrete->surprise())));
}

// An outcome with zero probability cannot win the maximum, however large its
// value.
TEST(SemiringDiscreteFactor, MaximumSkipsImpossible) {
  const SemiringDiscreteFactor factor(DecisionTreeFactor(A, "0 1"),
                                      DecisionTreeFactor(A, "100 -3"));
  EXPECT_DOUBLES_EQUAL(
      -3.0, factor.sum(actionOrdering, SemiringSum::Maximum())->expectation(),
      1e-9);
}

// The tilted mean is (1 / tilt) log sum_a pi(a) exp(tilt Q(a)), and the
// conditional holds Q minus it.
TEST(SemiringDiscreteFactor, Tilted) {
  const SemiringDiscreteFactor bucket = policy() * actionValues();
  for (double tilt : {0.5, -0.5, 2.0}) {
    const double expected =
        std::log(0.6 * std::exp(tilt * 8.0) + 0.4 * std::exp(tilt * 2.0)) /
        tilt;
    const auto [conditional, marginal] =
        bucket.eliminate(actionOrdering, SemiringSum::Tilted(tilt));
    EXPECT_DOUBLES_EQUAL(expected, marginal->expectation(), 1e-9);
    const auto discrete =
        std::dynamic_pointer_cast<SemiringDiscreteConditional>(conditional);
    EXPECT(assert_equal(pair(8.0 - expected, 2.0 - expected),
                        onAction(discrete->surprise()), 1e-9));
    // The tilted policy pi exp(tilt * surprise) is normalized as it stands.
    const Vector tilted = pair(0.6 * std::exp(tilt * (8.0 - expected)),
                               0.4 * std::exp(tilt * (2.0 - expected)));
    EXPECT_DOUBLES_EQUAL(1.0, tilted.sum(), 1e-9);
    EXPECT(assert_equal(tilted, onAction(discrete->tilted(tilt)), 1e-9));
  }
}

// The tilted mean runs from the smallest value to the largest as the tilt
// grows, and does not overflow for a strong tilt.
TEST(SemiringDiscreteFactor, TiltedLimits) {
  const SemiringDiscreteFactor bucket = policy() * actionValues();
  const auto tiltedMean = [&bucket](double tilt) {
    return bucket.sum(actionOrdering, SemiringSum::Tilted(tilt))
        ->expectation();
  };
  EXPECT_DOUBLES_EQUAL(5.6, tiltedMean(1e-9), 1e-6);  // the average
  EXPECT_DOUBLES_EQUAL(8.0, tiltedMean(1000.0), 1e-2);  // the maximum
  EXPECT_DOUBLES_EQUAL(2.0, tiltedMean(-1000.0), 1e-2);  // the minimum
  EXPECT(tiltedMean(-0.5) < 5.6 && 5.6 < tiltedMean(0.5));
}

// A conditional is normalized for the rule that produced it: summing it out
// again by that rule gives the semiring one, probability one and value zero.
TEST(SemiringDiscreteConditional, NormalizedForEveryRule) {
  const SemiringDiscreteFactor bucket = policy() * actionValues();
  for (const SemiringSum& rule :
       {SemiringSum::Average(), SemiringSum::Maximum(),
        SemiringSum::Tilted(0.7), SemiringSum::Tilted(-0.7)}) {
    const auto conditional = bucket.eliminate(actionOrdering, rule).first;
    const auto one = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
        conditional->sum(actionOrdering, rule));
    CHECK(one);
    const auto [probability, value] = one->evaluate(DiscreteValues());
    EXPECT_DOUBLES_EQUAL(1.0, probability, 1e-9);
    EXPECT_DOUBLES_EQUAL(0.0, value, 1e-9);
  }
}

}  // namespace one_decision
/* ************************************************************************* */
namespace track {

const size_t L = 0, R = 1;

DiscreteKey state(size_t t) { return {X(t), 3}; }
DiscreteKey action(size_t t) { return {U(t), 2}; }

/**
 * A robot on three cells makes two moves, Left or Right. A move succeeds with
 * probability 0.8, moving Right costs 1, and ending in cell 2 pays 10. The
 * robot starts in cell 0 or 1.
 */
SemiringFactorGraph graph(bool withCoinFlipPolicy) {
  SemiringFactorGraph result;
  result.emplace_shared<SemiringDiscreteFactor>(
      DecisionTreeFactor(state(0), "0.5 0.5 0"));
  for (size_t t = 0; t < 2; t++) {
    if (withCoinFlipPolicy) {
      result.emplace_shared<SemiringDiscreteFactor>(DecisionTreeFactor(
          state(t) & action(t), "0.5 0.5  0.5 0.5  0.5 0.5"));
    }
    result.emplace_shared<SemiringDiscreteFactor>(DecisionTreeFactor(
        state(t) & action(t) & state(t + 1),
        "1 0 0  0.2 0.8 0   0.8 0.2 0  0 0.2 0.8   0 0.8 0.2  0 0 1"));
    result.emplace_shared<SemiringDiscreteFactor>(
        SemiringDiscreteFactor::Reward(
            DecisionTreeFactor(state(t) & action(t), "0 -1  0 -1  0 -1")));
  }
  result.emplace_shared<SemiringDiscreteFactor>(
      SemiringDiscreteFactor::Reward(DecisionTreeFactor(state(2), "0 0 10")));
  return result;
}

const Ordering backward(KeyVector{X(2), U(1), X(1), U(0), X(0)});

/// The same rule for every action, and the average for the states.
SemiringRules atActions(const SemiringSum& rule) {
  SemiringRules rules;
  rules.setAll({U(0), U(1)}, rule);
  return rules;
}

// The average at the states and the maximum at the actions is dynamic
// programming: the best expected return, 6.1, and the best policy.
TEST(SemiringFactorGraph, DynamicProgramming) {
  const SemiringFactorGraph graph = track::graph(false);
  const SemiringRules rules = atActions(SemiringSum::Maximum());
  EXPECT_DOUBLES_EQUAL(6.1, graph.expectation(backward, rules), 1e-9);

  // Eliminating all but the first state leaves the best value function.
  const auto [bayesNet, remaining] = graph.eliminatePartialSequential(
      Ordering(KeyVector{X(2), U(1), X(1), U(0)}), rules);
  // The product of what remains is the prior with the value V*_0(s0).
  const auto value = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      remaining->product());
  CHECK(value);
  const DecisionTreeFactor values = value->value();
  EXPECT_DOUBLES_EQUAL(4.6, values(DiscreteValues{{X(0), 0}}), 1e-9);
  EXPECT_DOUBLES_EQUAL(7.6, values(DiscreteValues{{X(0), 1}}), 1e-9);

  // The conditionals of the actions hold the best moves: Right, except Left
  // in cell 0 at the last move.
  const auto last = std::dynamic_pointer_cast<SemiringDiscreteConditional>(
      bayesNet->at(1));
  const auto first = std::dynamic_pointer_cast<SemiringDiscreteConditional>(
      bayesNet->at(3));
  CHECK(last && first);
  const DiscreteConditional lastPolicy = last->greedy();
  const DiscreteConditional firstPolicy = first->greedy();
  const std::vector<size_t> bestLast = {L, R, R}, bestFirst = {R, R, R};
  for (size_t s = 0; s < 3; s++) {
    EXPECT_DOUBLES_EQUAL(
        1.0, lastPolicy(DiscreteValues{{U(1), bestLast[s]}, {X(1), s}}), 1e-9);
    EXPECT_DOUBLES_EQUAL(
        1.0, firstPolicy(DiscreteValues{{U(0), bestFirst[s]}, {X(0), s}}),
        1e-9);
  }
  // The regret of the other move at the last step: Q* - V* with
  // Q*_1 = [[0, -1], [0, 7], [2, 9]].
  const DecisionTreeFactor regret = last->surprise();
  EXPECT_DOUBLES_EQUAL(-1.0, regret(DiscreteValues{{U(1), R}, {X(1), 0}}),
                       1e-9);
  EXPECT_DOUBLES_EQUAL(-7.0, regret(DiscreteValues{{U(1), L}, {X(1), 1}}),
                       1e-9);
  EXPECT_DOUBLES_EQUAL(-7.0, regret(DiscreteValues{{U(1), L}, {X(1), 2}}),
                       1e-9);
}

// The maximum at every variable, the states included, is the return of the
// best trajectory that can occur, 9, which counts on a lucky slip.
TEST(SemiringFactorGraph, MaximumEverywhere) {
  SemiringRules rules;
  rules.setAll({X(0), U(0), X(1), U(1), X(2)}, SemiringSum::Maximum());
  EXPECT_DOUBLES_EQUAL(9.0, track::graph(false).expectation(backward, rules),
                       1e-9);
  EXPECT_DOUBLES_EQUAL(9.0, track::graph(true).expectation(backward, rules),
                       1e-9);
}

// The same tilt at every variable gives the tilted mean of the return under
// the coin-flip policy, (1 / tilt) log E[exp(tilt R)], in any order.
TEST(SemiringFactorGraph, TiltedEverywhere) {
  const SemiringFactorGraph graph = track::graph(true);
  for (const auto& [tilt, expected] :
       std::vector<std::pair<double, double>>{
           {0.5, 5.4251}, {-0.5, -0.2768}, {2.0, 7.6490}}) {
    SemiringRules rules;
    rules.setAll({X(0), U(0), X(1), U(1), X(2)}, SemiringSum::Tilted(tilt));
    EXPECT_DOUBLES_EQUAL(expected, graph.expectation(backward, rules), 1e-4);
    EXPECT_DOUBLES_EQUAL(
        expected,
        graph.expectation(Ordering(KeyVector{X(0), U(0), X(1), U(1), X(2)}),
                          rules),
        1e-4);
  }
}

// The average at the states and the soft maximum at the actions lies between
// the value of the coin flip, 1.4, and the best value, 6.1, and tends to each
// as the temperature grows or shrinks.
TEST(SemiringFactorGraph, SoftMaximumAtActions) {
  const SemiringFactorGraph graph = track::graph(true);
  const auto softValue = [&graph](double temperature) {
    return graph.expectation(
        backward, atActions(SemiringSum::SoftMaximum(temperature)));
  };
  EXPECT_DOUBLES_EQUAL(4.7536, softValue(1.0), 1e-4);
  EXPECT_DOUBLES_EQUAL(1.4, softValue(1e6), 1e-4);
  EXPECT_DOUBLES_EQUAL(6.1, softValue(1e-3), 1e-2);
  EXPECT(softValue(2.0) < softValue(1.0) && softValue(1.0) < softValue(0.5));
}

// Variables with different rules cannot share one elimination step, as
// happens in a multifrontal clique.
TEST(SemiringFactorGraph, MixedRulesInOneStep) {
  const SemiringFactorGraph graph = track::graph(false);
  const SemiringRules rules = atActions(SemiringSum::Maximum());
  CHECK_EXCEPTION(
      EliminateSemiringWith(rules)(graph, Ordering(KeyVector{X(2), U(1)})),
      std::invalid_argument);
}

}  // namespace track
/* ************************************************************************* */
namespace line {

const double noise = 0.5, priorMean = 2.0, priorVariance = 1.0;

/// The penalty z^2 on one scalar variable, as the reward (1, -z^2).
SemiringGaussianFactor penalty(Key key) {
  return SemiringGaussianFactor::Cost(
      HessianFactor(key, 2.0 * I_1x1, Vector1(0.0), 0.0));
}

/**
 * A robot on a line makes two moves, x' = x + u + w with w ~ N(0, 0.5), and
 * pays x^2 + u^2 per move and x^2 at the end; x0 ~ N(2, 1). There is no
 * policy factor.
 */
SemiringFactorGraph graph() {
  SemiringFactorGraph result;
  result.emplace_shared<SemiringGaussianFactor>(
      std::make_shared<JacobianFactor>(
          X(0), I_1x1, Vector1(priorMean),
          noiseModel::Isotropic::Variance(1, priorVariance)));
  for (size_t t = 0; t < 2; t++) {
    result.emplace_shared<SemiringGaussianFactor>(
        std::make_shared<JacobianFactor>(
            X(t + 1), I_1x1, X(t), -I_1x1, U(t), -I_1x1, Vector1(0.0),
            noiseModel::Isotropic::Variance(1, noise)));
    result.emplace_shared<SemiringGaussianFactor>(penalty(X(t)));
    result.emplace_shared<SemiringGaussianFactor>(penalty(U(t)));
  }
  result.emplace_shared<SemiringGaussianFactor>(penalty(X(2)));
  return result;
}

const Ordering backward(KeyVector{X(2), U(1), X(1), U(0), X(0)});

/// The maximum at the actions and a tilt, possibly zero, at the states.
SemiringRules rules(double tilt) {
  SemiringRules result;
  result.setAll({U(0), U(1)}, SemiringSum::Maximum());
  result.setAll({X(0), X(1), X(2)}, SemiringSum::Tilted(tilt));
  return result;
}

/// The gain K of the conditional u = -K x left on an action.
double gain(const SemiringConditional::shared_ptr& conditional) {
  const auto gaussian =
      std::dynamic_pointer_cast<SemiringGaussianConditional>(conditional);
  // The conditional is u + K x = 0, so K is the block on the parent.
  return gaussian->conditional()->S()(0, 0);
}

/**
 * The risk-sensitive Riccati recursion for this problem: the gains and the
 * tilted value at the root. A zero tilt gives the ordinary recursion.
 */
struct Recursion {
  double gain0, gain1, value;

  explicit Recursion(double tilt) {
    double P = 1.0, constant = 0.0;
    std::vector<double> gains;
    for (size_t step = 0; step < 2; step++) {
      const double tilted = P / (1.0 + 2.0 * tilt * P * noise);
      constant += tilt == 0.0 ? P * noise
                              : std::log(1.0 + 2.0 * tilt * P * noise) /
                                    (2.0 * tilt);
      gains.push_back(tilted / (1.0 + tilted));
      P = 1.0 + tilted - tilted * tilted / (1.0 + tilted);
    }
    gain1 = gains[0];
    gain0 = gains[1];
    const double tilted = P / (1.0 + 2.0 * tilt * P * priorVariance);
    constant += tilt == 0.0 ? P * priorVariance
                            : std::log(1.0 + 2.0 * tilt * P * priorVariance) /
                                  (2.0 * tilt);
    value = -(tilted * priorMean * priorMean + constant);
  }
};

// The average at the states and the maximum at the actions is the Riccati
// recursion: gains 0.6 and 0.5, and the best expected return -9.25.
TEST(SemiringFactorGraph, Riccati) {
  const SemiringFactorGraph graph = line::graph();
  EXPECT_DOUBLES_EQUAL(-9.25, graph.expectation(backward, rules(0.0)), 1e-9);
  const auto bayesNet = graph.eliminateSequential(backward, rules(0.0));
  EXPECT_DOUBLES_EQUAL(0.5, gain(bayesNet->at(1)), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.6, gain(bayesNet->at(3)), 1e-9);

  // The value function after one backward step is -(1.5 x^2 + 0.5).
  const auto remaining =
      graph.eliminatePartialSequential(Ordering(KeyVector{X(2), U(1)}),
                                       rules(0.0))
          .second;
  SemiringFactorGraph values;
  for (const auto& factor : *remaining) {
    if (factor->size() == 1 && factor->front() == X(1)) {
      values.push_back(factor);
    }
  }
  const auto value =
      std::dynamic_pointer_cast<SemiringGaussianFactor>(values.product());
  CHECK(value);
  VectorValues x;
  x.insert(X(1), Vector1(2.0));
  // The stage reward -x^2 of step 1 is still a separate factor in `values`.
  EXPECT_DOUBLES_EQUAL(-(1.5 * 4.0 + 0.5), value->value(x), 1e-9);

  // The conditional of an action carries the regret, zero at the best action
  // and -H_uu (u + K x)^2 elsewhere, with H_uu = 2 at the last move.
  const auto last =
      std::dynamic_pointer_cast<SemiringGaussianConditional>(bayesNet->at(1));
  VectorValues assignment;
  assignment.insert(X(1), Vector1(2.0));
  assignment.insert(U(1), Vector1(-1.0));
  EXPECT_DOUBLES_EQUAL(0.0, last->surprise(assignment), 1e-9);
  assignment.at(U(1)) = Vector1(0.0);
  EXPECT_DOUBLES_EQUAL(-2.0, last->surprise(assignment), 1e-9);
}

// A tilt at the states gives risk-sensitive control: the gains and the tilted
// value of the risk-sensitive Riccati recursion.
TEST(SemiringFactorGraph, RiskSensitiveRiccati) {
  const SemiringFactorGraph graph = line::graph();
  for (double tilt : {-0.25, -0.1, 0.5, 1.0, 2.0}) {
    const Recursion expected(tilt);
    EXPECT_DOUBLES_EQUAL(expected.value,
                         graph.expectation(backward, rules(tilt)), 1e-8);
    const auto bayesNet = graph.eliminateSequential(backward, rules(tilt));
    EXPECT_DOUBLES_EQUAL(expected.gain1, gain(bayesNet->at(1)), 1e-9);
    EXPECT_DOUBLES_EQUAL(expected.gain0, gain(bayesNet->at(3)), 1e-9);
  }
  // The numbers of the book for a tilt of one.
  EXPECT_DOUBLES_EQUAL(0.3636, Recursion(1.0).gain0, 1e-4);
  EXPECT_DOUBLES_EQUAL(0.3333, Recursion(1.0).gain1, 1e-4);
  EXPECT_DOUBLES_EQUAL(-2.89, Recursion(1.0).value, 5e-3);
}

// A negative tilt that is too strong for the noise has no finite tilted
// mean, and elimination says so.
TEST(SemiringFactorGraph, TiltBreakdown) {
  CHECK_EXCEPTION(line::graph().expectation(backward, rules(-0.4)),
                  std::invalid_argument);
}

// The maximum needs a variable without a density and a value that is
// concave in it.
TEST(SemiringGaussianFactor, MaximumRequirements) {
  const Ordering action(KeyVector{U(0)});
  // A density on the action, e.g., a policy.
  const SemiringGaussianFactor policy(std::make_shared<JacobianFactor>(
      U(0), I_1x1, X(0), 0.5 * I_1x1, Vector1(0.0),
      noiseModel::Isotropic::Variance(1, 0.1)));
  CHECK_EXCEPTION(
      policy.multiply(penalty(U(0)))->eliminate(action, SemiringSum::Maximum()),
      std::invalid_argument);
  // A reward +u^2 is convex in the action and has no maximum.
  const SemiringGaussianFactor convex = SemiringGaussianFactor::Reward(
      HessianFactor(U(0), 2.0 * I_1x1, Vector1(0.0), 0.0));
  CHECK_EXCEPTION(convex.eliminate(action, SemiringSum::Maximum()),
                  std::invalid_argument);
}

}  // namespace line
/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
