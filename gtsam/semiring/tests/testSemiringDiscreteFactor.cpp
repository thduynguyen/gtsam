/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testSemiringDiscreteFactor.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/semiring/SemiringDiscreteConditional.h>
#include <gtsam/semiring/SemiringDiscreteFactor.h>
#include <gtsam/semiring/SemiringGaussianFactor.h>
#include <gtsam/semiring/tests/TwoActionExample.h>

#include <cmath>

using namespace gtsam;

GTSAM_CONCEPT_TESTABLE_INST(SemiringDiscreteFactor)

/* ************************************************************************* */
namespace lifting {

using namespace two_action;

// A probability factor is lifted to (f, 0), so its value channel is zero.
TEST(SemiringDiscreteFactor, Probability) {
  const SemiringDiscreteFactor factor = transition();
  EXPECT((KeyVector{A.first, S.first}) == factor.keys());

  const auto [probability, value] = factor.evaluate(assignment(L, bad));
  EXPECT_DOUBLES_EQUAL(0.1, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0, value, 1e-9);
}

// A reward factor is lifted to (1, r), so its probability channel is one.
TEST(SemiringDiscreteFactor, Reward) {
  const SemiringDiscreteFactor factor = stateReward();
  EXPECT((KeyVector{S.first}) == factor.keys());

  const auto [probability, value] = factor.evaluate(assignment(R, good));
  EXPECT_DOUBLES_EQUAL(1.0, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(10.0, value, 1e-9);
}

// A probability and a value table combine to (p, p v), with the value
// recovered on evaluation.
TEST(SemiringDiscreteFactor, ProbabilityAndValue) {
  const SemiringDiscreteFactor factor(DecisionTreeFactor(A, "0.6 0.4"),
                                      DecisionTreeFactor(A, "8 2"));
  const auto [probability, value] = factor.evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.6, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(8.0, value, 1e-9);
  EXPECT_DOUBLES_EQUAL(4.8, factor.weightedValue()(assignment(L, good)), 1e-9);
  EXPECT(assert_equal(DecisionTreeFactor(A, "8 2"), factor.value()));
}

// The two channels can be given on different variables; the factor involves
// their union, in sorted order.
TEST(SemiringDiscreteFactor, UnionOfKeys) {
  const SemiringDiscreteFactor factor(DecisionTreeFactor(S, "0.3 0.7"),
                                      DecisionTreeFactor(A, "1 2"));
  EXPECT((KeyVector{A.first, S.first}) == factor.keys());
  EXPECT(factor.discreteKeys() == (A & S));

  const auto [probability, value] = factor.evaluate(assignment(R, good));
  EXPECT_DOUBLES_EQUAL(0.3, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(2.0, value, 1e-9);
}

// Tables that disagree on the cardinality of a variable are rejected.
TEST(SemiringDiscreteFactor, InconsistentCardinalities) {
  const DiscreteKey A3(A.first, 3);
  CHECK_EXCEPTION(SemiringDiscreteFactor(DecisionTreeFactor(A, "0.6 0.4"),
                                         DecisionTreeFactor(A3, "1 2 3")),
                  std::invalid_argument);
}

}  // namespace lifting
/* ************************************************************************* */
namespace operations {

using namespace two_action;

/// Check that two factors agree in both channels for every assignment.
bool agree(const SemiringDiscreteFactor& expected,
           const SemiringDiscreteFactor& actual, double tol = 1e-9) {
  for (const DiscreteValues& values : DiscreteValues::CartesianProduct(A & S)) {
    const auto e = expected.evaluate(values), a = actual.evaluate(values);
    if (std::abs(e.first - a.first) > tol) return false;
    if (std::abs(e.second - a.second) > tol) return false;
  }
  return true;
}

// The product multiplies probabilities and adds values.
TEST(SemiringDiscreteFactor, Product) {
  const SemiringDiscreteFactor actionFactor = policy() * actionReward();
  const SemiringDiscreteFactor stateFactor = transition() * stateReward();
  const SemiringDiscreteFactor product = actionFactor * stateFactor;
  EXPECT((KeyVector{A.first, S.first}) == product.keys());

  const auto [probability, value] = product.evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.6 * 0.9, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(-1.0 + 10.0, value, 1e-9);
}

// The product is commutative and associative.
TEST(SemiringDiscreteFactor, ProductLaws) {
  const SemiringDiscreteFactor f = policy() * actionReward();
  const SemiringDiscreteFactor g = transition(), h = stateReward();
  EXPECT(assert_equal(f * g, g * f));
  EXPECT(agree((f * g) * h, f * (g * h)));
}

// A variable with a single value cannot be a key of a table.
TEST(SemiringDiscreteFactor, SingleValuedKey) {
  const DiscreteKey single(7, 1);
  CHECK_EXCEPTION(
      SemiringDiscreteFactor(DecisionTreeFactor(single & A, "0.6 0.4")),
      std::invalid_argument);
}

// The virtual product agrees with the typed one.
TEST(SemiringDiscreteFactor, Multiply) {
  const SemiringDiscreteFactor expected = transition() * stateReward();
  const SemiringFactor::shared_ptr actual =
      transition().multiply(stateReward());
  EXPECT(actual->equals(expected));
}

// Summing out the next state leaves the expected reward for each action.
TEST(SemiringDiscreteFactor, Sum) {
  const SemiringDiscreteFactor joint = transition() * stateReward();
  const auto sum = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      joint.sum(Ordering(KeyVector{S.first})));
  CHECK(sum);
  EXPECT((KeyVector{A.first}) == sum->keys());

  const auto left = sum->evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(1.0, left.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(9.0, left.second, 1e-9);
  const auto right = sum->evaluate(assignment(R, good));
  EXPECT_DOUBLES_EQUAL(1.0, right.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(2.0, right.second, 1e-9);
}

// The product distributes over the sum, which is what makes variable
// elimination correct: a factor not involving the summed variable factors out.
TEST(SemiringDiscreteFactor, Distributivity) {
  const Ordering state(KeyVector{S.first});
  const SemiringDiscreteFactor f = policy() * actionReward();
  const SemiringDiscreteFactor g = transition() * stateReward();

  const auto sumOfProduct =
      std::dynamic_pointer_cast<SemiringDiscreteFactor>((f * g).sum(state));
  const auto sumOfG =
      std::dynamic_pointer_cast<SemiringDiscreteFactor>(g.sum(state));
  EXPECT(agree(f * *sumOfG, *sumOfProduct));
}

// Division by the sum yields the conditional probability and the surprise.
TEST(SemiringDiscreteFactor, Divide) {
  const SemiringDiscreteFactor joint = transition() * stateReward();
  const auto sum = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      joint.sum(Ordering(KeyVector{S.first})));
  const SemiringDiscreteFactor quotient = joint / *sum;

  const auto leftBad = quotient.evaluate(assignment(L, bad));
  EXPECT_DOUBLES_EQUAL(0.1, leftBad.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0 - 9.0, leftBad.second, 1e-9);
  const auto rightGood = quotient.evaluate(assignment(R, good));
  EXPECT_DOUBLES_EQUAL(0.2, rightGood.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(10.0 - 2.0, rightGood.second, 1e-9);

  // Multiplying back recovers the joint.
  EXPECT(agree(joint, quotient * *sum));
}

// The expectation of the full product is the expected total reward.
TEST(SemiringDiscreteFactor, Expectation) {
  const SemiringDiscreteFactor product =
      policy() * actionReward() * transition() * stateReward();
  EXPECT_DOUBLES_EQUAL(5.6, product.expectation(), 1e-9);

  // Unnormalized probabilities do not change the expectation.
  const SemiringDiscreteFactor scaled(DecisionTreeFactor(A, "3 3"));
  EXPECT_DOUBLES_EQUAL(5.6, (scaled * product).expectation(), 1e-9);
}

}  // namespace operations
/* ************************************************************************* */
namespace zero_probability {

using namespace two_action;

// Where the probability is zero the value is reported as zero, and
// elimination stays finite.
TEST(SemiringDiscreteFactor, ImpossibleOutcomes) {
  // Action R can never be taken, and L always leads to the good state.
  const SemiringDiscreteFactor never(DecisionTreeFactor(A, "1 0"));
  const SemiringDiscreteFactor certain(DecisionTreeFactor(A & S, "1 0 0.2 0.8"));
  const SemiringDiscreteFactor product =
      never * actionReward() * certain * stateReward();

  const auto impossible = product.evaluate(assignment(R, good));
  EXPECT_DOUBLES_EQUAL(0.0, impossible.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0, impossible.second, 1e-9);
  EXPECT_DOUBLES_EQUAL(9.0, product.expectation(), 1e-9);

  const auto [conditional, sum] =
      product.eliminate(Ordering(KeyVector{S.first}));
  const auto table =
      std::dynamic_pointer_cast<SemiringDiscreteConditional>(conditional);
  CHECK(table);
  for (const DiscreteValues& values : DiscreteValues::CartesianProduct(A & S)) {
    const auto [probability, surprise] = table->evaluate(values);
    EXPECT(std::isfinite(probability));
    EXPECT(std::isfinite(surprise));
  }
}

// The expectation is undefined when the total probability is zero.
TEST(SemiringDiscreteFactor, ZeroTotalProbability) {
  const SemiringDiscreteFactor zero(DecisionTreeFactor(A, "0 0"));
  CHECK_EXCEPTION((zero * actionReward()).expectation(), std::runtime_error);
}

}  // namespace zero_probability
/* ************************************************************************* */
namespace invalid_arguments {

using namespace two_action;

// Only variables of the factor can be summed out.
TEST(SemiringDiscreteFactor, SumOutMissingVariable) {
  CHECK_EXCEPTION(policy().sum(Ordering(KeyVector{S.first})),
                  std::invalid_argument);
}

// Discrete and Gaussian factors cannot be combined.
TEST(SemiringDiscreteFactor, MultiplyWithGaussian) {
  const SemiringGaussianFactor gaussian(std::make_shared<JacobianFactor>(
      5, I_1x1, Vector1(0.0), noiseModel::Unit::Create(1)));
  CHECK_EXCEPTION(policy().multiply(gaussian), std::invalid_argument);
}

}  // namespace invalid_arguments
/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
