/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testSemiringDiscreteConditional.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/semiring/SemiringDiscreteConditional.h>
#include <gtsam/semiring/tests/TwoActionExample.h>

#include <cmath>

using namespace gtsam;

GTSAM_CONCEPT_TESTABLE_INST(SemiringDiscreteConditional)

/* ************************************************************************* */
namespace state_given_action {

using namespace two_action;

/// Eliminate the next state from the transition and its reward.
SemiringDiscreteConditional::shared_ptr eliminateState() {
  const SemiringDiscreteFactor joint = transition() * stateReward();
  return std::dynamic_pointer_cast<SemiringDiscreteConditional>(
      joint.eliminate(Ordering(KeyVector{S.first})).first);
}

// The conditional lists the frontal variable first, then its parents.
TEST(SemiringDiscreteConditional, Keys) {
  const auto conditional = eliminateState();
  CHECK(conditional);
  EXPECT((KeyVector{S.first, A.first}) == conditional->keys());
  EXPECT_LONGS_EQUAL(1, conditional->nrFrontals());
  EXPECT_LONGS_EQUAL(1, conditional->nrParents());
  EXPECT_LONGS_EQUAL(S.first, conditional->firstFrontalKey());
}

// The probability channel is the ordinary conditional p(state | action).
TEST(SemiringDiscreteConditional, Probability) {
  const auto conditional = eliminateState();
  const DiscreteConditional expected(S, {A}, "9/1 2/8");
  EXPECT(assert_equal(expected, conditional->probability()));
}

// The value channel is the surprise of each outcome relative to the expected
// reward of its action, which is 9 for L and 2 for R.
TEST(SemiringDiscreteConditional, Surprise) {
  const auto conditional = eliminateState();
  const DecisionTreeFactor surprise = conditional->surprise();
  EXPECT_DOUBLES_EQUAL(10.0 - 9.0, surprise(assignment(L, good)), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0 - 9.0, surprise(assignment(L, bad)), 1e-9);
  EXPECT_DOUBLES_EQUAL(10.0 - 2.0, surprise(assignment(R, good)), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0 - 2.0, surprise(assignment(R, bad)), 1e-9);

  const auto [probability, value] = conditional->evaluate(assignment(R, bad));
  EXPECT_DOUBLES_EQUAL(0.8, probability, 1e-9);
  EXPECT_DOUBLES_EQUAL(-2.0, value, 1e-9);
}

// A conditional is normalized in the semiring sense: summing out the frontal
// variable gives probability one and, as surprises average out, value zero.
TEST(SemiringDiscreteConditional, Normalized) {
  const auto conditional = eliminateState();
  const auto sum = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      conditional->sum(Ordering(KeyVector{S.first})));
  CHECK(sum);
  for (size_t action : {L, R}) {
    const auto [probability, value] = sum->evaluate(assignment(action, good));
    EXPECT_DOUBLES_EQUAL(1.0, probability, 1e-9);
    EXPECT_DOUBLES_EQUAL(0.0, value, 1e-9);
  }
  EXPECT_DOUBLES_EQUAL(0.0, conditional->expectation(), 1e-9);
}

// Conditionals compare equal only if both channels and the frontals match.
TEST(SemiringDiscreteConditional, Equals) {
  const auto conditional = eliminateState();
  EXPECT(conditional->equals(*eliminateState()));

  const SemiringDiscreteFactor joint = transition() * stateReward();
  const SemiringDiscreteFactor other = transition();
  const auto withoutReward = other.eliminate(Ordering(KeyVector{S.first})).first;
  EXPECT(!conditional->equals(*withoutReward));
  EXPECT(!conditional->equals(joint));
}

}  // namespace state_given_action
/* ************************************************************************* */
namespace action {

using namespace two_action;

/// The action values Q, with the policy: the bucket of the action variable.
SemiringDiscreteFactor actionBucket() {
  const SemiringDiscreteFactor joint = transition() * stateReward();
  const auto stateValue = joint.sum(Ordering(KeyVector{S.first}));
  return policy() * actionReward() *
         *std::dynamic_pointer_cast<SemiringDiscreteFactor>(stateValue);
}

// Eliminating the action from the policy and the action values gives the
// policy with the advantage function A = Q - V, where V = 5.6.
TEST(SemiringDiscreteConditional, PolicyAndAdvantage) {
  const auto [conditional, sum] =
      actionBucket().eliminate(Ordering(KeyVector{A.first}));
  const auto table =
      std::dynamic_pointer_cast<SemiringDiscreteConditional>(conditional);
  CHECK(table);
  EXPECT_LONGS_EQUAL(0, table->nrParents());

  const auto left = table->evaluate(assignment(L, good));
  EXPECT_DOUBLES_EQUAL(0.6, left.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(8.0 - 5.6, left.second, 1e-9);
  const auto right = table->evaluate(assignment(R, good));
  EXPECT_DOUBLES_EQUAL(0.4, right.first, 1e-9);
  EXPECT_DOUBLES_EQUAL(2.0 - 5.6, right.second, 1e-9);

  EXPECT(sum->empty());
  EXPECT_DOUBLES_EQUAL(5.6, sum->expectation(), 1e-9);
}

// The weighted advantage p(a) A(a) is the gradient of the expected reward with
// respect to the logits of a softmax policy.
TEST(SemiringDiscreteConditional, SoftmaxPolicyGradient) {
  const auto conditional = std::dynamic_pointer_cast<
      SemiringDiscreteConditional>(
      actionBucket().eliminate(Ordering(KeyVector{A.first})).first);
  const DecisionTreeFactor& gradient = conditional->table().weightedValue();
  EXPECT_DOUBLES_EQUAL(1.44, gradient(assignment(L, good)), 1e-9);
  EXPECT_DOUBLES_EQUAL(-1.44, gradient(assignment(R, good)), 1e-9);
}

// Multiplying the conditional with the separator factor recovers the bucket,
// so a conditional can be used wherever a factor is expected.
TEST(SemiringDiscreteConditional, MultiplyRecoversJoint) {
  const SemiringDiscreteFactor bucket = actionBucket();
  const auto [conditional, sum] = bucket.eliminate(Ordering(KeyVector{A.first}));
  const auto product = std::dynamic_pointer_cast<SemiringDiscreteFactor>(
      conditional->multiply(*sum));
  CHECK(product);
  for (size_t action : {L, R}) {
    const auto expected = bucket.evaluate(assignment(action, good));
    const auto actual = product->evaluate(assignment(action, good));
    EXPECT_DOUBLES_EQUAL(expected.first, actual.first, 1e-9);
    EXPECT_DOUBLES_EQUAL(expected.second, actual.second, 1e-9);
  }
}

}  // namespace action
/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
