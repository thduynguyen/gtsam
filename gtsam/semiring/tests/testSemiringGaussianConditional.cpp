/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testSemiringGaussianConditional.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/semiring/SemiringGaussianConditional.h>

using namespace gtsam;

GTSAM_CONCEPT_TESTABLE_INST(SemiringGaussianConditional)

/* ************************************************************************* */
namespace policy {

const Key S = 0, A = 1, U = 2;

/// A linear-Gaussian policy a = 2 s + 1 + e, with e ~ N(0, 0.5^2).
GaussianConditional::shared_ptr gaussianPolicy() {
  return GaussianConditional::sharedMeanAndStddev(A, 2.0 * I_1x1, S,
                                                  Vector1(1.0), 0.5);
}

/// The quadratic a^2 - 2 a s, a value on the action and another variable.
HessianFactor::shared_ptr quadratic(Key other) {
  return std::make_shared<HessianFactor>(A, other, 2.0 * I_1x1, -2.0 * I_1x1,
                                         Vector1(0.0), 0.0 * I_1x1,
                                         Vector1(0.0), 0.0);
}

/// Values for the action, the state and the extra variable.
VectorValues values(double action, double state, double extra = 0.0) {
  VectorValues result;
  result.insert(A, Vector1(action));
  result.insert(S, Vector1(state));
  result.insert(U, Vector1(extra));
  return result;
}

// The conditional takes its keys and frontals from the Gaussian conditional.
TEST(SemiringGaussianConditional, Constructor) {
  const SemiringGaussianConditional conditional(gaussianPolicy(), quadratic(S));
  EXPECT((KeyVector{A, S}) == conditional.keys());
  EXPECT_LONGS_EQUAL(1, conditional.nrFrontals());
  EXPECT_LONGS_EQUAL(1, conditional.nrParents());
  EXPECT(assert_equal(*gaussianPolicy(), *conditional.conditional()));
  EXPECT(assert_equal(*quadratic(S), *conditional.surprise()));
  EXPECT_DOUBLES_EQUAL(9.0 - 2.0 * 3.0 * 2.0,
                       conditional.surprise(values(3.0, 2.0)), 1e-9);
}

// Without a surprise the value channel is zero.
TEST(SemiringGaussianConditional, ZeroSurprise) {
  const SemiringGaussianConditional conditional(gaussianPolicy());
  EXPECT(!conditional.surprise());
  EXPECT_DOUBLES_EQUAL(0.0, conditional.surprise(values(3.0, 2.0)), 1e-9);
}

// Variables that appear only in the surprise become additional parents.
TEST(SemiringGaussianConditional, SurpriseOnlyParents) {
  const SemiringGaussianConditional conditional(gaussianPolicy(), quadratic(U));
  EXPECT((KeyVector{A, S, U}) == conditional.keys());
  EXPECT_LONGS_EQUAL(1, conditional.nrFrontals());
  EXPECT_LONGS_EQUAL(2, conditional.nrParents());
}

// As a factor, the conditional has the Gaussian conditional as its only
// probability factor and the surprise as its value.
TEST(SemiringGaussianConditional, Factor) {
  const SemiringGaussianConditional conditional(gaussianPolicy(), quadratic(S));
  const auto factor = std::dynamic_pointer_cast<SemiringGaussianFactor>(
      conditional.factor());
  CHECK(factor);
  EXPECT_LONGS_EQUAL(1, factor->gaussian().size());
  EXPECT_DOUBLES_EQUAL(conditional.surprise(values(3.0, 2.0)),
                       factor->value(values(3.0, 2.0)), 1e-9);
}

// Conditionals compare equal only if both channels match.
TEST(SemiringGaussianConditional, Equals) {
  const SemiringGaussianConditional conditional(gaussianPolicy(), quadratic(S));
  EXPECT(conditional.equals(
      SemiringGaussianConditional(gaussianPolicy(), quadratic(S))));
  EXPECT(!conditional.equals(SemiringGaussianConditional(gaussianPolicy())));
  EXPECT(!conditional.equals(
      SemiringGaussianConditional(gaussianPolicy(), quadratic(U))));
}

}  // namespace policy
/* ************************************************************************* */
namespace one_step {

const Key S = 0, A = 1;

/**
 * Eliminate the action from the policy a = 2 s + 1 + e, e ~ N(0, 0.5^2), and
 * the action value Q(s, a) = -(a - s)^2.
 */
SemiringFactor::EliminationResult eliminateAction() {
  const SemiringGaussianFactor policy(
      GaussianConditional::sharedMeanAndStddev(A, 2.0 * I_1x1, S, Vector1(1.0),
                                               0.5));
  const auto actionValue = SemiringGaussianFactor::Cost(HessianFactor(
      A, S, 2.0 * I_1x1, -2.0 * I_1x1, Vector1(0.0), 2.0 * I_1x1, Vector1(0.0),
      0.0));
  return policy.multiply(actionValue)->eliminate(Ordering(KeyVector{A}));
}

/// Values for the action and the state.
VectorValues values(double action, double state) {
  VectorValues result;
  result.insert(A, Vector1(action));
  result.insert(S, Vector1(state));
  return result;
}

// Eliminating the action gives the state value V(s) = E[Q(s, a)], here
// -((s + 1)^2 + 0.25), and the advantage A(s, a) = Q(s, a) - V(s).
TEST(SemiringGaussianConditional, PolicyAndAdvantage) {
  const auto [conditional, sum] = eliminateAction();
  const auto policy =
      std::dynamic_pointer_cast<SemiringGaussianConditional>(conditional);
  const auto stateValue =
      std::dynamic_pointer_cast<SemiringGaussianFactor>(sum);
  CHECK(policy);
  CHECK(stateValue);

  const double V = -((2.0 + 1.0) * (2.0 + 1.0) + 0.25);
  EXPECT_DOUBLES_EQUAL(V, stateValue->value(values(0.0, 2.0)), 1e-9);

  const double Q = -(4.0 - 2.0) * (4.0 - 2.0);
  EXPECT_DOUBLES_EQUAL(Q - V, policy->surprise(values(4.0, 2.0)), 1e-9);

  // The probability channel is the policy itself.
  const VectorValues mean = policy->conditional()->solve(values(0.0, 2.0));
  EXPECT(assert_equal(Vector1(5.0), mean.at(A), 1e-9));
}

// A conditional is normalized in the semiring sense: the advantage averages
// to zero under the policy, for every state.
TEST(SemiringGaussianConditional, Normalized) {
  const auto conditional = eliminateAction().first;
  const auto sum = std::dynamic_pointer_cast<SemiringGaussianFactor>(
      conditional->sum(Ordering(KeyVector{A})));
  CHECK(sum);
  EXPECT_DOUBLES_EQUAL(0.0, sum->value(values(0.0, 2.0)), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0, sum->value(values(0.0, -3.0)), 1e-9);
}

}  // namespace one_step
/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
