/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testSemiringGaussianFactor.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/semiring/SemiringDiscreteFactor.h>
#include <gtsam/semiring/SemiringGaussianConditional.h>
#include <gtsam/semiring/SemiringGaussianFactor.h>

using namespace gtsam;

GTSAM_CONCEPT_TESTABLE_INST(SemiringGaussianFactor)

namespace {

/// A value for a single scalar variable.
VectorValues scalar(Key key, double value) {
  VectorValues values;
  values.insert(key, Vector1(value));
  return values;
}

/// Cast an elimination result to its Gaussian types.
struct GaussianResult {
  SemiringGaussianConditional::shared_ptr conditional;
  SemiringGaussianFactor::shared_ptr factor;

  explicit GaussianResult(const SemiringFactor::EliminationResult& result)
      : conditional(std::dynamic_pointer_cast<SemiringGaussianConditional>(
            result.first)),
        factor(std::dynamic_pointer_cast<SemiringGaussianFactor>(
            result.second)) {}
};

}  // namespace

/* ************************************************************************* */
namespace scalar_prior {

const Key X = 0;

/// The density x ~ N(1, 2^2).
GaussianFactor::shared_ptr prior() {
  return std::make_shared<JacobianFactor>(X, I_1x1, Vector1(1.0),
                                          noiseModel::Isotropic::Sigma(1, 2.0));
}

/// The quadratic x^2 - 3 x + 2.
HessianFactor quadratic() {
  return HessianFactor(X, 2.0 * I_1x1, Vector1(3.0), 4.0);
}

// A Gaussian factor is lifted to (f, 0), so its value channel is zero.
TEST(SemiringGaussianFactor, Probability) {
  const SemiringGaussianFactor factor(prior());
  EXPECT((KeyVector{X}) == factor.keys());
  EXPECT_LONGS_EQUAL(1, factor.gaussian().size());
  EXPECT(!factor.value());
  EXPECT_DOUBLES_EQUAL(0.0, factor.value(scalar(X, 3.0)), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.5, factor.error(scalar(X, 3.0)), 1e-9);
}

// A reward factor has the quadratic as its value and no probability; a cost
// factor has the negated quadratic.
TEST(SemiringGaussianFactor, RewardAndCost) {
  const auto reward = SemiringGaussianFactor::Reward(quadratic());
  EXPECT((KeyVector{X}) == reward.keys());
  EXPECT(reward.gaussian().empty());
  EXPECT_DOUBLES_EQUAL(9.0 - 9.0 + 2.0, reward.value(scalar(X, 3.0)), 1e-9);

  const auto cost = SemiringGaussianFactor::Cost(quadratic());
  EXPECT_DOUBLES_EQUAL(-2.0, cost.value(scalar(X, 3.0)), 1e-9);
}

// A Jacobian factor can also define the quadratic, through its error.
TEST(SemiringGaussianFactor, RewardFromJacobian) {
  const JacobianFactor residual(X, I_1x1, Vector1(1.0),
                                noiseModel::Isotropic::Sigma(1, 0.5));
  const auto cost = SemiringGaussianFactor::Cost(residual);
  EXPECT_DOUBLES_EQUAL(-0.5 * 16.0, cost.value(scalar(X, 3.0)), 1e-9);
}

// In the log domain the product collects the Gaussian factors and adds the
// quadratic values.
TEST(SemiringGaussianFactor, Multiply) {
  const SemiringGaussianFactor density(prior());
  const auto reward = SemiringGaussianFactor::Reward(quadratic());
  const auto product = std::dynamic_pointer_cast<SemiringGaussianFactor>(
      density.multiply(reward)->multiply(reward)->multiply(density));
  CHECK(product);
  EXPECT_LONGS_EQUAL(2, product->gaussian().size());
  EXPECT_DOUBLES_EQUAL(2 * 0.5, product->error(scalar(X, 3.0)), 1e-9);
  EXPECT_DOUBLES_EQUAL(2 * 2.0, product->value(scalar(X, 3.0)), 1e-9);
}

// The expectation of x^2 - 3 x + 2 under N(1, 4) is (1 + 4) - 3 + 2.
TEST(SemiringGaussianFactor, Expectation) {
  const SemiringGaussianFactor density(prior());
  const auto product = density.multiply(
      SemiringGaussianFactor::Reward(quadratic()));
  EXPECT_DOUBLES_EQUAL(4.0, product->expectation(), 1e-9);
  EXPECT_DOUBLES_EQUAL(0.0, density.expectation(), 1e-9);
}

// Eliminating the only variable leaves a constant holding the expectation,
// and a conditional holding the density with the surprise v - E[v].
TEST(SemiringGaussianFactor, Eliminate) {
  const auto product = SemiringGaussianFactor(prior()).multiply(
      SemiringGaussianFactor::Reward(quadratic()));
  const GaussianResult result(product->eliminate(Ordering(KeyVector{X})));
  CHECK(result.conditional);
  CHECK(result.factor);

  EXPECT(result.factor->empty());
  EXPECT_DOUBLES_EQUAL(4.0, result.factor->expectation(), 1e-9);

  EXPECT_LONGS_EQUAL(1, result.conditional->nrFrontals());
  const VectorValues mean = result.conditional->conditional()->solve(VectorValues());
  EXPECT(assert_equal(Vector1(1.0), mean.at(X), 1e-9));
  EXPECT_DOUBLES_EQUAL(2.0 - 4.0,
                       result.conditional->surprise(scalar(X, 3.0)), 1e-9);
}

// Factors compare equal only if both channels match.
TEST(SemiringGaussianFactor, Equals) {
  const SemiringGaussianFactor density(prior());
  const auto reward = SemiringGaussianFactor::Reward(quadratic());
  EXPECT(density.equals(SemiringGaussianFactor(prior())));
  EXPECT(reward.equals(SemiringGaussianFactor::Reward(quadratic())));
  EXPECT(!density.equals(reward));
  EXPECT(!density.equals(*density.multiply(reward)));
  EXPECT(!reward.equals(SemiringGaussianFactor::Cost(quadratic())));
}

}  // namespace scalar_prior
/* ************************************************************************* */
namespace dynamics {

const Key S = 0, N = 1, U = 2;

/// Linear dynamics n = 0.5 s + w, with the given noise on w.
SemiringGaussianFactor transition(const SharedDiagonal& noise) {
  return SemiringGaussianFactor(std::make_shared<JacobianFactor>(
      N, I_1x1, S, -0.5 * I_1x1, Vector1(0.0), noise));
}

/// The cost n^2 on the next state.
SemiringGaussianFactor nextStateCost() {
  return SemiringGaussianFactor::Cost(
      HessianFactor(N, 2.0 * I_1x1, Vector1(0.0), 0.0));
}

// Eliminating the next state takes the expectation of its value under the
// dynamics: E[-n^2 | s] = -(0.25 s^2 + 0.09), a quadratic value function.
TEST(SemiringGaussianFactor, ExpectedValueGivenState) {
  const auto bucket =
      transition(noiseModel::Isotropic::Sigma(1, 0.3)).multiply(nextStateCost());
  const GaussianResult result(bucket->eliminate(Ordering(KeyVector{N})));
  CHECK(result.factor);
  EXPECT((KeyVector{S}) == result.factor->keys());
  EXPECT_DOUBLES_EQUAL(-(0.25 * 4.0 + 0.09),
                       result.factor->value(scalar(S, 2.0)), 1e-9);
  EXPECT_DOUBLES_EQUAL(-0.09, result.factor->value(scalar(S, 0.0)), 1e-9);

  // The dynamics carry no information about the current state.
  EXPECT_DOUBLES_EQUAL(0.0, result.factor->error(scalar(S, 2.0)), 1e-9);
}

// The surprise is the value minus its expectation given the parents.
TEST(SemiringGaussianFactor, Surprise) {
  const auto bucket =
      transition(noiseModel::Isotropic::Sigma(1, 0.3)).multiply(nextStateCost());
  const GaussianResult result(bucket->eliminate(Ordering(KeyVector{N})));
  CHECK(result.conditional);
  EXPECT((KeyVector{N, S}) == result.conditional->keys());

  VectorValues values = scalar(S, 2.0);
  values.insert(N, Vector1(1.5));
  EXPECT_DOUBLES_EQUAL(-1.5 * 1.5 + (0.25 * 4.0 + 0.09),
                       result.conditional->surprise(values), 1e-9);
}

// With deterministic dynamics, given as a constrained noise model, the next
// state is substituted exactly and no noise term appears.
TEST(SemiringGaussianFactor, DeterministicDynamics) {
  const auto bucket =
      transition(noiseModel::Constrained::All(1)).multiply(nextStateCost());
  const GaussianResult result(bucket->eliminate(Ordering(KeyVector{N})));
  CHECK(result.factor);
  EXPECT_DOUBLES_EQUAL(-0.25 * 4.0, result.factor->value(scalar(S, 2.0)),
                       1e-9);
}

// A value can involve variables without any Gaussian factor, as long as they
// are not eliminated: here E[-(n - u)^2 | s, u] = -((0.5 s - u)^2 + 0.09).
TEST(SemiringGaussianFactor, ValueOnlyVariable) {
  const auto tracking = SemiringGaussianFactor::Cost(HessianFactor(
      N, U, 2.0 * I_1x1, -2.0 * I_1x1, Vector1(0.0), 2.0 * I_1x1, Vector1(0.0),
      0.0));
  const auto bucket =
      transition(noiseModel::Isotropic::Sigma(1, 0.3)).multiply(tracking);
  const GaussianResult result(bucket->eliminate(Ordering(KeyVector{N})));
  CHECK(result.factor);
  EXPECT((KeyVector{S, U}) == result.factor->keys());
  EXPECT((KeyVector{N, S, U}) == result.conditional->keys());

  VectorValues values = scalar(S, 2.0);
  values.insert(U, Vector1(3.0));
  EXPECT_DOUBLES_EQUAL(-(4.0 + 0.09), result.factor->value(values), 1e-9);
}

// Expectations over a variable need a density on it.
TEST(SemiringGaussianFactor, EliminateWithoutDensity) {
  CHECK_EXCEPTION(nextStateCost().eliminate(Ordering(KeyVector{N})),
                  std::invalid_argument);
  CHECK_EXCEPTION(nextStateCost().expectation(), std::invalid_argument);
}

// Gaussian and discrete factors cannot be combined.
TEST(SemiringGaussianFactor, MultiplyWithDiscrete) {
  const SemiringDiscreteFactor discrete(
      DecisionTreeFactor(DiscreteKey(5, 2), "0.6 0.4"));
  CHECK_EXCEPTION(nextStateCost().multiply(discrete), std::invalid_argument);
}

}  // namespace dynamics
/* ************************************************************************* */
namespace multivariate {

const Key X = 0;

// The expectation of an indefinite quadratic under a correlated Gaussian
// matches the closed form 0.5 (mu' G mu + tr(G Sigma)) - g' mu + 0.5 f.
TEST(SemiringGaussianFactor, IndefiniteQuadratic) {
  // A x = b + e gives mean inv(A) b and covariance inv(A) inv(A)'.
  const Matrix2 A{{2.0, 1.0}, {0.0, 1.0}};
  const Vector2 b{3.0, 1.0};
  const SemiringGaussianFactor density(std::make_shared<JacobianFactor>(
      X, A, b, noiseModel::Unit::Create(2)));

  const Matrix2 G{{1.0, 2.0}, {2.0, -3.0}};
  const Vector2 g{1.0, -1.0};
  const double f = 0.5;
  const auto reward =
      SemiringGaussianFactor::Reward(HessianFactor(X, G, g, f));

  const Matrix2 Ainv = A.inverse();
  const Vector2 mu = Ainv * b;
  const Matrix2 Sigma = Ainv * Ainv.transpose();
  const double expected =
      0.5 * (mu.dot(G * mu) + (G * Sigma).trace()) - g.dot(mu) + 0.5 * f;
  EXPECT_DOUBLES_EQUAL(-1.0, expected, 1e-9);
  EXPECT_DOUBLES_EQUAL(expected, density.multiply(reward)->expectation(),
                       1e-9);
}

}  // namespace multivariate
/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
