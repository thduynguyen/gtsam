/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringGaussianFactor.h
 * @brief Gaussian factor with a quadratic value, in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/semiring/SemiringFactor.h>

#include <string>

namespace gtsam {

/**
 * A semiring factor over continuous variables, in the log domain. The
 * probability channel is a product of Gaussian factors, kept as a
 * GaussianFactorGraph, and the value channel is a quadratic
 *
 *   v(x) = 0.5 x' G x - g' x + 0.5 f,
 *
 * stored as a HessianFactor whose error is the value. Unlike a Gaussian
 * factor, the quadratic may be indefinite, since rewards can have any sign; it
 * is never factorized, only added and substituted into.
 *
 * In the log domain the semiring product adds both channels, so it concatenates
 * the Gaussian factors and adds the quadratics. Eliminating variables x runs
 * ordinary Gaussian elimination on the probability channel, unchanged by the
 * values, to obtain p(x | S), and replaces the value by its expectation under
 * that conditional, which is again a quadratic in the separator S.
 *
 * Normalization constants of the Gaussian factors are not tracked, as
 * elsewhere in GTSAM; they do not affect expected values.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringGaussianFactor : public SemiringFactor {
 public:
  using This = SemiringGaussianFactor;
  using Base = SemiringFactor;
  using shared_ptr = std::shared_ptr<This>;

 private:
  GaussianFactorGraph gaussian_;     ///< probability channel
  HessianFactor::shared_ptr value_;  ///< value channel, nullptr if zero

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor for I/O
  SemiringGaussianFactor() {}

  /// Lift a Gaussian factor to (f, 0), e.g., linear dynamics or a policy.
  explicit SemiringGaussianFactor(const GaussianFactor::shared_ptr& gaussian);

  /**
   * Construct from a product of Gaussian factors and a quadratic value, which
   * can be nullptr to denote a zero value.
   */
  explicit SemiringGaussianFactor(const GaussianFactorGraph& gaussian,
                                  const HessianFactor::shared_ptr& value = {});

  /// A reward factor with value v(x) = quadratic.error(x) and no probability.
  static SemiringGaussianFactor Reward(const GaussianFactor& quadratic);

  /// A cost factor with value v(x) = -quadratic.error(x) and no probability.
  static SemiringGaussianFactor Cost(const GaussianFactor& quadratic);

  /// @}
  /// @name Testable
  /// @{

  /// Print with optional formatter.
  void print(
      const std::string& s = "SemiringGaussianFactor",
      const KeyFormatter& formatter = DefaultKeyFormatter) const override;

  /// Check equality up to tolerance.
  bool equals(const SemiringFactor& other, double tol = 1e-9) const override;

  /// @}
  /// @name Standard Interface
  /// @{

  /// The probability channel, a product of Gaussian factors.
  const GaussianFactorGraph& gaussian() const { return gaussian_; }

  /// The value channel as a quadratic, nullptr if the value is zero.
  const HessianFactor::shared_ptr& value() const { return value_; }

  using Base::error;

  /// The error of the probability channel, i.e., its negative log-density.
  double error(const VectorValues& x) const { return gaussian_.error(x); }

  /// Evaluate the value channel v(x).
  double value(const VectorValues& x) const;

  /// @}
  /// @name Semiring operations
  /// @{

  /// Semiring product with another factor, which must also be Gaussian.
  SemiringFactor::shared_ptr multiply(
      const SemiringFactor& other) const override;

  /**
   * Eliminate the frontal variables: Gaussian elimination on the probability
   * channel, and the expectation of the value under the resulting conditional.
   * Every frontal variable must appear in the probability channel.
   */
  EliminationResult eliminate(const Ordering& frontalKeys) const override;

  /// Expected value under the Gaussian density on all variables.
  double expectation() const override;

  /// @}
};

/// traits
template <>
struct traits<SemiringGaussianFactor>
    : public Testable<SemiringGaussianFactor> {};

}  // namespace gtsam
