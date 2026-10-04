/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringGaussianConditional.h
 * @brief Gaussian conditional with a quadratic surprise
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/semiring/SemiringConditional.h>
#include <gtsam/semiring/SemiringGaussianFactor.h>

namespace gtsam {

/**
 * A semiring conditional over continuous variables: a Gaussian conditional
 * p(x | S) together with the quadratic surprise D(x, S) = v - E[v | S], whose
 * expectation under the conditional is zero for every S.
 *
 * When x is an action and S a state, p is a linear-Gaussian policy and D is
 * the advantage function Q(s, a) - V(s). When x is a next state, p is the
 * dynamics and D is the temporal-difference residual.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringGaussianConditional : public SemiringConditional {
 public:
  using This = SemiringGaussianConditional;
  using Base = SemiringConditional;
  using shared_ptr = std::shared_ptr<This>;

 private:
  GaussianConditional::shared_ptr conditional_;  ///< p(frontals | parents)

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor for I/O
  SemiringGaussianConditional() {}

  /**
   * Construct from a Gaussian conditional and a quadratic surprise, which can
   * be nullptr to denote zero. Any variables that appear only in the surprise
   * become additional parents.
   */
  explicit SemiringGaussianConditional(
      const GaussianConditional::shared_ptr& conditional,
      const HessianFactor::shared_ptr& surprise = {});

  /// @}
  /// @name Standard Interface
  /// @{

  /// The probability channel, the Gaussian conditional p(frontals | parents).
  const GaussianConditional::shared_ptr& conditional() const {
    return conditional_;
  }

  /// The surprise v - E[v | parents] as a quadratic, nullptr if it is zero.
  const HessianFactor::shared_ptr& surprise() const;

  /// Evaluate the surprise at x, which holds the frontals and the parents.
  double surprise(const VectorValues& x) const;

  /// @}
};

/// traits
template <>
struct traits<SemiringGaussianConditional>
    : public Testable<SemiringGaussianConditional> {};

}  // namespace gtsam
