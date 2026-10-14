/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringConditional.h
 * @brief Base class for conditionals valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/inference/Conditional-inst.h>
#include <gtsam/semiring/SemiringFactor.h>

#include <string>

namespace gtsam {

/**
 * A semiring factor in conditional form, the result of dividing a product
 * factor by its semiring sum over the frontal variables.
 *
 * Its probability channel is the conditional p(frontals | parents), and its
 * value channel is the surprise v - E[v | parents]: how much the value changes
 * once the frontal variables are known. In a Markov decision process it is the
 * advantage when the frontal variable is an action, and the temporal-difference
 * residual when it is a state. The surprise averages to zero under the
 * conditional, so a conditional is normalized in the semiring sense: summing
 * out its frontal variables yields the semiring one, (1, 0).
 *
 * The conditional is itself a SemiringFactor, so it can be multiplied with
 * other factors and eliminated again, as done when computing marginals.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringConditional
    : public SemiringFactor,
      public Conditional<SemiringFactor, SemiringConditional> {
 public:
  using This = SemiringConditional;
  using shared_ptr = std::shared_ptr<This>;
  using BaseFactor = SemiringFactor;
  using BaseConditional = Conditional<SemiringFactor, SemiringConditional>;

 protected:
  /// Both channels of the conditional, stored as a factor.
  SemiringFactor::shared_ptr factor_;

  /// @name Standard Constructors
  /// @{

  /// Default constructor for I/O
  SemiringConditional() {}

  /**
   * Construct from the keys, ordered with the frontal variables first, the
   * number of frontal variables, and both channels stored as a factor.
   */
  SemiringConditional(const KeyVector& orderedKeys, size_t nrFrontals,
                      const SemiringFactor::shared_ptr& factor)
      : BaseFactor(orderedKeys), BaseConditional(nrFrontals), factor_(factor) {}

  /// @}

 public:
  /// @name Testable
  /// @{

  /// Print with optional formatter.
  void print(
      const std::string& s = "SemiringConditional",
      const KeyFormatter& formatter = DefaultKeyFormatter) const override;

  /// Check equality up to tolerance.
  bool equals(const SemiringFactor& other, double tol = 1e-9) const override;

  /// @}
  /// @name Standard Interface
  /// @{

  /// Both channels of the conditional, as a factor.
  const SemiringFactor::shared_ptr& factor() const { return factor_; }

  /// @}
  /// @name Semiring operations
  /// @{

  /// Semiring product of this conditional, as a factor, with another factor.
  SemiringFactor::shared_ptr multiply(
      const SemiringFactor& other) const override;

  using SemiringFactor::eliminate;

  /// Eliminate the frontal variables from this conditional, as a factor.
  EliminationResult eliminate(const Ordering& frontalKeys,
                              const SemiringSum& sum) const override;

  /// Expected value of this conditional, as a factor.
  double expectation() const override;

  /// @}
};

/// traits
template <>
struct traits<SemiringConditional> : public Testable<SemiringConditional> {};

}  // namespace gtsam
