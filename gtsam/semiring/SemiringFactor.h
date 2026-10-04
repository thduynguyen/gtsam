/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringFactor.h
 * @brief Base class for factors valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/base/Testable.h>
#include <gtsam/inference/Factor.h>
#include <gtsam/inference/Ordering.h>

#include <gtsam/semiring/SemiringSum.h>

#include <memory>
#include <utility>

namespace gtsam {

class SemiringConditional;

/**
 * Abstract base class for factors whose values live in the expectation
 * semiring, so that variable elimination computes expected additive values
 * (e.g., expected total reward) rather than only probabilities.
 *
 * Every factor carries two channels over its variables x:
 *  - a probability channel p(x), an ordinary (unnormalized) potential, and
 *  - a value channel v(x), the value accumulated so far given x.
 *
 * Together they represent the dual number p(x) (1 + v(x) eps), eps^2 = 0. The
 * semiring product multiplies probabilities and adds values, and the semiring
 * sum over a variable adds probabilities and averages values under the
 * resulting conditional. A probability factor f is lifted as (f, 0) and a
 * reward factor r as (1, r).
 *
 * Derived classes implement the three operations variable elimination needs:
 * the product (multiply), the sum over frontal variables, and the division that
 * yields a conditional (the latter two combined in eliminate).
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringFactor : public Factor {
 public:
  using This = SemiringFactor;
  using Base = Factor;
  using shared_ptr = std::shared_ptr<This>;

  /// A conditional on the frontal variables and a factor on the separator.
  using EliminationResult =
      std::pair<std::shared_ptr<SemiringConditional>, shared_ptr>;

 protected:
  /// @name Standard Constructors
  /// @{

  /// Default constructor for I/O
  SemiringFactor() {}

  /// Construct from a container of keys.
  template <class CONTAINER>
  explicit SemiringFactor(const CONTAINER& keys) : Base(keys) {}

  /// @}

 public:
  /// @name Testable
  /// @{

  /// Check equality up to tolerance.
  virtual bool equals(const SemiringFactor& other, double tol = 1e-9) const = 0;

  /// @}
  /// @name Semiring operations
  /// @{

  /**
   * Semiring product: probabilities multiply and values add. The result
   * involves the union of the keys of both factors.
   */
  virtual shared_ptr multiply(const SemiringFactor& other) const = 0;

  /**
   * Eliminate the frontal variables with the given rule. Returns the
   * conditional on the frontal variables given the separator and the semiring
   * sum over the frontal variables, a new factor on the separator.
   *
   * The value channel of the new factor is the values of the frontal outcomes
   * merged by the rule: their average E[v | separator], their maximum, or
   * their tilted mean. The value channel of the conditional is v minus that
   * merged value: the surprise, the regret, or the soft advantage.
   */
  virtual EliminationResult eliminate(const Ordering& frontalKeys,
                                      const SemiringSum& sum) const = 0;

  /// Eliminate the frontal variables by averaging, the expectation semiring.
  EliminationResult eliminate(const Ordering& frontalKeys) const {
    return eliminate(frontalKeys, SemiringSum());
  }

  /// Semiring sum over the frontal variables, a new factor on the separator.
  shared_ptr sum(const Ordering& frontalKeys,
                 const SemiringSum& sum = SemiringSum()) const;

  /**
   * Expected value E[v] under the normalized probability channel, obtained by
   * summing out all variables.
   */
  virtual double expectation() const = 0;

  /// @}
};

/// traits
template <>
struct traits<SemiringFactor> : public Testable<SemiringFactor> {};

}  // namespace gtsam
