/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringDiscreteFactor.h
 * @brief Table-backed factor valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteValues.h>
#include <gtsam/semiring/SemiringFactor.h>

#include <string>
#include <utility>

namespace gtsam {

/**
 * A semiring factor over discrete variables, stored as two tables: the
 * probability p(x) and the weighted value w(x) = p(x) v(x). The pair (p, w) is
 * the dual number p(x) (1 + v(x) eps), an element of the expectation semiring:
 *
 *  - product: (p1, w1) * (p2, w2) = (p1 p2, p1 w2 + p2 w1)
 *  - sum:     (p1, w1) + (p2, w2) = (p1 + p2, w1 + w2)
 *  - divide:  (p1, w1) / (p2, w2) = (p1 / p2, (w1 p2 - p1 w2) / p2^2)
 *
 * Storing the weighted value keeps all three operations exact where the
 * probability is zero, in which case the value is reported as zero.
 *
 * Both tables are kept on the same keys, sorted in increasing order.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringDiscreteFactor : public SemiringFactor {
 public:
  using This = SemiringDiscreteFactor;
  using Base = SemiringFactor;
  using shared_ptr = std::shared_ptr<This>;

 private:
  DecisionTreeFactor probability_;  ///< p(x)
  DecisionTreeFactor weighted_;     ///< p(x) v(x)

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor for I/O
  SemiringDiscreteFactor() {}

  /// Lift a probability factor f to (f, 0), e.g., a transition or a policy.
  explicit SemiringDiscreteFactor(const DecisionTreeFactor& probability);

  /// Construct from a probability table p and a value table v, as (p, p v).
  SemiringDiscreteFactor(const DecisionTreeFactor& probability,
                         const DecisionTreeFactor& value);

  /// Lift a reward table r to (1, r).
  static SemiringDiscreteFactor Reward(const DecisionTreeFactor& reward);

  /// Construct directly from the two stored tables, p and p v.
  static SemiringDiscreteFactor FromChannels(
      const DecisionTreeFactor& probability,
      const DecisionTreeFactor& weightedValue);

  /// @}
  /// @name Testable
  /// @{

  /// Print with optional formatter.
  void print(
      const std::string& s = "SemiringDiscreteFactor",
      const KeyFormatter& formatter = DefaultKeyFormatter) const override;

  /// Check equality up to tolerance.
  bool equals(const SemiringFactor& other, double tol = 1e-9) const override;

  /// @}
  /// @name Standard Interface
  /// @{

  /// The probability table p(x).
  const DecisionTreeFactor& probability() const { return probability_; }

  /// The weighted value table p(x) v(x).
  const DecisionTreeFactor& weightedValue() const { return weighted_; }

  /// The value table v(x), zero wherever p(x) is zero.
  DecisionTreeFactor value() const;

  /// The discrete keys, with cardinalities.
  DiscreteKeys discreteKeys() const { return probability_.discreteKeys(); }

  /// Evaluate both channels for an assignment, as the pair (p, v).
  std::pair<double, double> evaluate(const DiscreteValues& values) const;

  /// @}
  /// @name Semiring operations
  /// @{

  /// Semiring product with another discrete factor.
  This operator*(const This& other) const;

  /**
   * Semiring division by a factor on a subset of the keys, typically the sum
   * of this factor over its frontal variables.
   */
  This operator/(const This& other) const;

  /// Semiring product with another factor, which must also be discrete.
  SemiringFactor::shared_ptr multiply(
      const SemiringFactor& other) const override;

  /// Sum out the frontal variables and divide to obtain the conditional.
  EliminationResult eliminate(const Ordering& frontalKeys) const override;

  /// Expected value: total weighted value divided by total probability.
  double expectation() const override;

  /// @}

 private:
  /// Semiring sum over the frontal variables.
  This sumOut(const Ordering& frontalKeys) const;
};

/// traits
template <>
struct traits<SemiringDiscreteFactor>
    : public Testable<SemiringDiscreteFactor> {};

}  // namespace gtsam
