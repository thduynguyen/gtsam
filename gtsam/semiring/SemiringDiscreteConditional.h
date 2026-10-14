/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringDiscreteConditional.h
 * @brief Table-backed conditional valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/semiring/SemiringConditional.h>
#include <gtsam/semiring/SemiringDiscreteFactor.h>

#include <utility>

namespace gtsam {

/**
 * A semiring conditional over discrete variables: the pair
 * (p(x | S), p(x | S) D(x, S)), where D = v - E[v | S] is the surprise of the
 * frontal assignment x relative to the average for the separator assignment S.
 *
 * When x is an action and S a state, p is the policy and D is the advantage
 * function Q(s, a) - V(s). When x is a next state, p is the dynamics and D is
 * the temporal-difference residual.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringDiscreteConditional : public SemiringConditional {
 public:
  using This = SemiringDiscreteConditional;
  using Base = SemiringConditional;
  using shared_ptr = std::shared_ptr<This>;

  /// @name Standard Constructors
  /// @{

  /// Default constructor for I/O
  SemiringDiscreteConditional() {}

  /**
   * Construct by semiring division of a joint factor by its marginal on the
   * separator, i.e., its semiring sum over the frontal variables.
   */
  SemiringDiscreteConditional(const SemiringDiscreteFactor& joint,
                              const SemiringDiscreteFactor& marginal,
                              const Ordering& frontalKeys);

  /// @}
  /// @name Standard Interface
  /// @{

  /// Both channels of the conditional, as a discrete semiring factor.
  const SemiringDiscreteFactor& table() const;

  /// The probability channel as a conditional p(frontals | parents).
  DiscreteConditional probability() const;

  /// The surprise table v - E[v | parents].
  DecisionTreeFactor surprise() const { return table().value(); }

  /**
   * The greedy choice: for every parent assignment, probability one on the
   * frontal assignment with the largest surprise, shared equally among ties.
   * When the frontal variable is an action, this is the greedy policy.
   */
  DiscreteConditional greedy() const;

  /**
   * The conditional reweighted toward high surprise,
   *   q(x | S) proportional to p(x | S) exp(tilt * surprise(x, S)).
   * When the frontal variable is an action and tilt = 1 / temperature, this is
   * the soft, or softmax, policy. After elimination with the tilted rule of
   * the same tilt it is already normalized.
   */
  DiscreteConditional tilted(double tilt) const;

  using Base::evaluate;

  /// Evaluate both channels for an assignment, as the pair (p, surprise).
  std::pair<double, double> evaluate(const DiscreteValues& values) const {
    return table().evaluate(values);
  }

  /// @}

 private:
  /// Weights on the keys of this conditional, normalized over the frontals.
  DiscreteConditional normalized(const DecisionTreeFactor& weights) const;
};

/// traits
template <>
struct traits<SemiringDiscreteConditional>
    : public Testable<SemiringDiscreteConditional> {};

}  // namespace gtsam
