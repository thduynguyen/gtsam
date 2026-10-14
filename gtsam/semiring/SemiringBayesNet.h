/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringBayesNet.h
 * @brief Bayes net of conditionals valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/inference/BayesNet.h>
#include <gtsam/inference/FactorGraph.h>
#include <gtsam/semiring/SemiringConditional.h>

namespace gtsam {

/**
 * A Bayes net of semiring conditionals, the result of sequential elimination
 * of a SemiringFactorGraph. Each conditional carries the surprise of its
 * frontal variables given its parents.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringBayesNet : public BayesNet<SemiringConditional> {
 public:
  typedef BayesNet<SemiringConditional> Base;
  typedef SemiringBayesNet This;
  typedef SemiringConditional ConditionalType;
  typedef std::shared_ptr<This> shared_ptr;
  typedef std::shared_ptr<ConditionalType> sharedConditional;

  /// @name Standard Constructors
  /// @{

  /// Construct empty Bayes net
  SemiringBayesNet() {}

  /// Construct from iterator over conditionals
  template <typename ITERATOR>
  SemiringBayesNet(ITERATOR firstConditional, ITERATOR lastConditional)
      : Base(firstConditional, lastConditional) {}

  /// Construct from container of conditionals
  template <class CONTAINER>
  explicit SemiringBayesNet(const CONTAINER& conditionals) {
    push_back(conditionals);
  }

  /// @}
  /// @name Testable
  /// @{

  /// Check equality up to tolerance.
  bool equals(const This& bn, double tol = 1e-9) const;

  /// @}
};

/// traits
template <>
struct traits<SemiringBayesNet> : public Testable<SemiringBayesNet> {};

}  // namespace gtsam
