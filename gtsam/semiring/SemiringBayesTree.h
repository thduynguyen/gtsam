/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringBayesTree.h
 * @brief Bayes tree of conditionals valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/inference/BayesTree.h>
#include <gtsam/inference/BayesTreeCliqueBase.h>
#include <gtsam/semiring/SemiringBayesNet.h>
#include <gtsam/semiring/SemiringFactorGraph.h>

#include <string>

namespace gtsam {

/* ************************************************************************* */
/// A clique in a SemiringBayesTree
class GTSAM_EXPORT SemiringBayesTreeClique
    : public BayesTreeCliqueBase<SemiringBayesTreeClique, SemiringFactorGraph> {
 public:
  typedef SemiringBayesTreeClique This;
  typedef BayesTreeCliqueBase<SemiringBayesTreeClique, SemiringFactorGraph>
      Base;
  typedef std::shared_ptr<This> shared_ptr;
  typedef std::weak_ptr<This> weak_ptr;
  SemiringBayesTreeClique() {}
  SemiringBayesTreeClique(
      const std::shared_ptr<SemiringConditional>& conditional)
      : Base(conditional) {}
};

/* ************************************************************************* */
/**
 * A Bayes tree of semiring conditionals, the result of multifrontal
 * elimination of a SemiringFactorGraph.
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringBayesTree
    : public BayesTree<SemiringBayesTreeClique> {
 private:
  typedef BayesTree<SemiringBayesTreeClique> Base;

 public:
  typedef SemiringBayesTree This;
  typedef std::shared_ptr<This> shared_ptr;

  /// Default constructor, creates an empty Bayes tree
  SemiringBayesTree() {}

  /// Check equality up to tolerance.
  bool equals(const This& other, double tol = 1e-9) const;
};

/// traits
template <>
struct traits<SemiringBayesTreeClique>
    : public Testable<SemiringBayesTreeClique> {};
template <>
struct traits<SemiringBayesTree> : public Testable<SemiringBayesTree> {};

}  // namespace gtsam
