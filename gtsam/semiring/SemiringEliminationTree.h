/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringEliminationTree.h
 * @brief Elimination tree for semiring factor graphs
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/inference/EliminationTree.h>
#include <gtsam/semiring/SemiringBayesNet.h>
#include <gtsam/semiring/SemiringFactorGraph.h>

namespace gtsam {

/**
 * Elimination tree for a SemiringFactorGraph, used in sequential elimination.
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringEliminationTree
    : public EliminationTree<SemiringBayesNet, SemiringFactorGraph> {
 public:
  typedef EliminationTree<SemiringBayesNet, SemiringFactorGraph>
      Base;                                    ///< Base class
  typedef SemiringEliminationTree This;        ///< This class
  typedef std::shared_ptr<This> shared_ptr;    ///< Shared pointer to this class

  /**
   * Build the elimination tree of a factor graph using a precomputed column
   * structure, the set of factors involving each variable.
   */
  SemiringEliminationTree(const SemiringFactorGraph& factorGraph,
                          const VariableIndex& structure,
                          const Ordering& order);

  /**
   * Build the elimination tree of a factor graph. This computes the column
   * structure, so use the other constructor if it is already available.
   */
  SemiringEliminationTree(const SemiringFactorGraph& factorGraph,
                          const Ordering& order);

  /// Test whether the tree is equal to another
  bool equals(const This& other, double tol = 1e-9) const;
};

}  // namespace gtsam
