/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringJunctionTree.h
 * @brief Junction tree for semiring factor graphs
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/inference/JunctionTree.h>
#include <gtsam/semiring/SemiringBayesTree.h>
#include <gtsam/semiring/SemiringFactorGraph.h>

namespace gtsam {

// Forward declarations
class SemiringEliminationTree;

/**
 * Junction tree for a SemiringFactorGraph, the intermediate data structure of
 * multifrontal elimination: each cluster holds factors and eliminates several
 * variables at once.
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringJunctionTree
    : public JunctionTree<SemiringBayesTree, SemiringFactorGraph> {
 public:
  typedef JunctionTree<SemiringBayesTree, SemiringFactorGraph>
      Base;                                  ///< Base class
  typedef SemiringJunctionTree This;         ///< This class
  typedef std::shared_ptr<This> shared_ptr;  ///< Shared pointer to this class

  /// Build the junction tree from an elimination tree.
  SemiringJunctionTree(const SemiringEliminationTree& eliminationTree);
};

}  // namespace gtsam
