/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringJunctionTree.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/inference/JunctionTree-inst.h>
#include <gtsam/semiring/SemiringEliminationTree.h>
#include <gtsam/semiring/SemiringJunctionTree.h>

namespace gtsam {

// Instantiate base classes
template class EliminatableClusterTree<SemiringBayesTree, SemiringFactorGraph>;
template class JunctionTree<SemiringBayesTree, SemiringFactorGraph>;

/* ************************************************************************* */
SemiringJunctionTree::SemiringJunctionTree(
    const SemiringEliminationTree& eliminationTree)
    : Base(eliminationTree) {}

}  // namespace gtsam
