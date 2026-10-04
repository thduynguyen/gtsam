/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringEliminationTree.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/inference/EliminationTree-inst.h>
#include <gtsam/semiring/SemiringEliminationTree.h>

namespace gtsam {

// Instantiate base class
template class EliminationTree<SemiringBayesNet, SemiringFactorGraph>;

/* ************************************************************************* */
SemiringEliminationTree::SemiringEliminationTree(
    const SemiringFactorGraph& factorGraph, const VariableIndex& structure,
    const Ordering& order)
    : Base(factorGraph, structure, order) {}

/* ************************************************************************* */
SemiringEliminationTree::SemiringEliminationTree(
    const SemiringFactorGraph& factorGraph, const Ordering& order)
    : Base(factorGraph, order) {}

/* ************************************************************************* */
bool SemiringEliminationTree::equals(const This& other, double tol) const {
  return Base::equals(other, tol);
}

}  // namespace gtsam
