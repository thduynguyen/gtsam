/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringBayesTree.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/inference/BayesTree-inst.h>
#include <gtsam/inference/BayesTreeCliqueBase-inst.h>
#include <gtsam/semiring/SemiringBayesTree.h>

namespace gtsam {

// Instantiate base classes
template class BayesTreeCliqueBase<SemiringBayesTreeClique,
                                   SemiringFactorGraph>;
template class BayesTree<SemiringBayesTreeClique>;

/* ************************************************************************* */
bool SemiringBayesTree::equals(const This& other, double tol) const {
  return Base::equals(other, tol);
}

}  // namespace gtsam
