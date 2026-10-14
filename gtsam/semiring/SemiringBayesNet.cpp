/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringBayesNet.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/inference/FactorGraph-inst.h>
#include <gtsam/semiring/SemiringBayesNet.h>

namespace gtsam {

// Instantiate base class
template class FactorGraph<SemiringConditional>;

/* ************************************************************************* */
bool SemiringBayesNet::equals(const This& bn, double tol) const {
  return Base::equals(bn, tol);
}

}  // namespace gtsam
