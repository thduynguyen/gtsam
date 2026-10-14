/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringFactor.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/semiring/SemiringConditional.h>
#include <gtsam/semiring/SemiringFactor.h>

namespace gtsam {

/* ************************************************************************* */
SemiringFactor::shared_ptr SemiringFactor::sum(const Ordering& frontalKeys,
                                               const SemiringSum& sum) const {
  return eliminate(frontalKeys, sum).second;
}

}  // namespace gtsam
