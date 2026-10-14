/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringGaussianConditional.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/semiring/SemiringGaussianConditional.h>

namespace gtsam {

namespace {

/// The conditional's keys, followed by keys that are only in the surprise.
KeyVector orderedKeys(const GaussianConditional::shared_ptr& conditional,
                      const HessianFactor::shared_ptr& surprise) {
  KeyVector keys = conditional->keys();
  if (surprise) {
    KeySet extra(surprise->begin(), surprise->end());
    for (Key key : keys) extra.erase(key);
    keys.insert(keys.end(), extra.begin(), extra.end());
  }
  return keys;
}

/// Both channels of the conditional as a Gaussian semiring factor.
SemiringFactor::shared_ptr channelsAsFactor(
    const GaussianConditional::shared_ptr& conditional,
    const HessianFactor::shared_ptr& surprise) {
  GaussianFactorGraph gaussian;
  gaussian.push_back(conditional);
  return std::make_shared<SemiringGaussianFactor>(gaussian, surprise);
}

}  // namespace

/* ************************************************************************* */
SemiringGaussianConditional::SemiringGaussianConditional(
    const GaussianConditional::shared_ptr& conditional,
    const HessianFactor::shared_ptr& surprise)
    : Base(orderedKeys(conditional, surprise), conditional->nrFrontals(),
           channelsAsFactor(conditional, surprise)),
      conditional_(conditional) {}

/* ************************************************************************* */
const HessianFactor::shared_ptr& SemiringGaussianConditional::surprise()
    const {
  return static_cast<const SemiringGaussianFactor&>(*factor_).value();
}

/* ************************************************************************* */
double SemiringGaussianConditional::surprise(const VectorValues& x) const {
  return static_cast<const SemiringGaussianFactor&>(*factor_).value(x);
}

}  // namespace gtsam
