/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringDiscreteConditional.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/semiring/SemiringDiscreteConditional.h>

namespace gtsam {

namespace {

/// The keys of the joint, with the frontal keys moved to the front.
KeyVector orderedKeys(const SemiringDiscreteFactor& joint,
                      const Ordering& frontalKeys) {
  KeyVector keys(frontalKeys.begin(), frontalKeys.end());
  for (Key key : joint.keys()) {
    if (!frontalKeys.contains(key)) keys.push_back(key);
  }
  return keys;
}

}  // namespace

/* ************************************************************************* */
SemiringDiscreteConditional::SemiringDiscreteConditional(
    const SemiringDiscreteFactor& joint, const SemiringDiscreteFactor& marginal,
    const Ordering& frontalKeys)
    : Base(orderedKeys(joint, frontalKeys), frontalKeys.size(),
           std::make_shared<SemiringDiscreteFactor>(joint / marginal)) {}

/* ************************************************************************* */
const SemiringDiscreteFactor& SemiringDiscreteConditional::table() const {
  return static_cast<const SemiringDiscreteFactor&>(*factor_);
}

/* ************************************************************************* */
DiscreteConditional SemiringDiscreteConditional::probability() const {
  const DecisionTreeFactor& probability = table().probability();
  DiscreteKeys discreteKeys;
  for (Key key : keys()) {
    discreteKeys.emplace_back(key, probability.cardinality(key));
  }
  return DiscreteConditional(nrFrontals(), discreteKeys, probability);
}

}  // namespace gtsam
