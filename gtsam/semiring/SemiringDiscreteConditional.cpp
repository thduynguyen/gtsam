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

#include <algorithm>
#include <cmath>
#include <limits>

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
DiscreteConditional SemiringDiscreteConditional::greedy() const {
  const double infinity = std::numeric_limits<double>::infinity();
  const DecisionTreeFactor& probability = table().probability();
  const Ordering frontalKeys(KeyVector(beginFrontals(), endFrontals()));
  const DecisionTreeFactor surprises = probability.apply(
      table().weightedValue(), [infinity](const double p, const double w) {
        return p > 0.0 ? w / p : -infinity;
      });
  const DecisionTreeFactor best = *surprises.combine(
      frontalKeys,
      [](const double a, const double b) { return std::max(a, b); });
  const DecisionTreeFactor chosen =
      surprises.apply(best, [](const double v, const double b) {
        return std::isfinite(v) && v >= b - 1e-12 ? 1.0 : 0.0;
      });
  return normalized(chosen);
}

/* ************************************************************************* */
DiscreteConditional SemiringDiscreteConditional::tilted(double tilt) const {
  const DecisionTreeFactor& probability = table().probability();
  const Ordering frontalKeys(KeyVector(beginFrontals(), endFrontals()));
  const DecisionTreeFactor surprises = table().value();
  // Shift by the extreme surprise, so that no exponential overflows.
  const DecisionTreeFactor extreme = *surprises.combine(
      frontalKeys, [tilt](const double a, const double b) {
        return tilt > 0.0 ? std::max(a, b) : std::min(a, b);
      });
  const DecisionTreeFactor stretched =
      surprises.apply(extreme, [tilt](const double v, const double e) {
        return std::exp(tilt * (v - e));
      });
  return normalized(probability * stretched);
}

/* ************************************************************************* */
DiscreteConditional SemiringDiscreteConditional::normalized(
    const DecisionTreeFactor& weights) const {
  const Ordering frontalKeys(KeyVector(beginFrontals(), endFrontals()));
  const DecisionTreeFactor total = *weights.combine(
      frontalKeys, [](const double a, const double b) { return a + b; });
  DiscreteKeys discreteKeys;
  for (Key key : keys()) {
    discreteKeys.emplace_back(key, table().probability().cardinality(key));
  }
  return DiscreteConditional(nrFrontals(), discreteKeys, weights / total);
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
