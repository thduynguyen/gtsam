/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringDiscreteFactor.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/semiring/SemiringDiscreteConditional.h>
#include <gtsam/semiring/SemiringDiscreteFactor.h>

#include <iostream>
#include <map>
#include <stdexcept>

namespace gtsam {

namespace {

using ADT = AlgebraicDecisionTree<Key>;

double add(const double& a, const double& b) { return a + b; }
double subtract(const double& a, const double& b) { return a - b; }

/// The union of the discrete keys of two tables, sorted in increasing order.
DiscreteKeys unionKeys(const DecisionTreeFactor& a,
                       const DecisionTreeFactor& b) {
  std::map<Key, size_t> cardinalities = a.cardinalities();
  for (const auto& [key, cardinality] : b.cardinalities()) {
    const auto [it, inserted] = cardinalities.emplace(key, cardinality);
    if (!inserted && it->second != cardinality) {
      throw std::invalid_argument(
          "SemiringDiscreteFactor: inconsistent cardinalities for key " +
          DefaultKeyFormatter(key));
    }
  }
  DiscreteKeys keys;
  for (const auto& [key, cardinality] : cardinalities) {
    keys.emplace_back(key, cardinality);
  }
  return keys;
}

/// View a semiring factor or conditional as a discrete semiring factor.
const SemiringDiscreteFactor& asDiscrete(const SemiringFactor& factor) {
  if (const auto* conditional =
          dynamic_cast<const SemiringConditional*>(&factor)) {
    return asDiscrete(*conditional->factor());
  }
  if (const auto* discrete =
          dynamic_cast<const SemiringDiscreteFactor*>(&factor)) {
    return *discrete;
  }
  throw std::invalid_argument(
      "SemiringDiscreteFactor: cannot combine with a non-discrete factor");
}

}  // namespace

/* ************************************************************************* */
SemiringDiscreteFactor::SemiringDiscreteFactor(
    const DecisionTreeFactor& probability)
    : This(FromChannels(
          probability,
          DecisionTreeFactor(probability.discreteKeys(), ADT(0.0)))) {}

/* ************************************************************************* */
SemiringDiscreteFactor::SemiringDiscreteFactor(
    const DecisionTreeFactor& probability, const DecisionTreeFactor& value)
    : This(FromChannels(probability, probability * value)) {}

/* ************************************************************************* */
SemiringDiscreteFactor SemiringDiscreteFactor::Reward(
    const DecisionTreeFactor& reward) {
  return FromChannels(DecisionTreeFactor(reward.discreteKeys(), ADT(1.0)),
                      reward);
}

/* ************************************************************************* */
SemiringDiscreteFactor SemiringDiscreteFactor::FromChannels(
    const DecisionTreeFactor& probability,
    const DecisionTreeFactor& weightedValue) {
  const DiscreteKeys keys = unionKeys(probability, weightedValue);
  This factor;
  factor.keys_ = keys.indices();
  factor.probability_ =
      DecisionTreeFactor(keys, static_cast<const ADT&>(probability));
  factor.weighted_ =
      DecisionTreeFactor(keys, static_cast<const ADT&>(weightedValue));
  return factor;
}

/* ************************************************************************* */
void SemiringDiscreteFactor::print(const std::string& s,
                                   const KeyFormatter& formatter) const {
  std::cout << s << std::endl;
  probability_.print(" probability:", formatter);
  value().print(" value:", formatter);
}

/* ************************************************************************* */
bool SemiringDiscreteFactor::equals(const SemiringFactor& other,
                                    double tol) const {
  const auto* factor = dynamic_cast<const This*>(&other);
  return factor && probability_.equals(factor->probability_, tol) &&
         weighted_.equals(factor->weighted_, tol);
}

/* ************************************************************************* */
DecisionTreeFactor SemiringDiscreteFactor::value() const {
  return weighted_ / probability_;
}

/* ************************************************************************* */
std::pair<double, double> SemiringDiscreteFactor::evaluate(
    const DiscreteValues& values) const {
  const double probability = probability_(values);
  const double weighted = weighted_(values);
  return {probability, probability == 0.0 ? 0.0 : weighted / probability};
}

/* ************************************************************************* */
SemiringDiscreteFactor SemiringDiscreteFactor::operator*(
    const This& other) const {
  return FromChannels(probability_ * other.probability_,
                      (probability_ * other.weighted_)
                          .apply(weighted_ * other.probability_, add));
}

/* ************************************************************************* */
SemiringDiscreteFactor SemiringDiscreteFactor::operator/(
    const This& other) const {
  const DecisionTreeFactor probability = probability_ / other.probability_;
  const DecisionTreeFactor otherValue = other.weighted_ / other.probability_;
  return FromChannels(probability,
                      (weighted_ / other.probability_)
                          .apply(probability * otherValue, subtract));
}

/* ************************************************************************* */
SemiringFactor::shared_ptr SemiringDiscreteFactor::multiply(
    const SemiringFactor& other) const {
  return std::make_shared<This>(*this * asDiscrete(other));
}

/* ************************************************************************* */
SemiringDiscreteFactor SemiringDiscreteFactor::sumOut(
    const Ordering& frontalKeys) const {
  const std::map<Key, size_t> cardinalities = probability_.cardinalities();
  for (Key key : frontalKeys) {
    if (!cardinalities.count(key)) {
      throw std::invalid_argument(
          "SemiringDiscreteFactor: cannot sum out " +
          DefaultKeyFormatter(key) + ", which is not in the factor");
    }
  }
  return FromChannels(*probability_.combine(frontalKeys, add),
                      *weighted_.combine(frontalKeys, add));
}

/* ************************************************************************* */
SemiringFactor::EliminationResult SemiringDiscreteFactor::eliminate(
    const Ordering& frontalKeys) const {
  const This marginal = sumOut(frontalKeys);
  auto conditional = std::make_shared<SemiringDiscreteConditional>(
      *this, marginal, frontalKeys);
  return {conditional, std::make_shared<This>(marginal)};
}

/* ************************************************************************* */
double SemiringDiscreteFactor::expectation() const {
  const This total = sumOut(Ordering(keys()));
  const auto [probability, value] = total.evaluate(DiscreteValues());
  if (probability == 0.0) {
    throw std::runtime_error(
        "SemiringDiscreteFactor::expectation: total probability is zero");
  }
  return value;
}

}  // namespace gtsam
