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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

namespace gtsam {

namespace {

using ADT = AlgebraicDecisionTree<Key>;

double add(const double& a, const double& b) { return a + b; }
double subtract(const double& a, const double& b) { return a - b; }
double larger(const double& a, const double& b) { return std::max(a, b); }
double smaller(const double& a, const double& b) { return std::min(a, b); }

/// p * v where p is positive, and zero elsewhere, also if v is infinite.
double weigh(const double& p, const double& v) { return p > 0.0 ? p * v : 0.0; }

/**
 * The value table w / p, with a given value for impossible outcomes, so that
 * they cannot win a maximum (minus infinity) or a minimum (plus infinity).
 */
DecisionTreeFactor valuesOfPossible(const DecisionTreeFactor& probability,
                                    const DecisionTreeFactor& weighted,
                                    double impossible) {
  return probability.apply(weighted,
                           [impossible](const double p, const double w) {
                             return p > 0.0 ? w / p : impossible;
                           });
}

/**
 * The merged value of the frontal outcomes for every separator assignment:
 * their maximum, or their tilted mean
 *   (1 / tilt) log sum_x p(x | S) exp(tilt v(x, S)).
 * The tilted mean is computed relative to the extreme value, so that no
 * exponential overflows. The result is zero where the total probability is.
 */
DecisionTreeFactor mergedValue(const DecisionTreeFactor& probability,
                               const DecisionTreeFactor& weighted,
                               const DecisionTreeFactor& total,
                               const Ordering& frontalKeys,
                               const SemiringSum& sum) {
  const double infinity = std::numeric_limits<double>::infinity();
  const bool upward = sum.isMaximum() || sum.tilt() > 0.0;
  const DecisionTreeFactor values =
      valuesOfPossible(probability, weighted, upward ? -infinity : infinity);
  const DecisionTreeFactor extreme =
      *values.combine(frontalKeys, upward ? larger : smaller);
  const auto finite = [](const double p, const double v) {
    return p > 0.0 ? v : 0.0;
  };
  if (sum.isMaximum()) return total.apply(extreme, finite);

  // sum_x p(x, S) exp(tilt (v - extreme)), with every exponent <= 0.
  const double tilt = sum.tilt();
  const DecisionTreeFactor stretched =
      values.apply(extreme, [tilt](const double v, const double e) {
        return std::isfinite(v) ? std::exp(tilt * (v - e)) : 0.0;
      });
  const DecisionTreeFactor mass =
      *(probability * stretched).combine(frontalKeys, add);
  const DecisionTreeFactor relative =
      mass.apply(total, [tilt](const double m, const double p) {
        return p > 0.0 ? std::log(m / p) / tilt : 0.0;
      });
  return total.apply(extreme, finite).apply(relative, add);
}

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
    // A decision tree has no branch for a variable with a single value, and
    // tables on such a variable are read back wrongly.
    if (cardinality < 2) {
      throw std::invalid_argument(
          "SemiringDiscreteFactor: key " + DefaultKeyFormatter(key) +
          " has fewer than two values; leave it out of the factor");
    }
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
    const Ordering& frontalKeys, const SemiringSum& sum) const {
  const std::map<Key, size_t> cardinalities = probability_.cardinalities();
  for (Key key : frontalKeys) {
    if (!cardinalities.count(key)) {
      throw std::invalid_argument(
          "SemiringDiscreteFactor: cannot sum out " +
          DefaultKeyFormatter(key) + ", which is not in the factor");
    }
  }
  const DecisionTreeFactor total = *probability_.combine(frontalKeys, add);
  if (sum.isAverage()) {
    return FromChannels(total, *weighted_.combine(frontalKeys, add));
  }
  const DecisionTreeFactor merged =
      mergedValue(probability_, weighted_, total, frontalKeys, sum);
  return FromChannels(total, total.apply(merged, weigh));
}

/* ************************************************************************* */
SemiringFactor::EliminationResult SemiringDiscreteFactor::eliminate(
    const Ordering& frontalKeys, const SemiringSum& sum) const {
  const This marginal = sumOut(frontalKeys, sum);
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
