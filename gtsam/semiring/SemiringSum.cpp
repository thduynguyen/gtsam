/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SemiringSum.cpp
 * @brief   How a variable is summed out of a semiring factor
 * @author  Duy Ta
 */

#include <gtsam/semiring/SemiringSum.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace gtsam {

/* ************************************************************************* */
SemiringSum SemiringSum::Tilted(double tilt) {
  return tilt == 0.0 ? SemiringSum() : SemiringSum(TILTED, tilt);
}

/* ************************************************************************* */
SemiringSum SemiringSum::SoftMaximum(double temperature) {
  if (!(temperature > 0.0)) {
    throw std::invalid_argument(
        "SemiringSum::SoftMaximum: the temperature must be positive");
  }
  return SemiringSum(TILTED, 1.0 / temperature);
}

/* ************************************************************************* */
void SemiringSum::print(const std::string& s) const {
  std::cout << s;
  switch (kind_) {
    case AVERAGE:
      std::cout << "average";
      break;
    case MAXIMUM:
      std::cout << "maximum";
      break;
    case TILTED:
      std::cout << "tilted, tilt = " << tilt_;
      break;
  }
  std::cout << std::endl;
}

/* ************************************************************************* */
bool SemiringSum::equals(const SemiringSum& other, double tol) const {
  return kind_ == other.kind_ && std::abs(tilt_ - other.tilt_) <= tol;
}

/* ************************************************************************* */
void SemiringRules::print(const std::string& s,
                          const KeyFormatter& formatter) const {
  std::cout << s << std::endl;
  for (const auto& [key, sum] : rules_) {
    sum.print(" " + formatter(key) + ": ");
  }
}

/* ************************************************************************* */
bool SemiringRules::equals(const SemiringRules& other, double tol) const {
  KeySet keys;
  for (const auto& [key, sum] : rules_) keys.insert(key);
  for (const auto& [key, sum] : other.rules_) keys.insert(key);
  for (Key key : keys) {
    if (!at(key).equals(other.at(key), tol)) return false;
  }
  return true;
}

/* ************************************************************************* */
void SemiringRules::setAll(const KeyVector& keys, const SemiringSum& sum) {
  for (Key key : keys) rules_[key] = sum;
}

/* ************************************************************************* */
SemiringSum SemiringRules::at(Key key) const {
  const auto it = rules_.find(key);
  return it == rules_.end() ? SemiringSum() : it->second;
}

/* ************************************************************************* */
SemiringSum SemiringRules::common(const Ordering& keys) const {
  if (keys.empty()) return SemiringSum();
  const SemiringSum first = at(keys.front());
  for (Key key : keys) {
    if (!at(key).equals(first)) {
      throw std::invalid_argument(
          "SemiringRules: variables with different rules cannot be eliminated "
          "together; eliminate them one at a time");
    }
  }
  return first;
}

}  // namespace gtsam
