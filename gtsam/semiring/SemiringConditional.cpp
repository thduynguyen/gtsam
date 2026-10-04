/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringConditional.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/semiring/SemiringConditional.h>

#include <iostream>

namespace gtsam {

/* ************************************************************************* */
void SemiringConditional::print(const std::string& s,
                                const KeyFormatter& formatter) const {
  BaseConditional::print(s, formatter);
  if (factor_) factor_->print("", formatter);
}

/* ************************************************************************* */
bool SemiringConditional::equals(const SemiringFactor& other,
                                 double tol) const {
  const auto* conditional = dynamic_cast<const This*>(&other);
  if (!conditional) return false;
  if (!BaseConditional::equals(*conditional, tol)) return false;
  if (!factor_ || !conditional->factor_) {
    return factor_ == conditional->factor_;
  }
  return factor_->equals(*conditional->factor_, tol);
}

/* ************************************************************************* */
SemiringFactor::shared_ptr SemiringConditional::multiply(
    const SemiringFactor& other) const {
  return factor_->multiply(other);
}

/* ************************************************************************* */
SemiringFactor::EliminationResult SemiringConditional::eliminate(
    const Ordering& frontalKeys, const SemiringSum& sum) const {
  return factor_->eliminate(frontalKeys, sum);
}

/* ************************************************************************* */
double SemiringConditional::expectation() const {
  return factor_->expectation();
}

}  // namespace gtsam
