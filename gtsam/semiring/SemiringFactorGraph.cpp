/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringFactorGraph.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/inference/EliminateableFactorGraph-inst.h>
#include <gtsam/inference/FactorGraph-inst.h>
#include <gtsam/semiring/SemiringBayesNet.h>
#include <gtsam/semiring/SemiringBayesTree.h>
#include <gtsam/semiring/SemiringEliminationTree.h>
#include <gtsam/semiring/SemiringFactorGraph.h>
#include <gtsam/semiring/SemiringJunctionTree.h>

#include <stdexcept>

namespace gtsam {

// Instantiate base classes
template class FactorGraph<SemiringFactor>;
template class EliminateableFactorGraph<SemiringFactorGraph>;

/* ************************************************************************* */
bool SemiringFactorGraph::equals(const This& fg, double tol) const {
  return Base::equals(fg, tol);
}

/* ************************************************************************* */
SemiringFactor::shared_ptr SemiringFactorGraph::product() const {
  SemiringFactor::shared_ptr result;
  for (const sharedFactor& factor : factors_) {
    if (factor) result = result ? result->multiply(*factor) : factor;
  }
  return result;
}

/* ************************************************************************* */
double SemiringFactorGraph::expectation(const Ordering& ordering) const {
  // Elimination discards factors on no variables, which is where the total
  // ends up, so collect them as they are produced.
  SemiringFactor::shared_ptr total;
  const Eliminate collectTotal = [&total](const This& factors,
                                          const Ordering& keys) {
    auto result = EliminateSemiring(factors, keys);
    if (result.second->empty()) {
      total = total ? total->multiply(*result.second) : result.second;
    }
    return result;
  };
  eliminateSequential(ordering, collectTotal);
  return total ? total->expectation() : 0.0;
}

/* ************************************************************************* */
double SemiringFactorGraph::expectation() const {
  return expectation(Ordering::Colamd(VariableIndex(*this)));
}

/* ************************************************************************* */
std::pair<std::shared_ptr<SemiringConditional>, std::shared_ptr<SemiringFactor>>
EliminateSemiring(const SemiringFactorGraph& factors,
                  const Ordering& frontalKeys) {
  const SemiringFactor::shared_ptr product = factors.product();
  if (!product) {
    throw std::invalid_argument("EliminateSemiring: no factors to eliminate");
  }
  return product->eliminate(frontalKeys);
}

}  // namespace gtsam
