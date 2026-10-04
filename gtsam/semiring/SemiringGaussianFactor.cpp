/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringGaussianFactor.cpp
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/linear/Scatter.h>
#include <gtsam/semiring/SemiringGaussianConditional.h>
#include <gtsam/semiring/SemiringGaussianFactor.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace gtsam {

namespace {

/// View a semiring factor or conditional as a Gaussian semiring factor.
const SemiringGaussianFactor& asGaussian(const SemiringFactor& factor) {
  if (const auto* conditional =
          dynamic_cast<const SemiringConditional*>(&factor)) {
    return asGaussian(*conditional->factor());
  }
  if (const auto* gaussian =
          dynamic_cast<const SemiringGaussianFactor*>(&factor)) {
    return *gaussian;
  }
  throw std::invalid_argument(
      "SemiringGaussianFactor: cannot combine with a non-Gaussian factor");
}

/// Record the dimension of every variable in a Gaussian factor.
void addDimensions(const GaussianFactor& factor,
                   std::map<Key, size_t>* dimensions) {
  for (auto it = factor.begin(); it != factor.end(); ++it) {
    (*dimensions)[*it] = factor.getDim(it);
  }
}

/// Look up the dimensions of the given keys, in order.
std::vector<size_t> dimensionsOf(const KeyVector& keys,
                                 const std::map<Key, size_t>& dimensions) {
  std::vector<size_t> result;
  for (Key key : keys) result.push_back(dimensions.at(key));
  return result;
}

/// Build a quadratic from keys and a dense augmented information matrix.
HessianFactor::shared_ptr makeQuadratic(
    const KeyVector& keys, const std::map<Key, size_t>& dimensions,
    const Matrix& augmentedInformation) {
  return std::make_shared<HessianFactor>(
      keys, SymmetricBlockMatrix(dimensionsOf(keys, dimensions),
                                 augmentedInformation, true));
}

/// The quadratic scale * quadratic.error(x).
HessianFactor::shared_ptr scaledQuadratic(const GaussianFactor& quadratic,
                                          double scale) {
  const HessianFactor hessian(quadratic);
  std::map<Key, size_t> dimensions;
  addDimensions(hessian, &dimensions);
  return makeQuadratic(hessian.keys(), dimensions,
                       scale * hessian.augmentedInformation());
}

/// The sum of two quadratics, either of which can be nullptr to denote zero.
HessianFactor::shared_ptr addQuadratics(const HessianFactor::shared_ptr& a,
                                        const HessianFactor::shared_ptr& b) {
  if (!a) return b;
  if (!b) return a;
  GaussianFactorGraph graph;
  graph.push_back(a);
  graph.push_back(b);
  return std::make_shared<HessianFactor>(graph);
}

/// The dense augmented information of a quadratic on the given ordered keys.
Matrix denseQuadratic(const HessianFactor::shared_ptr& quadratic,
                      const KeyVector& keys,
                      const std::map<Key, size_t>& dimensions) {
  Scatter scatter;
  for (Key key : keys) scatter.add(key, dimensions.at(key));
  GaussianFactorGraph graph;
  graph.push_back(quadratic);
  return HessianFactor(graph, scatter).augmentedInformation();
}

/**
 * A Gaussian conditional p(x | S) written as x = K S + k + W e, with e a
 * standard normal vector, for separator variables S stacked in a given order.
 */
struct AffineGaussian {
  Matrix K;  ///< gain on the separator
  Vector k;  ///< offset
  Matrix W;  ///< square-root covariance, Sigma = W W'

  /// Convert a conditional R x + T S = d, with noise sigmas on each row.
  AffineGaussian(const GaussianConditional& conditional,
                 const KeyVector& separator,
                 const std::map<Key, size_t>& dimensions) {
    const Matrix R = conditional.R();
    const auto upper = R.triangularView<Eigen::Upper>();
    const size_t frontalDim = R.rows();

    // Scatter the parent blocks of the conditional into the full separator.
    std::map<Key, size_t> offsets;
    size_t separatorDim = 0;
    for (Key key : separator) {
      offsets[key] = separatorDim;
      separatorDim += dimensions.at(key);
    }
    Matrix T = Matrix::Zero(frontalDim, separatorDim);
    for (auto it = conditional.beginParents(); it != conditional.endParents();
         ++it) {
      T.middleCols(offsets.at(*it), dimensions.at(*it)) = conditional.getA(it);
    }

    const auto& model = conditional.get_model();
    const Vector sigmas = model ? model->sigmas() : Vector::Ones(frontalDim);
    K = -upper.solve(T);
    k = upper.solve(Vector(conditional.d()));
    W = upper.solve(Matrix(sigmas.asDiagonal()));
  }
};

/**
 * Expectation of a quadratic in z = [x; S] under x = K S + k + W e, as a
 * quadratic in S. Both quadratics are dense augmented information matrices.
 */
Matrix expectedQuadratic(const Matrix& augmented,
                         const AffineGaussian& affine) {
  const size_t frontalDim = affine.K.rows(), separatorDim = affine.K.cols();
  const size_t dim = frontalDim + separatorDim;

  // [z; -1] = P [S; -1] + [W e; 0; 0]
  Matrix P = Matrix::Zero(dim + 1, separatorDim + 1);
  P.topLeftCorner(frontalDim, separatorDim) = affine.K;
  P.topRightCorner(frontalDim, 1) = -affine.k;
  P.bottomRightCorner(separatorDim + 1, separatorDim + 1).setIdentity();

  Matrix expected = P.transpose() * augmented * P;
  // The noise contributes the constant 0.5 tr(G_xx Sigma).
  expected(separatorDim, separatorDim) +=
      (affine.W.transpose() *
       augmented.topLeftCorner(frontalDim, frontalDim) * affine.W)
          .trace();
  return expected;
}

}  // namespace

/* ************************************************************************* */
SemiringGaussianFactor::SemiringGaussianFactor(
    const GaussianFactor::shared_ptr& gaussian)
    : This(GaussianFactorGraph(
          std::vector<GaussianFactor::shared_ptr>{gaussian})) {}

/* ************************************************************************* */
SemiringGaussianFactor::SemiringGaussianFactor(
    const GaussianFactorGraph& gaussian, const HessianFactor::shared_ptr& value)
    : gaussian_(gaussian), value_(value) {
  KeySet keys = gaussian_.keys();
  if (value_) keys.insert(value_->begin(), value_->end());
  keys_.assign(keys.begin(), keys.end());
}

/* ************************************************************************* */
SemiringGaussianFactor SemiringGaussianFactor::Reward(
    const GaussianFactor& quadratic) {
  return This(GaussianFactorGraph(), scaledQuadratic(quadratic, 1.0));
}

/* ************************************************************************* */
SemiringGaussianFactor SemiringGaussianFactor::Cost(
    const GaussianFactor& quadratic) {
  return This(GaussianFactorGraph(), scaledQuadratic(quadratic, -1.0));
}

/* ************************************************************************* */
void SemiringGaussianFactor::print(const std::string& s,
                                   const KeyFormatter& formatter) const {
  std::cout << s << std::endl;
  gaussian_.print(" probability:", formatter);
  if (value_) {
    value_->print(" value:", formatter);
  } else {
    std::cout << " value: zero" << std::endl;
  }
}

/* ************************************************************************* */
bool SemiringGaussianFactor::equals(const SemiringFactor& other,
                                    double tol) const {
  const auto* factor = dynamic_cast<const This*>(&other);
  if (!factor || !gaussian_.equals(factor->gaussian_, tol)) return false;
  if (!value_ || !factor->value_) return value_ == factor->value_;
  return value_->equals(*factor->value_, tol);
}

/* ************************************************************************* */
double SemiringGaussianFactor::value(const VectorValues& x) const {
  return value_ ? value_->error(x) : 0.0;
}

/* ************************************************************************* */
SemiringFactor::shared_ptr SemiringGaussianFactor::multiply(
    const SemiringFactor& other) const {
  const This& factor = asGaussian(other);
  GaussianFactorGraph gaussian = gaussian_;
  gaussian.push_back(factor.gaussian_);
  return std::make_shared<This>(gaussian,
                                addQuadratics(value_, factor.value_));
}

/* ************************************************************************* */
SemiringFactor::EliminationResult SemiringGaussianFactor::eliminate(
    const Ordering& frontalKeys) const {
  const KeySet gaussianKeys = gaussian_.keys();
  for (Key key : frontalKeys) {
    if (!gaussianKeys.count(key)) {
      throw std::invalid_argument(
          "SemiringGaussianFactor::eliminate: no Gaussian factor involves " +
          DefaultKeyFormatter(key) + ", so expectations over it are undefined");
    }
  }

  // Probability channel: ordinary Gaussian elimination.
  const auto [conditional, remaining] =
      EliminatePreferCholesky(gaussian_, frontalKeys);
  GaussianFactorGraph separatorGaussian;
  if (remaining && !remaining->empty()) separatorGaussian.push_back(remaining);
  if (!value_) {
    return {std::make_shared<SemiringGaussianConditional>(conditional),
            std::make_shared<This>(separatorGaussian)};
  }

  // Stack the variables as z = [frontals; separator].
  KeyVector separator;
  for (Key key : keys()) {
    if (!frontalKeys.contains(key)) separator.push_back(key);
  }
  KeyVector stacked(conditional->beginFrontals(), conditional->endFrontals());
  stacked.insert(stacked.end(), separator.begin(), separator.end());
  std::map<Key, size_t> dimensions;
  for (const auto& factor : gaussian_) addDimensions(*factor, &dimensions);
  addDimensions(*value_, &dimensions);

  // Value channel: expectation under the conditional, and the surprise.
  const Matrix value = denseQuadratic(value_, stacked, dimensions);
  const Matrix expected = expectedQuadratic(
      value, AffineGaussian(*conditional, separator, dimensions));
  Matrix surprise = value;
  surprise.bottomRightCorner(expected.rows(), expected.cols()) -= expected;

  return {std::make_shared<SemiringGaussianConditional>(
              conditional, makeQuadratic(stacked, dimensions, surprise)),
          std::make_shared<This>(
              separatorGaussian,
              makeQuadratic(separator, dimensions, expected))};
}

/* ************************************************************************* */
double SemiringGaussianFactor::expectation() const {
  if (empty()) return value_ ? 0.5 * value_->constantTerm() : 0.0;
  return eliminate(Ordering(keys())).second->expectation();
}

}  // namespace gtsam
