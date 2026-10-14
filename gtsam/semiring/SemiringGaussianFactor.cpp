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
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/Scatter.h>
#include <gtsam/semiring/SemiringGaussianConditional.h>
#include <gtsam/semiring/SemiringGaussianFactor.h>

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
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

/**
 * The largest finite absolute entry of the information matrix of a factor.
 * A hard constraint has infinite entries, which are skipped here and reported
 * through `constrained`.
 */
double informationScale(const GaussianFactor& factor,
                        bool* constrained = nullptr) {
  const Matrix information = factor.information();
  double scale = 0.0;
  for (Eigen::Index i = 0; i < information.size(); i++) {
    const double entry = std::abs(information.data()[i]);
    if (std::isfinite(entry)) {
      scale = std::max(scale, entry);
    } else if (constrained) {
      *constrained = true;
    }
  }
  return scale;
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

  /// Construct from the gain, the offset and the square-root covariance.
  AffineGaussian(const Matrix& K, const Vector& k, const Matrix& W)
      : K(K), k(k), W(W) {}

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
 * Mean, or tilted mean, of a quadratic in z = [x; S] under x = K S + k + W e,
 * as a quadratic in S. Both quadratics are dense augmented information
 * matrices.
 *
 * Write the value along x = mu + W e, with mu = K S + k, as
 *   v = v(mu) + b' e + 0.5 e' M e,
 * where M = W' G_xx W and b is affine in S. Its mean under e ~ N(0, I) is
 * v(mu) + 0.5 tr(M). Its tilted mean (1 / tilt) log E[exp(tilt v)] is
 *   v(mu) + 0.5 tilt b' N^-1 b - (0.5 / tilt) log det N,   N = I - tilt M,
 * which exists only if N is positive definite.
 */
Matrix expectedQuadratic(const Matrix& augmented, const AffineGaussian& affine,
                         double tilt = 0.0) {
  const size_t frontalDim = affine.K.rows(), separatorDim = affine.K.cols();
  const size_t dim = frontalDim + separatorDim;

  // [z; -1] = P [S; -1] + [W e; 0; 0]
  Matrix P = Matrix::Zero(dim + 1, separatorDim + 1);
  P.topLeftCorner(frontalDim, separatorDim) = affine.K;
  P.topRightCorner(frontalDim, 1) = -affine.k;
  P.bottomRightCorner(separatorDim + 1, separatorDim + 1).setIdentity();

  Matrix expected = P.transpose() * augmented * P;
  const Matrix M = affine.W.transpose() *
                   augmented.topLeftCorner(frontalDim, frontalDim) * affine.W;
  if (tilt == 0.0) {
    // The noise contributes the constant 0.5 tr(G_xx Sigma).
    expected(separatorDim, separatorDim) += M.trace();
    return expected;
  }

  const size_t noiseDim = affine.W.cols();
  if (noiseDim == 0) return expected;
  const Matrix N = Matrix::Identity(noiseDim, noiseDim) - tilt * M;
  const Eigen::LLT<Matrix> cholesky(N);
  if (cholesky.info() != Eigen::Success) {
    throw std::invalid_argument(
        "SemiringGaussianFactor: the tilt is too strong for the noise, so the "
        "tilted mean of the value is infinite");
  }
  // b = B [S; -1], the slope of the value along the noise directions.
  const Matrix B = affine.W.transpose() * augmented.topRows(frontalDim) * P;
  expected += tilt * B.transpose() * cholesky.solve(B);
  const double logDeterminant =
      2.0 * cholesky.matrixLLT().diagonal().array().log().sum();
  expected(separatorDim, separatorDim) -= logDeterminant / tilt;
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
SemiringFactor::EliminationResult SemiringGaussianFactor::eliminateByMaximum(
    const Ordering& frontalKeys) const {
  if (!value_) {
    throw std::invalid_argument(
        "SemiringGaussianFactor::eliminate: there is no value to maximize");
  }

  // Stack the variables as z = [frontals; separator].
  const KeyVector frontals(frontalKeys.begin(), frontalKeys.end());
  KeyVector separator;
  for (Key key : keys()) {
    if (!frontalKeys.contains(key)) separator.push_back(key);
  }
  KeyVector stacked = frontals;
  stacked.insert(stacked.end(), separator.begin(), separator.end());
  std::map<Key, size_t> dimensions;
  for (const auto& factor : gaussian_) addDimensions(*factor, &dimensions);
  addDimensions(*value_, &dimensions);
  for (Key key : frontals) {
    if (!dimensions.count(key)) {
      throw std::invalid_argument(
          "SemiringGaussianFactor::eliminate: the value does not involve " +
          DefaultKeyFormatter(key));
    }
  }
  size_t frontalDim = 0, separatorDim = 0;
  for (Key key : frontals) frontalDim += dimensions.at(key);
  for (Key key : separator) separatorDim += dimensions.at(key);

  // The value is 0.5 [x; S; -1]' G [x; S; -1]. It is stationary in x where
  // G_xx x + G_xS S - g_x = 0, a maximum if G_xx is negative definite.
  const Matrix value = denseQuadratic(value_, stacked, dimensions);
  const Eigen::LLT<Matrix> cholesky(
      -value.topLeftCorner(frontalDim, frontalDim));
  if (cholesky.info() != Eigen::Success) {
    throw std::invalid_argument(
        "SemiringGaussianFactor::eliminate: the value is not strictly concave "
        "in the variables to maximize over, so it has no maximum");
  }
  // No Gaussian factor may carry information on the variables to maximize
  // over. Eliminating a normalized conditional, such as the dynamics, leaves a
  // factor on them that is flat up to rounding; such factors are dropped.
  const double negligible =
      1e-9 *
      value.topLeftCorner(frontalDim, frontalDim).cwiseAbs().maxCoeff();
  GaussianFactorGraph separatorGaussian;
  for (const auto& factor : gaussian_) {
    bool involvesFrontal = false;
    for (Key key : *factor) {
      if (frontalKeys.contains(key)) involvesFrontal = true;
    }
    if (!involvesFrontal) {
      separatorGaussian.push_back(factor);
      continue;
    }
    bool constrained = false;
    if (informationScale(*factor, &constrained) > negligible || constrained) {
      throw std::invalid_argument(
          "SemiringGaussianFactor::eliminate: a variable with a Gaussian "
          "density cannot be eliminated by maximum; remove its density, "
          "e.g., the policy factor");
    }
  }

  const Matrix K = cholesky.solve(
      Matrix(value.block(0, frontalDim, frontalDim, separatorDim)));
  const Vector k = -cholesky.solve(
      Vector(value.block(0, frontalDim + separatorDim, frontalDim, 1)));

  // Substitute the maximizer x = K S + k, which has no noise.
  const Matrix best = expectedQuadratic(
      value, AffineGaussian(K, k, Matrix::Zero(frontalDim, 0)));
  Matrix regret = value;
  regret.bottomRightCorner(best.rows(), best.cols()) -= best;

  // The maximizer as a deterministic conditional, x - K S = k.
  std::vector<std::pair<Key, Matrix>> terms;
  size_t offset = 0;
  for (Key key : frontals) {
    Matrix block = Matrix::Zero(frontalDim, dimensions.at(key));
    block.middleRows(offset, dimensions.at(key)).setIdentity();
    terms.emplace_back(key, block);
    offset += dimensions.at(key);
  }
  offset = 0;
  for (Key key : separator) {
    terms.emplace_back(key, -K.middleCols(offset, dimensions.at(key)));
    offset += dimensions.at(key);
  }
  const auto conditional = std::make_shared<GaussianConditional>(
      terms, frontals.size(), k, noiseModel::Constrained::All(frontalDim));

  return {std::make_shared<SemiringGaussianConditional>(
              conditional, makeQuadratic(stacked, dimensions, regret)),
          std::make_shared<This>(separatorGaussian,
                                 makeQuadratic(separator, dimensions, best))};
}

/* ************************************************************************* */
SemiringFactor::EliminationResult SemiringGaussianFactor::eliminate(
    const Ordering& frontalKeys, const SemiringSum& sum) const {
  if (sum.isMaximum()) return eliminateByMaximum(frontalKeys);

  const KeySet gaussianKeys = gaussian_.keys();
  for (Key key : frontalKeys) {
    if (!gaussianKeys.count(key)) {
      throw std::invalid_argument(
          "SemiringGaussianFactor::eliminate: no Gaussian factor involves " +
          DefaultKeyFormatter(key) + ", so expectations over it are undefined");
    }
  }

  // Probability channel: ordinary Gaussian elimination. QR keeps the rows of
  // the factors, so that a flat remainder is recognized exactly, by having no
  // rows, and not through a tolerance on rounding errors.
  const auto [conditional, remaining] = EliminateQR(gaussian_, frontalKeys);
  // Keep what is left on the separator, unless it is flat. Eliminating a
  // normalized conditional, such as the dynamics, uses up all its rows and
  // leaves a factor with none, which carries no information.
  GaussianFactorGraph separatorGaussian;
  if (remaining && !remaining->empty() && remaining->rows() > 0) {
    separatorGaussian.push_back(remaining);
  }
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

  // Value channel: mean or tilted mean under the conditional, and the
  // surprise.
  const Matrix value = denseQuadratic(value_, stacked, dimensions);
  const Matrix expected = expectedQuadratic(
      value, AffineGaussian(*conditional, separator, dimensions), sum.tilt());
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
