/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file SemiringFactorGraph.h
 * @brief Factor graph of factors valued in the expectation semiring
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/inference/EliminateableFactorGraph.h>
#include <gtsam/inference/FactorGraph.h>
#include <gtsam/semiring/SemiringConditional.h>
#include <gtsam/semiring/SemiringFactor.h>

#include <string>
#include <utility>

namespace gtsam {

// Forward declarations
class SemiringFactorGraph;
class SemiringBayesNet;
class SemiringEliminationTree;
class SemiringBayesTree;
class SemiringJunctionTree;

/**
 * Main elimination function for SemiringFactorGraph: multiply all factors,
 * sum out the frontal variables, and divide to obtain the conditional. It only
 * uses the SemiringFactor interface, so it works for any factor family.
 *
 * @param factors The factors involving the frontal variables.
 * @param frontalKeys The variables to eliminate.
 * @return The conditional on the frontal variables and the separator factor.
 * @ingroup semiring
 */
GTSAM_EXPORT
std::pair<std::shared_ptr<SemiringConditional>, std::shared_ptr<SemiringFactor>>
EliminateSemiring(const SemiringFactorGraph& factors,
                  const Ordering& frontalKeys);

template <>
struct EliminationTraits<SemiringFactorGraph> {
  typedef SemiringFactor FactorType;  ///< Type of factors in factor graph
  typedef SemiringFactorGraph FactorGraphType;  ///< Type of the factor graph
  typedef SemiringConditional ConditionalType;  ///< Type of conditionals
  typedef SemiringBayesNet BayesNetType;        ///< Type of Bayes net
  typedef SemiringEliminationTree EliminationTreeType;  ///< Elimination tree
  typedef SemiringBayesTree BayesTreeType;              ///< Type of Bayes tree
  typedef SemiringJunctionTree JunctionTreeType;  ///< Type of Junction tree

  /// The default dense elimination function
  static std::pair<std::shared_ptr<ConditionalType>,
                   std::shared_ptr<FactorType>>
  DefaultEliminate(const FactorGraphType& factors, const Ordering& keys) {
    return EliminateSemiring(factors, keys);
  }

  /// The default ordering generation function
  static Ordering DefaultOrderingFunc(
      const FactorGraphType& graph,
      std::optional<std::reference_wrapper<const VariableIndex>>
          variableIndex) {
    return Ordering::Colamd((*variableIndex).get());
  }
};

/**
 * A factor graph of semiring factors. Variable elimination on it computes
 * expected additive values: for a Markov decision process with transition,
 * policy and reward factors, eliminating backward in time performs Bellman
 * backups, leaving value functions in the separator factors and surprises
 * (advantage functions and temporal-difference residuals) in the conditionals.
 *
 * All factors in a graph must belong to the same family, discrete or Gaussian.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringFactorGraph
    : public FactorGraph<SemiringFactor>,
      public EliminateableFactorGraph<SemiringFactorGraph> {
 public:
  using This = SemiringFactorGraph;          ///< this class
  using Base = FactorGraph<SemiringFactor>;  ///< base factor graph type
  using BaseEliminateable =
      EliminateableFactorGraph<This>;        ///< for elimination
  using shared_ptr = std::shared_ptr<This>;  ///< shared_ptr to This

  /// @name Standard Constructors
  /// @{

  /// Default constructor
  SemiringFactorGraph() {}

  /// Construct from iterator over factors
  template <typename ITERATOR>
  SemiringFactorGraph(ITERATOR firstFactor, ITERATOR lastFactor)
      : Base(firstFactor, lastFactor) {}

  /// Construct from container of factors (shared_ptr or plain objects)
  template <class CONTAINER>
  explicit SemiringFactorGraph(const CONTAINER& factors) : Base(factors) {}

  /**
   * Implicit copy/downcast constructor to override explicit template
   * container constructor
   */
  template <class DERIVED_FACTOR>
  SemiringFactorGraph(const FactorGraph<DERIVED_FACTOR>& graph) : Base(graph) {}

  /// @}
  /// @name Testable
  /// @{

  /// Check equality up to tolerance.
  bool equals(const This& fg, double tol = 1e-9) const;

  /// @}
  /// @name Standard Interface
  /// @{

  /// Semiring product of all factors, nullptr if the graph is empty.
  SemiringFactor::shared_ptr product() const;

  /**
   * Expected total value under the normalized product of all probability
   * channels, e.g., the expected total reward of a policy. All variables are
   * eliminated in the given order.
   */
  double expectation(const Ordering& ordering) const;

  /// Expected total value, eliminating in the default (COLAMD) order.
  double expectation() const;

  /// @}
};

/// traits
template <>
struct traits<SemiringFactorGraph> : public Testable<SemiringFactorGraph> {};

}  // namespace gtsam
