/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    SemiringSum.h
 * @brief   How a variable is summed out of a semiring factor
 * @author  Duy Ta
 */

#pragma once

#include <gtsam/base/Testable.h>
#include <gtsam/dllexport.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Ordering.h>

#include <map>
#include <string>

namespace gtsam {

/**
 * The rule by which a variable is summed out of a semiring factor.
 *
 * All rules share the semiring product: probabilities multiply and values add.
 * They differ in how the values v of the outcomes x of the eliminated variable
 * are merged, given the separator S:
 *
 *  - Average: the mean of v under p(x | S). This is the expectation semiring,
 *    used for variables decided by chance, and for actions under a policy.
 *  - Maximum: the largest v among the outcomes with p(x | S) > 0, used for
 *    variables the agent chooses.
 *  - Tilted: (1 / tilt) log sum_x p(x | S) exp(tilt * v), which lies between
 *    the smallest and the largest value. A positive tilt leans toward the
 *    larger values and a negative tilt toward the smaller ones; a zero tilt is
 *    the average. With tilt = 1 / temperature it is the soft maximum.
 *
 * In every case the probabilities of the outcomes are added, and the value
 * channel of the resulting conditional is v minus the merged value: the
 * advantage, the regret, or the soft advantage.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringSum {
 public:
  /// The kind of rule.
  enum Kind { AVERAGE, MAXIMUM, TILTED };

 private:
  Kind kind_ = AVERAGE;
  double tilt_ = 0.0;

  SemiringSum(Kind kind, double tilt) : kind_(kind), tilt_(tilt) {}

 public:
  /// @name Standard Constructors
  /// @{

  /// The default rule is the average.
  SemiringSum() {}

  /// The mean of the values under the conditional probabilities.
  static SemiringSum Average() { return SemiringSum(); }

  /// The largest value among the possible outcomes.
  static SemiringSum Maximum() { return SemiringSum(MAXIMUM, 0.0); }

  /// The tilted mean of the values; a zero tilt gives the average.
  static SemiringSum Tilted(double tilt);

  /// The soft maximum at a positive temperature: the tilt 1 / temperature.
  static SemiringSum SoftMaximum(double temperature);

  /// @}
  /// @name Testable
  /// @{

  /// Print the rule.
  void print(const std::string& s = "") const;

  /// Check equality up to tolerance on the tilt.
  bool equals(const SemiringSum& other, double tol = 1e-9) const;

  /// @}
  /// @name Standard Interface
  /// @{

  /// The kind of rule.
  Kind kind() const { return kind_; }

  /// True if the rule is the average.
  bool isAverage() const { return kind_ == AVERAGE; }

  /// True if the rule is the maximum.
  bool isMaximum() const { return kind_ == MAXIMUM; }

  /// True if the rule is a tilted mean with a nonzero tilt.
  bool isTilted() const { return kind_ == TILTED; }

  /// The tilt, zero unless the rule is a tilted mean.
  double tilt() const { return tilt_; }

  /// @}
};

/// traits
template <>
struct traits<SemiringSum> : public Testable<SemiringSum> {};

/**
 * The rule for each variable of a semiring factor graph. A variable without an
 * entry is averaged out.
 *
 * With more than one kind of rule in a graph the elimination order is no
 * longer free: a variable must be eliminated before every chosen variable that
 * is decided without knowing it. For a Markov decision process that is the
 * backward order, each state before the action that leads to it.
 *
 * @ingroup semiring
 */
class GTSAM_EXPORT SemiringRules {
 private:
  std::map<Key, SemiringSum> rules_;

 public:
  /// @name Standard Constructors
  /// @{

  /// All variables are averaged out.
  SemiringRules() {}

  /// @}
  /// @name Testable
  /// @{

  /// Print the rules that differ from the average.
  void print(const std::string& s = "SemiringRules",
             const KeyFormatter& formatter = DefaultKeyFormatter) const;

  /// Check equality up to tolerance on the tilts.
  bool equals(const SemiringRules& other, double tol = 1e-9) const;

  /// @}
  /// @name Standard Interface
  /// @{

  /// Set the rule of one variable.
  void set(Key key, const SemiringSum& sum) { rules_[key] = sum; }

  /// Set the same rule for several variables.
  void setAll(const KeyVector& keys, const SemiringSum& sum);

  /// The rule of a variable, the average if none was set.
  SemiringSum at(Key key) const;

  /// The number of variables with a rule that was set.
  size_t size() const { return rules_.size(); }

  /**
   * The rule shared by variables that are eliminated together. Throws
   * std::invalid_argument if their rules differ.
   */
  SemiringSum common(const Ordering& keys) const;

  /// @}
};

/// traits
template <>
struct traits<SemiringRules> : public Testable<SemiringRules> {};

}  // namespace gtsam
