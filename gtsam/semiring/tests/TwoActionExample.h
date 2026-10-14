/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file TwoActionExample.h
 * @brief A one-step decision problem shared by the semiring unit tests
 * @date Oct 3, 2026
 * @author Duy-Nguyen Ta
 */

#pragma once

#include <gtsam/discrete/DiscreteValues.h>
#include <gtsam/semiring/SemiringDiscreteFactor.h>

namespace gtsam {

/**
 * One decision from a single start state. The action is L or R, chosen with
 * probability (0.6, 0.4), where L costs 1. The next state is good or bad, with
 * p(good | L) = 0.9 and p(good | R) = 0.2, and the good state pays 10.
 *
 * The action values are Q(L) = -1 + 9 = 8 and Q(R) = 0 + 2 = 2, so the
 * expected total reward is 0.6 * 8 + 0.4 * 2 = 5.6.
 */
namespace two_action {

const DiscreteKey A(0, 2), S(1, 2);
const size_t L = 0, R = 1, good = 0, bad = 1;

/// The policy p(action), lifted to (p, 0).
inline SemiringDiscreteFactor policy() {
  return SemiringDiscreteFactor(DecisionTreeFactor(A, "0.6 0.4"));
}

/// The transition p(state | action), lifted to (p, 0).
inline SemiringDiscreteFactor transition() {
  return SemiringDiscreteFactor(DecisionTreeFactor(A & S, "0.9 0.1 0.2 0.8"));
}

/// The reward for the action, lifted to (1, r).
inline SemiringDiscreteFactor actionReward() {
  return SemiringDiscreteFactor::Reward(DecisionTreeFactor(A, "-1 0"));
}

/// The reward for the next state, lifted to (1, r).
inline SemiringDiscreteFactor stateReward() {
  return SemiringDiscreteFactor::Reward(DecisionTreeFactor(S, "10 0"));
}

/// An assignment to the action and the next state.
inline DiscreteValues assignment(size_t action, size_t state) {
  return {{A.first, action}, {S.first, state}};
}

}  // namespace two_action

}  // namespace gtsam
