# Semiring factor graphs

This module lets a factor graph represent a Markov decision process (MDP) and
compute expected rewards on it with GTSAM's ordinary variable elimination.

Every factor entry carries two numbers, a probability and a value, and the
product and sum of elimination are replaced by those of the *expectation
semiring*. Elimination then produces the quantities that optimal control and
reinforcement learning (RL) are built on: the expected return of a policy, its
value functions, and its advantages.

Each variable can also be summed out by a maximum or by a tilted mean in
place of the average. With the maximum at the action variables, elimination
is dynamic programming, and for linear-Gaussian problems the Riccati
recursion; with a tilt it gives soft and risk-sensitive control.

The aim is to express optimal control and RL as operations on factor graphs,
for readers who know factor graphs from SLAM.

## Documentation

The module is documented by a book, *Semiring factor graphs*, that maps the
algorithms of optimal control and RL onto one framework with two stages: an
inner elimination on the semiring factor graph that evaluates the current
policy, and an outer step that improves it. The book and its companion
notebooks are not part of this public repository; this branch exists to build
the GTSAM wheels that the notebooks install.

## Copyright and license

The module's code is part of GTSAM, under its BSD license. The book is
Copyright © 2026 Duy Ta and Frank Dellaert, all rights reserved.

## Classes

| Class | Role |
|---|---|
| `SemiringFactor` | Abstract factor with a probability and a value channel; defines `multiply`, `sum`, `eliminate` and `expectation`. |
| `SemiringConditional` | A factor in conditional form; its value channel is the surprise, the value minus its conditional mean. |
| `SemiringDiscreteFactor`, `SemiringDiscreteConditional` | The table-backed family, for discrete states and actions. |
| `SemiringGaussianFactor`, `SemiringGaussianConditional` | The linear-Gaussian family, with quadratic values. |
| `SemiringFactorGraph` | Factor graph with `eliminateSequential`, `eliminatePartialSequential`, `eliminateMultifrontal` and `expectation`. |
| `SemiringSum`, `SemiringRules` | How a variable is summed out: by the average (the default), the maximum, or a tilted mean, of which the soft maximum is a case; and the rule of each variable of a graph. |
| `SemiringBayesNet`, `SemiringBayesTree` | Results of sequential and multifrontal elimination. |

## Quick start

One decision: an action L or R chosen with probability (0.6, 0.4), where L
costs 1, then a good or bad outcome, where good pays 10.

```python
from gtsam import DecisionTreeFactor
from gtsam import SemiringDiscreteFactor, SemiringFactorGraph

A, S = (0, 2), (1, 2)  # (key, cardinality) of the action and the outcome

graph = SemiringFactorGraph()
# Probability tables are lifted to (p, 0), reward tables to (1, r).
graph.push_back(SemiringDiscreteFactor(DecisionTreeFactor(A, "0.6 0.4")))
graph.push_back(SemiringDiscreteFactor.Reward(DecisionTreeFactor(A, "-1 0")))
graph.push_back(
    SemiringDiscreteFactor(DecisionTreeFactor([A, S], "0.9 0.1 0.2 0.8")))
graph.push_back(SemiringDiscreteFactor.Reward(DecisionTreeFactor(S, "10 0")))

graph.expectation()  # 5.6, the expected total reward

bayesNet = graph.eliminateSequential()
```

The best action in place of the average under the policy: sum the action out
by maximum, and the outcome, which is decided by chance, by the average. With
different rules the order must be given, the outcome first.

```python
from gtsam import Ordering, SemiringRules, SemiringSum

rules = SemiringRules()
rules.set(0, SemiringSum.Maximum())  # key 0 is the action

ordering = Ordering()
ordering.push_back(1)  # the outcome, by average
ordering.push_back(0)  # the action, by maximum
graph.expectation(ordering, rules)  # 8.0, the value of the better action
```

## Tests

- C++: `gtsam/semiring/tests/`, run with the `check.semiring` target.
- Python: `python/gtsam/tests/test_SemiringFactorGraph.py`.

## Limitations

- **One rule per elimination step.** Each variable is summed out by its own
  rule, but variables eliminated together must share one. With different rules
  in a graph the ordering must be given explicitly, and multifrontal
  elimination is not supported.
- **Gaussian maximum and tilt.** The maximum applies to a variable without a
  density, such as an action without a policy factor, and needs a value that
  is concave in it. The tilted mean applies to a variable with a density, and
  fails when the tilt is too strong for the noise.
- **No gradients in one pass.** The second-order semiring that carries a
  gradient is not implemented; the module gives the gradient through the
  advantages and the marginals.
- **Discrete keys need at least two values.** A table on a variable with a
  single value is rejected.
- **One family per graph.** Discrete and Gaussian factors cannot be mixed;
  combining them throws.
- **Known models.** The dynamics must be given as factors. Sampling-based RL
  replaces the exact sums with estimates, which this module does not do; it
  provides the exact reference those methods approximate.
- **Linear-Gaussian only for continuous variables.** Nonlinear dynamics would
  have to be linearized first.
- **Numerical range.** Discrete probabilities are stored directly, not as logs,
  so a long chain of unnormalized factors can underflow. Gaussian normalization
  constants are not tracked, as elsewhere in GTSAM; expected values are
  unaffected.
- **Wrappers.** The classes are wrapped for Python, not MATLAB.
