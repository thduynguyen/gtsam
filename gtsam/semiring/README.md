# Semiring factor graphs

This module lets a factor graph represent a Markov decision process (MDP) and
compute expected rewards on it with GTSAM's ordinary variable elimination.

Every factor entry carries two numbers, a probability and a value, and the
product and sum of elimination are replaced by those of the *expectation
semiring*. Elimination then produces the quantities that optimal control and
reinforcement learning (RL) are built on: the expected return of a policy, its
value functions, and its advantages.

The aim is to express optimal control and RL as operations on factor graphs,
for readers who know factor graphs from SLAM.

## Documentation

The documentation is a series of chapters in [`doc/`](doc). Each chapter has a
companion notebook that runs its examples. The notebooks open in Google Colab,
where their first cell installs a GTSAM wheel that contains this module.

| Chapter | Content | Run the examples |
|---|---|---|
| [1. MDPs as factor graphs: evaluating a policy by variable elimination](doc/chapter01.md) | An MDP as a factor graph; why ordinary elimination cannot evaluate it; semiring factors and their operators; the correspondence between RL quantities and elimination; a discrete and a continuous (LQR) worked example; how to use the module. | [notebook](doc/chapter01_examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thduynguyen/gtsam/blob/feature/semiringfactor/gtsam/semiring/doc/chapter01_examples.ipynb) |
| [2. Finding the best policy in one pass](doc/chapter02.md) | A teaser for policy optimization: for tabular MDPs and for LQR, replacing the expectation over actions by a maximum yields the best policy in one backward pass. For LQR this is, line by line, the Riccati recursion of classic control. | [notebook](doc/chapter02_examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thduynguyen/gtsam/blob/feature/semiringfactor/gtsam/semiring/doc/chapter02_examples.ipynb) |

Planned chapters:

- **Optimal control with known dynamics.** A generic two-stage framework on
  semiring factor graphs, an inner elimination that evaluates a policy inside
  an outer optimization that improves it, with the algorithms of control mapped
  to optimization on the graph.
- **Reinforcement learning.** The case where the dynamics are unknown or only
  available through a simulator, with RL algorithms mapped to the same two
  stages, the exact sums and integrals replaced by sample approximations.

## Classes

| Class | Role |
|---|---|
| `SemiringFactor` | Abstract factor with a probability and a value channel; defines `multiply`, `sum`, `eliminate` and `expectation`. |
| `SemiringConditional` | A factor in conditional form; its value channel is the surprise, the value minus its conditional mean. |
| `SemiringDiscreteFactor`, `SemiringDiscreteConditional` | The table-backed family, for discrete states and actions. |
| `SemiringGaussianFactor`, `SemiringGaussianConditional` | The linear-Gaussian family, with quadratic values. |
| `SemiringFactorGraph` | Factor graph with `eliminateSequential`, `eliminatePartialSequential`, `eliminateMultifrontal` and `expectation`. |
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

[Chapter 1](doc/chapter01.md) explains what the conditionals of the Bayes net
contain, and works through two larger examples in its Sections 7 and 8.

## Tests

- C++: `gtsam/semiring/tests/`, run with the `check.semiring` target.
- Python: `python/gtsam/tests/test_SemiringFactorGraph.py`. It checks every
  number in the worked examples of the chapters.

## Limitations

- **Expectation semiring only.** The built-in elimination evaluates a given
  policy. The maximum over action variables of Chapter 2, which finds the best
  policy, is not a built-in elimination function yet; it can be written by
  hand with the factor interface, as shown there.
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
