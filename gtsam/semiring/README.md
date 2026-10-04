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

The documentation is a book, best read online at
**https://thduynguyen.github.io/gtsam/**, where the maths, figures and
collapsible notes are typeset properly.

The book maps the algorithms of optimal control and RL onto one framework with
two stages: an inner elimination on the semiring factor graph that evaluates
the current policy, and an outer step that improves it. Chapter 5 defines the
framework, and Appendix C lists every algorithm of the book with its choices.

The sources are in [`doc/`](doc), written in [MyST Markdown](https://mystmd.org).
GitHub's own preview of those files shows the text but not the book's
formatting. To build the book locally, run `myst build --html` or `myst start`
in `doc/`.

Each chapter has a companion notebook that runs its examples and checks every
number quoted in the chapter. The notebooks open in Google Colab from the
badge at the top of each chapter; those that use this module install a GTSAM
wheel that contains it in their first cell.

<!-- chapters:begin -->
**Part I: Exact elimination on decision graphs**

- 1. [MDPs as factor graphs: evaluating a policy by variable elimination](doc/chapter01.md) ([notebook](doc/chapter01_examples.ipynb))
- 2. [The semiring family](doc/chapter02.md) ([notebook](doc/chapter02_examples.ipynb))
- 3. [Infinite horizon and discounting](doc/chapter03.md) ([notebook](doc/chapter03_examples.ipynb))
- 4. [Decision nodes and elimination order](doc/chapter04.md) ([notebook](doc/chapter04_examples.ipynb))
- 5. [Gradients by elimination: the two-stage framework](doc/chapter05.md) ([notebook](doc/chapter05_examples.ipynb))

**Appendices**

- [The GTSAM implementation](doc/appendix_a.md)
- [Notation](doc/appendix_b.md)
<!-- chapters:end -->

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
- Python: `python/gtsam/tests/test_SemiringFactorGraph.py`. It checks the
  numbers of the worked examples that use the module (Chapters 1, 4 and 6).
- The companion notebooks of the chapters assert every number they quote.

## Limitations

- **Expectation semiring only.** The built-in elimination evaluates a given
  policy. The maximum over action variables of Chapter 4, which finds the best
  policy, is not a built-in elimination function yet; it can be written by
  hand with the factor interface, as shown there. The other semirings of
  Chapter 2 are run in the notebooks, in numpy.
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
