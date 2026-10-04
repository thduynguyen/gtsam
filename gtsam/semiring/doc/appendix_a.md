# Appendix A: The GTSAM implementation

This appendix describes how the `gtsam/semiring` module is built: its classes,
how it plugs into GTSAM's elimination machinery, how the two factor families
store and eliminate their entries, and the tests that check the laws of
[Chapter 2](chapter02.md). It is meant for readers who want to read the
source, extend the module, or check what exactly a method returns.

The short version: the module adds **one abstract factor type with three
operations**, multiply, eliminate and expectation, and **one elimination
function** written against that interface. Everything else, the orderings,
sequential and multifrontal elimination, Bayes nets and Bayes trees, is
GTSAM's existing generic code.

## 1. The classes

| Class | Role |
|---|---|
| `SemiringFactor` | abstract factor with a probability and a value channel; declares `multiply`, `eliminate`, `sum` and `expectation` |
| `SemiringConditional` | a factor in conditional form; its value channel is the surprise |
| `SemiringDiscreteFactor`, `SemiringDiscreteConditional` | the table-backed family, stored as $(p, w)$ |
| `SemiringGaussianFactor`, `SemiringGaussianConditional` | the linear-Gaussian family, stored as $(\ell, v)$ |
| `SemiringFactorGraph` | the factor graph; adds `product` and `expectation` |
| `EliminateSemiring` | the elimination function: multiply, sum out, divide |
| `SemiringBayesNet`, `SemiringBayesTree`, `SemiringEliminationTree`, `SemiringJunctionTree` | the results and intermediate structures of elimination; thin instantiations of GTSAM's templates |

## 2. The factor interface

`SemiringFactor` derives from GTSAM's `Factor`, which holds the keys. It adds
the three operations elimination needs. The product $\otimes$ is `multiply`.
The sum $\oplus$ over a set of variables and the division $\oslash$ are
combined in `eliminate`, which returns both the conditional and the new
factor:

```cpp
class GTSAM_EXPORT SemiringFactor : public Factor {
 public:
  /// A conditional on the frontal variables and a factor on the separator.
  using EliminationResult =
      std::pair<std::shared_ptr<SemiringConditional>, shared_ptr>;

  /// Semiring product: probabilities multiply and values add.
  virtual shared_ptr multiply(const SemiringFactor& other) const = 0;

  /// Eliminate the frontal variables: the conditional, whose value channel is
  /// the surprise, and the semiring sum, whose value channel is E[v | separator].
  virtual EliminationResult eliminate(const Ordering& frontalKeys) const = 0;

  /// Semiring sum over the frontal variables, a new factor on the separator.
  shared_ptr sum(const Ordering& frontalKeys) const;

  /// Expected value E[v] under the normalized probability channel.
  virtual double expectation() const = 0;
};
```

`sum` is `eliminate(...).second`. The operations are virtual, so the
elimination function does not need to know which family it is working on.

## 3. The conditional

GTSAM's conditionals derive from their factor type and from the template
`Conditional<FACTOR, DERIVED>`, which records how many of the keys are frontal
variables. `SemiringConditional` follows the pattern:

```cpp
class GTSAM_EXPORT SemiringConditional
    : public SemiringFactor,
      public Conditional<SemiringFactor, SemiringConditional> {
 protected:
  /// Both channels of the conditional, stored as a factor.
  SemiringFactor::shared_ptr factor_;
  ...
};
```

A conditional *holds* a factor of its family, with both channels, and forwards
`multiply`, `eliminate` and `expectation` to it. A conditional is therefore a
semiring factor in its own right. That is what lets the conditionals of a
Bayes net be multiplied and eliminated again, which GTSAM does when it computes
marginals from a Bayes tree.

The two concrete conditionals add typed accessors:

| Method | `SemiringDiscreteConditional` | `SemiringGaussianConditional` |
|---|---|---|
| the probability channel | `probability()`, a `DiscreteConditional` | `conditional()`, a `GaussianConditional` |
| the value channel | `surprise()`, a `DecisionTreeFactor` | `surprise()`, a `HessianFactor`, or `surprise(values)` |

## 4. Plugging into elimination

GTSAM's elimination algorithms are templates over the graph type. A graph
type opts in by deriving from `EliminateableFactorGraph` and specializing
`EliminationTraits`, which names the related types and a default elimination
function:

```cpp
template <>
struct EliminationTraits<SemiringFactorGraph> {
  typedef SemiringFactor FactorType;
  typedef SemiringFactorGraph FactorGraphType;
  typedef SemiringConditional ConditionalType;
  typedef SemiringBayesNet BayesNetType;
  typedef SemiringEliminationTree EliminationTreeType;
  typedef SemiringBayesTree BayesTreeType;
  typedef SemiringJunctionTree JunctionTreeType;

  static std::pair<std::shared_ptr<ConditionalType>,
                   std::shared_ptr<FactorType>>
  DefaultEliminate(const FactorGraphType& factors, const Ordering& keys) {
    return EliminateSemiring(factors, keys);
  }
  ...
};
```

The elimination function receives the factors of one bucket and the variables
to eliminate, and is the three steps of Chapter 1 in two lines:

```cpp
std::pair<std::shared_ptr<SemiringConditional>, std::shared_ptr<SemiringFactor>>
EliminateSemiring(const SemiringFactorGraph& factors,
                  const Ordering& frontalKeys) {
  const SemiringFactor::shared_ptr product = factors.product();   // multiply
  return product->eliminate(frontalKeys);                // sum out and divide
}
```

With this in place `SemiringFactorGraph` inherits `eliminateSequential`,
`eliminatePartialSequential`, `eliminateMultifrontal` and the other methods of
`EliminateableFactorGraph`, with any ordering, exactly as `GaussianFactorGraph`
and `DiscreteFactorGraph` do.

**The constant at the root.** When the last variable is eliminated, the new
factor has no variables left. Its value is the expected return $J$. GTSAM's
elimination discards factors without variables, so
`SemiringFactorGraph::expectation` runs sequential elimination with a wrapper
around `EliminateSemiring` that collects them:

```cpp
double SemiringFactorGraph::expectation(const Ordering& ordering) const {
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
```

The same mechanism, a custom elimination function passed to
`eliminateSequential`, is how a different rule for some variables could be
added, for example the maximum over action variables of
[Chapter 4](chapter04.md).

## 5. The discrete family

A `SemiringDiscreteFactor` stores the pair in the form $(p, w)$ of Chapter 1,
Section 5: two `DecisionTreeFactor` tables on the same keys, the probability
and the weighted value $w = p\, v$.

| Operation | What it does to the two tables |
|---|---|
| lift a probability table $f$ | $(f,\; 0)$: constructor `SemiringDiscreteFactor(f)` |
| lift a reward table $r$ | $(1,\; r)$: `SemiringDiscreteFactor::Reward(r)` |
| product, `operator*` | $(p_1 p_2,\;\; p_1 w_2 + p_2 w_1)$ |
| sum over a variable | both tables are summed, as in ordinary sum-product |
| division, `operator/` | $\left(\dfrac{p}{p_S},\;\; \dfrac{w\, p_S - p\, w_S}{p_S^2}\right)$, with $0 / 0 = 0$ |
| read the value | `value()` returns the table $w / p$ |

`eliminate` multiplies nothing itself (the product was formed by
`EliminateSemiring`); it sums both tables over the frontal variables and
divides the product by the sum. `evaluate(values)` returns the pair $(p, v)$ of
one entry, and `probability()` and `weightedValue()` return the stored tables.

## 6. The Gaussian family

A `SemiringGaussianFactor` stores the pair in the log-dual form $(\ell, v)$ of
[Chapter 2](chapter02.md), Section 6:

- the probability channel is a `GaussianFactorGraph`, whose error is
  $-\ell$ up to a constant;
- the value channel is a `HessianFactor`, read as a general quadratic
  function. It may be indefinite, because a value is a reward of any sign and
  is never factorized.

| Operation | What it does |
|---|---|
| lift a Gaussian factor | $(\ell,\; 0)$: constructor `SemiringGaussianFactor(factor)` |
| lift a quadratic | `SemiringGaussianFactor::Reward(q)`, value $+$ the error of `q`, or `::Cost(q)`, value $-$ the error of `q` |
| product | the Gaussian factors are concatenated and the quadratics are added |
| eliminate | see below |

**Eliminating a variable** $x$ with separator $S$ works on the two channels in
turn.

1. *Probability channel.* Ordinary Gaussian elimination
   (`EliminatePreferCholesky`) of the Gaussian factors gives the conditional
   $p(x \mid S)$ and a new Gaussian factor on $S$. The conditional is an
   affine function of the separator plus noise,

   $$x = K\, S + k + W e, \qquad e \sim N(0, I).$$

2. *Value channel.* The value $v(x, S)$ is a quadratic. Its expectation under
   the conditional is obtained by substituting the mean $K S + k$ for $x$ and
   adding a constant for the noise, the trace of the $x$-block of the quadratic
   times the conditional covariance $W W^\top$. This is the rule "substitute
   the mean, add the variance" of Chapter 1, Section 8, for vectors. The
   result is the quadratic $\bar v(S)$ of the new factor.

3. *Division.* The value channel of the conditional is the difference of the
   two quadratics, $v(x, S) - \bar v(S)$: the surprise.

A constrained noise model, a hard equality such as deterministic dynamics or a
deterministic policy $u = -K x$, is the case $W = 0$: the substitution is
exact and the constant vanishes. A variable that appears only in the value
channel cannot be eliminated, since there is no distribution to average it
under, and `eliminate` throws.

Normalization constants of the Gaussian densities are not tracked, as
elsewhere in GTSAM. Expected values do not depend on them.

## 7. The tests of the laws

The unit tests check the algebraic laws of Chapter 2 directly, on a small
fixture with one decision (`tests/TwoActionExample.h`), and the results of
elimination against independent computations.

| Law or result | Test |
|---|---|
| the product is commutative and associative | `SemiringDiscreteFactor.ProductLaws` |
| the product distributes over the sum (Chapter 2, Axiom 3) | `SemiringDiscreteFactor.Distributivity` |
| division inverts the product: conditional $\otimes$ marginal $=$ joint | `SemiringDiscreteFactor.Divide`, `SemiringDiscreteConditional.MultiplyRecoversJoint` |
| the normalization invariant $\bigoplus_x c(x \mid S) = (1, 0)$ | `SemiringDiscreteConditional.Normalized`, `SemiringGaussianConditional.Normalized` |
| the product of all conditionals has expectation zero | `SemiringFactorGraph.BayesNetIsCentered` |
| the conditional of an action holds the policy and the advantage | `SemiringDiscreteConditional.PolicyAndAdvantage`, `SemiringGaussianConditional.PolicyAndAdvantage` |
| impossible outcomes and zero total probability are handled | `SemiringDiscreteFactor.ImpossibleOutcomes`, `.ZeroTotalProbability` |
| expected return and value functions of a tabular MDP | `SemiringFactorGraph.ExpectedTotalReward`, `.ValueFunction`, against brute-force enumeration |
| expected cost and value functions of a linear-quadratic problem | `SemiringFactorGraph.ExpectedTotalCost`, `.QuadraticValueFunction`, `.QuadraticAdvantage`, against the Lyapunov recursion |
| sequential and multifrontal elimination agree | `SemiringFactorGraph.EliminateMultifrontal`, `.MultifrontalMarginal`, `.GaussianMultifrontal` |
| the two families cannot be mixed | `SemiringFactorGraph.MixedFamilies` |

The Python tests in `python/gtsam/tests/test_SemiringFactorGraph.py` check
every number of the worked examples of Chapters 1, 4 and 6. Run the C++ tests
with the target `check.semiring`.

## 8. The Python wrapper

The classes are wrapped in `gtsam/semiring/semiring.i`, for Python only. The
names and methods are those of the C++ classes:

```python
from gtsam import SemiringFactorGraph
from gtsam import SemiringDiscreteFactor, SemiringGaussianFactor

graph = SemiringFactorGraph()
graph.push_back(...)                       # lifted factors
graph.expectation()                        # J
bayesNet = graph.eliminateSequential(ordering)
bayesNet.at(i).probability()               # discrete: p(x | S)
bayesNet.at(i).surprise()                  # the value channel
bayesTree = graph.eliminateMultifrontal()
bayesTree.marginalFactor(key)              # a factor on one variable
conditional, newFactor = factor.eliminate(ordering)   # one step, by hand
```

The elimination function itself is available as `gtsam.EliminateSemiring`.
Passing a custom elimination function is possible in C++ only.

## 9. Extending the module

**A new factor family** (for example, factors on a mixture of discrete and
continuous variables) derives from `SemiringFactor` and implements `multiply`,
`eliminate`, `expectation` and `equals`, with a conditional class that derives
from `SemiringConditional`. Nothing else changes: `EliminateSemiring` and the
graph work through the base interface.

**Another semiring of the family.** The other members of
[Chapter 2](chapter02.md) are not implemented in C++. The companion notebooks
run them with a generic routine in numpy. In the module they would be:

| Semiring | What would be needed |
|---|---|
| tilted, or soft maximum | a discrete family storing $(p, m)$ with $m = p\, e^{\kappa v}$; both tables follow the sum-product rules, so it is two `DecisionTreeFactor`s with independent products and sums |
| maximum over the actions, average over the states | a custom elimination function that, for an action key, replaces the sum over the frontal variable by a maximum of the value channel and returns the maximizing action as a deterministic conditional |
| second-order ([Chapter 5](chapter05.md)) | a family storing $(p, w, \dot p, \dot w)$ per parameter; useful for few parameters only |

These are listed as limitations in the module's README.

---

Previous: [Chapter 5: Gradients by elimination: the two-stage framework](chapter05.md).
Next: [Appendix B: Notation](appendix_b.md).
