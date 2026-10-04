# Appendix A: The GTSAM implementation

:::{div}
:class: in-progress
**This book is a work in progress.** It is still being written and revised: its content is incomplete and may contain errors.
:::

This appendix describes how the `gtsam/semiring` module is built: its classes,
how it plugs into GTSAM's elimination machinery, how the two factor families
store and eliminate their entries, and the tests that check the laws of
[Chapter 2](chapter02.md). It is meant for readers who want to read the
source, extend the module, or check what exactly a method returns.

The short version: the module adds **one abstract factor type with three
operations**, multiply, eliminate and expectation, **a rule for how a variable
is summed out**, and **one elimination function** written against that
interface. Everything else, the orderings,
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
| `SemiringSum` | the rule by which a variable is summed out: average, maximum or tilted mean |
| `SemiringRules` | the rule of each variable of a graph |
| `EliminateSemiring`, `EliminateSemiringWith` | the elimination function: multiply, sum out, divide; the second takes the rules |
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

  /// Eliminate the frontal variables with the given rule: the conditional,
  /// and the semiring sum, a new factor on the separator.
  virtual EliminationResult eliminate(const Ordering& frontalKeys,
                                      const SemiringSum& sum) const = 0;

  /// Eliminate the frontal variables by averaging, the expectation semiring.
  EliminationResult eliminate(const Ordering& frontalKeys) const;

  /// Semiring sum over the frontal variables, a new factor on the separator.
  shared_ptr sum(const Ordering& frontalKeys,
                 const SemiringSum& sum = SemiringSum()) const;

  /// Expected value E[v] under the normalized probability channel.
  virtual double expectation() const = 0;
};
```

`sum` is `eliminate(...).second`. The operations are virtual, so the
elimination function does not need to know which family it is working on.

**The rule for the sum.** All the semirings of [Chapter 2](chapter02.md) that
carry a value share one product, so the product is not a parameter. The sum
is: a `SemiringSum` says how the values of the outcomes of the eliminated
variable are merged.

| Rule | Value of the new factor | Value channel of the conditional |
|---|---|---|
| `SemiringSum::Average()`, the default | $\mathbb{E}[v \mid S]$ | the surprise: advantage or TD residual |
| `SemiringSum::Maximum()` | $\max_x v$, over the outcomes with nonzero probability | the regret |
| `SemiringSum::Tilted(kappa)` | $\frac{1}{\kappa} \log \sum_x p(x \mid S)\, e^{\kappa v}$ | the soft advantage |
| `SemiringSum::SoftMaximum(eta)` | the same, with $\kappa = 1 / \eta$ | the soft advantage |

In every case the probabilities of the outcomes are added, and the value
channel of the conditional is $v$ minus the merged value.

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

**A rule per variable.** The same mechanism, a custom elimination function
passed to `eliminateSequential`, gives each variable its own rule. A
`SemiringRules` object maps keys to rules, and `EliminateSemiringWith(rules)`
is the elimination function that looks the rule up:

```cpp
SemiringFactorGraph::Eliminate EliminateSemiringWith(const SemiringRules& rules) {
  return [rules](const SemiringFactorGraph& factors, const Ordering& frontalKeys) {
    const SemiringFactor::shared_ptr product = factors.product();
    return product->eliminate(frontalKeys, rules.common(frontalKeys));
  };
}
```

`SemiringFactorGraph` has overloads of `eliminateSequential`,
`eliminatePartialSequential` and `expectation` that take the rules directly:

```cpp
SemiringRules rules;
rules.setAll({U(0), U(1)}, SemiringSum::Maximum());   // the actions
graph.expectation(ordering, rules);                   // the best expected return
graph.eliminateSequential(ordering, rules);
```

Two consequences of mixing rules, both from [Chapter 4](chapter04.md):

- **The order is no longer free.** The default COLAMD ordering is valid only
  when all variables share one rule. With different rules the ordering must be
  given: for a Markov decision process, backward in time.
- **Variables eliminated together must share a rule.** `rules.common` throws
  otherwise. Sequential elimination takes one variable at a time and is always
  fine. Multifrontal elimination, which groups variables into cliques, is not
  supported with mixed rules.

## 5. The discrete family

A `SemiringDiscreteFactor` stores the pair in the form $(p, w)$ of Chapter 1,
Section 5: two `DecisionTreeFactor` tables on the same keys, the probability
and the weighted value $w = p\, v$.

| Operation | What it does to the two tables |
|---|---|
| lift a probability table $f$ | $(f,\; 0)$: constructor `SemiringDiscreteFactor(f)` |
| lift a reward table $r$ | $(1,\; r)$: `SemiringDiscreteFactor::Reward(r)` |
| product, `operator*` | $(p_1 p_2,\;\; p_1 w_2 + p_2 w_1)$ |
| sum over a variable, by average | both tables are summed, as in ordinary sum-product |
| sum over a variable, by maximum | the probability table is summed; the value is the largest $w / p$ among the entries with $p > 0$ |
| sum over a variable, by tilted mean | the probability table is summed; the value is the tilted mean of $w / p$, computed relative to its extreme value so that no exponential overflows |
| division, `operator/` | $\left(\dfrac{p}{p_S},\;\; \dfrac{w\, p_S - p\, w_S}{p_S^2}\right)$, with $0 / 0 = 0$ |
| read the value | `value()` returns the table $w / p$ |

`eliminate` multiplies nothing itself (the product was formed by
`EliminateSemiring`); it sums out the frontal variables by the given rule and
divides the product by the result. `evaluate(values)` returns the pair
$(p, v)$ of one entry, and `probability()` and `weightedValue()` return the
stored tables.

A discrete conditional has two more methods, for the cases where its frontal
variable is a choice:

| Method | Returns |
|---|---|
| `greedy()` | a `DiscreteConditional` with probability one on the frontal value of largest surprise, shared equally among ties: the greedy policy |
| `tilted(kappa)` | the conditional reweighted by $e^{\kappa \cdot \text{surprise}}$ and normalized: the soft policy |

Because the value channel is stored weighted by the probability, a frontal
value with probability zero has no stored surprise, and `greedy()` ranks only
the values with nonzero probability. To choose among all actions, eliminate
the action from a bucket without a policy factor, with the maximum rule.

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

1. *Probability channel.* Ordinary Gaussian elimination (`EliminateQR`) of
   the Gaussian factors gives the conditional $p(x \mid S)$ and a new Gaussian
   factor on $S$. QR is used because it keeps the rows of the factors: when a
   normalized conditional such as the dynamics is eliminated, the new factor
   has no rows left, and is dropped as carrying no information. With Cholesky
   it would be a matrix of rounding errors. The conditional is an
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

**With the tilted rule,** step 2 computes the tilted mean of the quadratic
value in place of its mean. Along $x = \mu + W e$, with $\mu = K S + k$, the
value is $v(\mu) + b^\top e + \tfrac{1}{2} e^\top M e$, where $b$ is affine in
the separator, and its tilted mean is

$$v(\mu) + \frac{\kappa}{2}\, b^\top N^{-1} b - \frac{1}{2 \kappa} \log \det N,
\qquad N = I - \kappa M,$$

again a quadratic in the separator. It exists only if $N$ is positive
definite; otherwise the tilt is too strong for the noise, and `eliminate`
throws `std::invalid_argument` ([Chapter 8](chapter08.md) calls this the
breakdown). The conditional is not changed by the tilt.

**With the maximum rule,** the variable must have no density: no Gaussian
factor may involve it, as for an action without a policy factor. The value is
a quadratic $\tfrac{1}{2} x^\top G_{xx}\, x + x^\top G_{xS}\, S - g_x^\top x + \dots$,
and its maximizer solves a linear system,

$$G_{xx}\, x + G_{xS}\, S - g_x = 0 \quad\Longrightarrow\quad x = K S + k.$$

$G_{xx}$ must be negative definite, that is, the value strictly concave in
$x$; otherwise there is no maximum and `eliminate` throws. The conditional is
the maximizer as a deterministic `GaussianConditional` with a constrained
noise model, its value channel is the regret, and the new factor is the value
with the maximizer substituted. For linear dynamics and quadratic rewards
this is one step of the Riccati recursion ([Chapter 6](chapter06.md)).

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
| the maximum and the tilted mean of one variable, their limits, and outcomes of zero probability | `SemiringDiscreteFactor.Maximum`, `.MaximumSkipsImpossible`, `.Tilted`, `.TiltedLimits` |
| the invariant $\bigoplus_x c = \mathbf{1}$ for every rule | `SemiringDiscreteConditional.NormalizedForEveryRule` |
| average at the states, maximum at the actions: dynamic programming | `SemiringFactorGraph.DynamicProgramming` |
| the maximum and a tilt at every variable; the soft maximum at the actions | `SemiringFactorGraph.MaximumEverywhere`, `.TiltedEverywhere`, `.SoftMaximumAtActions` |
| the Riccati recursion, and its risk-sensitive version with a tilt at the states | `SemiringFactorGraph.Riccati`, `.RiskSensitiveRiccati`, `.TiltBreakdown` |

The tests of the rules are in `tests/testSemiringSum.cpp`. The Python tests
in `python/gtsam/tests/test_SemiringFactorGraph.py` check the numbers of the
worked examples of Chapters 1, 2, 4, 6 and 8. Run the C++ tests
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

rules = SemiringRules()                    # a rule per variable
rules.setAll(actionKeys, SemiringSum.Maximum())
graph.expectation(ordering, rules)         # the best expected return
graph.eliminateSequential(ordering, rules)
conditional, newFactor = factor.eliminate(ordering, SemiringSum.Tilted(0.5))
conditional.greedy()                       # discrete: the greedy policy
conditional.tilted(0.5)                    # discrete: the soft policy
```

The elimination functions are available as `gtsam.EliminateSemiring` and
`gtsam.EliminateSemiringWith(rules)`.

## 9. Extending the module

**A new factor family** (for example, factors on a mixture of discrete and
continuous variables) derives from `SemiringFactor` and implements `multiply`,
`eliminate`, `expectation` and `equals`, with a conditional class that derives
from `SemiringConditional`. Nothing else changes: `EliminateSemiring` and the
graph work through the base interface.

**What is not implemented.**

| Feature | What would be needed |
|---|---|
| the second-order semiring of [Chapter 5](chapter05.md), for gradients in one pass | a family storing $(p, w, \dot p, \dot w)$ per parameter; useful for few parameters only. The notebook of Chapter 5 runs it in numpy |
| a log-domain discrete family | tables of $(\ell, v)$, to avoid the underflow of long products of small probabilities |
| multifrontal elimination with different rules | a check that every clique groups variables of one rule, and an ordering that respects the constraint of Chapter 4 |
| the maximum of a Gaussian variable that has a density, and the soft maximum of one that has none | a convention for the probability channel in each case |

These are listed as limitations in the module's README.

---

Previous: [Chapter 26: Robotics case studies](chapter26.md).
Next: [Appendix B: Notation](appendix_b.md).
