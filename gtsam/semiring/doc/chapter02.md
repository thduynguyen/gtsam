# Chapter 2: The semiring family

:::{div}
:class: in-progress
**This book is a work in progress.** It is still being written and revised: its content is incomplete and may contain errors.
:::

[Chapter 1](chapter01.md) changed two things: what a factor entry holds, a
pair $(p, v)$ in place of one number, and how two entries are multiplied and
added. Variable elimination itself was not touched.

That is a general fact. Elimination only multiplies factors and adds over the
values of a variable, so it runs on any kind of entry that has a product and a
sum with the right properties, a *semiring*. Changing the semiring changes the
question that the same algorithm answers on the same graph.

This chapter covers the semirings that control and RL use:

- the five members of the family, and what each computes;
- the properties elimination needs, stated as axioms;
- division, which is what turns results into conditionals, and the invariant
  every conditional satisfies;
- the logarithmic form of the pair, which the Gaussian factors of the module
  use.

**Run the examples.** The code of this chapter's examples is in the companion
notebook [chapter02_examples.ipynb](chapter02_examples.ipynb), which runs
online:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thduynguyen/gtsam/blob/feature/semiringfactor/gtsam/semiring/doc/chapter02_examples.ipynb)

## 1. One algorithm, five questions

Elimination of a variable $x$ with separator $S$ has three steps, as in
Chapter 1. Written with a general product $\otimes$, sum $\oplus$ and division
$\oslash$:

$$\psi(x, S) = \bigotimes_i f_i \;\;\text{(multiply)}, \qquad
\phi(S) = \bigoplus_x \psi(x, S) \;\;\text{(add over } x\text{)}, \qquad
c(x \mid S) = \psi \oslash \phi \;\;\text{(divide)}.$$

The notebook implements the first two steps in about thirty lines of numpy and
runs them, unchanged, on the track example of Chapter 1 (three cells, two
moves, a coin-flip policy) with five semirings.

Each semiring answers a question about the trajectories $\tau$ of the graph.
Recall from Chapter 1 that $p(\tau)$ is the probability of a trajectory and
$R(\tau)$ its return, the sum of its rewards. Two of the five need a word of
explanation first.

- **Tilted.** Between the average of the returns and their maximum lies a
  family of in-between averages. A *tilted* average counts every trajectory
  with its probability **and** with an extra weight $e^{\kappa R(\tau)}$ that
  grows with its return. The number $\kappa$ is called the **tilt**. With
  $\kappa = 0$ there is no extra weight, and the result is the plain average.
  With $\kappa > 0$ the trajectories with a high return count more, and with
  $\kappa < 0$ those with a low return count more.
- **Soft maximum.** The same rule, written with $\eta = 1 / \kappa$, a positive
  number called the *temperature*. Under this name it is used as a smooth
  stand-in for the maximum.

| Semiring | An entry holds | The question it answers | As a formula | Result on the track | In the literature |
|---|---|---|---|---|---|
| sum-product | a probability $p$ | What is the total probability of all trajectories? | $\sum_\tau p(\tau)$ | $1$ | [\[4\]](#ch02-aji2000), [\[5\]](#ch02-dechter1999), [\[6\]](#ch02-kschischang2001) |
| expectation | a pair $(p, v)$ | What return does the policy collect on average? | $\sum_\tau p(\tau)\, R(\tau)$ | $1.4$ | [\[7\]](#ch02-eisner2002), [\[8\]](#ch02-li2009) |
| max-sum | a value $v$ | What is the return of the best trajectory that can occur? | $\max_{\tau \,:\, p(\tau) > 0} R(\tau)$ | $9$ | [\[4\]](#ch02-aji2000), [\[5\]](#ch02-dechter1999), [\[9\]](#ch02-viterbi1967) |
| tilted, with tilt $\kappa$ | a pair $(p, v)$ | What is the average return, if lucky trajectories count more ($\kappa > 0$) or less ($\kappa < 0$)? | $\frac{1}{\kappa} \log \sum_\tau p(\tau)\, e^{\kappa R(\tau)}$ | $5.43$ for $\kappa = 0.5$, and $-0.28$ for $\kappa = -0.5$ | [\[10\]](#ch02-howard1972), [\[11\]](#ch02-jacobson1973), [\[12\]](#ch02-whittle1990) |
| soft maximum, with temperature $\eta$ | a pair $(p, v)$ | What is a smooth version of the best return? | $\eta \log \sum_\tau p(\tau)\, e^{R(\tau) / \eta}$ | $5.43$ for $\eta = 2$ | [\[13\]](#ch02-kappen2005), [\[14\]](#ch02-todorov2006), [\[15\]](#ch02-levine2018) |

**The last two rows are the same rule.** Substituting $\eta = 1 / \kappa$
turns one formula into the other, which is why $\eta = 2$ gives the same
number as $\kappa = 0.5$. They are listed separately because they have
different names in the literature and are used for different purposes, as
Section 2 explains. So the family has five names and four different rules.

The graph, the tables and the elimination order are the same in all five rows.
Only $\otimes$ and $\oplus$ differ. Section 2 defines them for each row. The
numbers in the last column refer to the [references](#chapter02-references) at
the end of the chapter.

## 2. The five semirings

A semiring is specified by four things: what an entry is, the product
$\otimes$, the sum $\oplus$, and how the terms of the MDP are lifted to
entries. It also has a zero $\mathbf{0}$, the entry of an impossible outcome,
and a one $\mathbf{1}$, the entry of a factor that changes nothing.

Of the five names in the table of Section 1, the first four have a
subsection each below. The fifth, the soft maximum, is the tilted semiring
with its parameter written differently, and has a short subsection of its
own.

### Sum-product

This is ordinary elimination, the one GTSAM uses for discrete factor graphs.

| | |
|---|---|
| entry | a probability $p$ |
| product | $p_1 \otimes p_2 = p_1\, p_2$ |
| sum | $p_1 \oplus p_2 = p_1 + p_2$ |
| zero, one | $0$, $1$ |
| a probability table $f$ becomes | $f$ |
| a reward table $r$ becomes | $1$: rewards are ignored |

Summing out every variable gives the total probability of all trajectories,

$$\bigoplus_\tau \bigotimes_i f_i = \sum_\tau p(\tau) = 1.$$

On a graph with unnormalized factors the same sum is the normalization constant
$Z$, and the conditionals form the Bayes net. This semiring knows nothing about
rewards.

### Expectation

This is the semiring of Chapter 1.

| | |
|---|---|
| entry | a pair $(p, v)$: probability and value |
| product | $(p_1, v_1) \otimes (p_2, v_2) = (p_1 p_2,\;\; v_1 + v_2)$ |
| sum | $(p_1, v_1) \oplus (p_2, v_2) = \left(p_1 + p_2,\;\; \dfrac{p_1 v_1 + p_2 v_2}{p_1 + p_2}\right)$ |
| zero, one | $(0, \cdot)$, $(1, 0)$ |
| a probability table $f$ becomes | $(f, 0)$ |
| a reward table $r$ becomes | $(1, r)$ |

The sum merges two exclusive outcomes of the eliminated variable: their
probabilities add, and their values are *averaged*, each weighted by its
probability. Summing out every variable gives $(1, J)$ with

$$J = \sum_\tau p(\tau)\, R(\tau) = 1.4 \quad \text{on the track}.$$

### Max-sum

Merge two outcomes by keeping the *better* one, where the expectation
semiring took their average. The probability can then be dropped, for the
following reason.

**Why the probability disappears.** Start from the pair $(p, v)$ of the
expectation semiring and change only the value of the sum, from the average to
the maximum:

$$(p_1, v_1) \otimes (p_2, v_2) = (p_1 p_2,\;\; v_1 + v_2),
\qquad
(p_1, v_1) \oplus (p_2, v_2) = \big(p_1 + p_2,\;\; \max(v_1, v_2)\big).$$

Look at what the value needs from the probability. In the expectation
semiring the probabilities were the *weights* of the average, so the value
could not be computed without them. Here neither rule for the value uses
them: values add, and the larger one is kept. Only one fact about the
probability still matters: an outcome with $p = 0$ cannot occur, and it must
not win the maximum.

That one fact can be stored in the value itself. Give an impossible outcome
the value $-\infty$: it never wins a maximum, and it stays $-\infty$ whatever
is added to it. A possible outcome keeps its value. With this convention the
probability carries no further information, and an entry is a single number:

| | |
|---|---|
| entry | a value $v$, with $-\infty$ for an outcome that cannot occur |
| product | $v_1 \otimes v_2 = v_1 + v_2$ |
| sum | $v_1 \oplus v_2 = \max(v_1, v_2)$ |
| zero, one | $-\infty$, $0$ |
| a probability table $f$ becomes | $0$ where $f > 0$, and $-\infty$ where $f = 0$ |
| a reward table $r$ becomes | $r$ |

A probability table is lifted by keeping only that one fact: a possible
outcome contributes $0$, which adds nothing to the value, and an impossible
one contributes $-\infty$. How probable a possible outcome is plays no role.
Eliminating every variable gives the return of the best trajectory among those
that can occur:

$$\bigoplus_\tau \bigotimes_i f_i = \max_{\tau \,:\, p(\tau) > 0} R(\tau) = 9
\quad \text{on the track}.$$

That trajectory starts in cell 1, moves Right into cell 2, and then moves
*Left* and slips, which keeps the robot at the charger without paying for a
move: $R = -1 + 0 + 10 = 9$.

This is **NOT** something the robot can achieve. The maximum was taken over
every variable, the states as well as the actions, as if the robot could
choose its start and choose to slip. An agent chooses only its actions.
[Chapter 4](chapter04.md) takes the maximum over the actions alone, and gets
$6.1$.

:::{dropdown} How does this relate to GTSAM's most probable assignment?
GTSAM finds a most probable assignment of a discrete graph with *max-product*:
entries are probabilities, the product is multiplication and the sum is the
maximum,

$$\max_\tau \prod_i f_i(\tau).$$

Taking logarithms turns the product into a sum and leaves the maximum alone, so
max-product on $f$ is max-sum on $\log f$:

$$\log \max_\tau \prod_i f_i = \max_\tau \sum_i \log f_i.$$

Nonlinear least squares is the continuous case: $\log f_i$ is minus a squared
error, and the maximum is the least-squares solution. So a SLAM optimizer is a
max-sum eliminator. The table above differs from it only in the lifting: a
reward enters as $r$ and a probability as $0$ or $-\infty$, where MAP
estimation would enter a probability as $\log f$.
:::

### Tilted

Keep the pair $(p, v)$ and the product of the expectation semiring, and change
how the values of two outcomes merge: use an average that leans toward the
larger value (for $\kappa > 0$) or toward the smaller one (for $\kappa < 0$).

| | |
|---|---|
| entry | a pair $(p, v)$: probability and value |
| product | $(p_1, v_1) \otimes (p_2, v_2) = (p_1 p_2,\;\; v_1 + v_2)$ |
| sum | $(p_1, v_1) \oplus (p_2, v_2) = \left(p_1 + p_2,\;\; \dfrac{1}{\kappa} \log \dfrac{p_1 e^{\kappa v_1} + p_2 e^{\kappa v_2}}{p_1 + p_2}\right)$ |
| zero, one | $(0, \cdot)$, $(1, 0)$ |
| a probability table $f$ becomes | $(f, 0)$ |
| a reward table $r$ becomes | $(1, r)$ |

Over all values of the eliminated variable $x$, the value of the new factor is
the **tilted mean** of $v$,

$$\bar v_\kappa(S) = \frac{1}{\kappa} \log \sum_x p(x \mid S)\, e^{\kappa\, v(x, S)}.$$

The number $\kappa$ is the *tilt*. The tilted mean always lies between the
smallest and the largest value, and $\kappa$ moves it between them:

$$\min_x v \;\;\xleftarrow{\;\kappa \to -\infty\;}\;\; \bar v_\kappa
\;\;\xrightarrow{\;\kappa \to 0\;}\;\; \sum_x p(x \mid S)\, v
\qquad\text{and}\qquad
\bar v_\kappa \;\;\xrightarrow{\;\kappa \to +\infty\;}\;\; \max_x v.$$

The minimum and maximum run over the outcomes with nonzero probability.

*A small example.* A fair coin pays 0 or 10. Its ordinary mean is 5. The
tilted mean is $\frac{1}{\kappa} \log\big(0.5\, e^{0} + 0.5\, e^{10 \kappa}\big)$:

| $\kappa$ | $-5$ | $-0.5$ | $-0.01$ | $0.01$ | $0.5$ | $5$ |
|---|---|---|---|---|---|---|
| tilted mean | $0.14$ | $1.37$ | $4.88$ | $5.13$ | $8.63$ | $9.86$ |

A small tilt gives nearly the mean. A positive tilt moves the result toward
the better outcome, 10, and a negative tilt toward the worse one, 0.

Eliminating every variable of the track gives the tilted mean of the return,
$\frac{1}{\kappa} \log \mathbb{E}[e^{\kappa R}]$:

| $\kappa$ | $-20$ | $-2$ | $-0.5$ | $-0.01$ | $0.01$ | $0.5$ | $2$ | $20$ |
|---|---|---|---|---|---|---|---|---|
| tilted mean of $R$ | $-1.85$ | $-0.93$ | $-0.28$ | $1.33$ | $1.47$ | $5.43$ | $7.65$ | $8.84$ |

It runs from the worst return, $-2$, through the average, $1.4$, to the best,
$9$. The expectation semiring is the middle of this family, and max-sum is its
right end.

:::{dropdown} Why does the tilted mean become the ordinary mean as the tilt goes to zero?
For a small $\kappa$, $e^{\kappa v} \approx 1 + \kappa v$, so

$$\sum_x p(x \mid S)\, e^{\kappa v} \approx 1 + \kappa \sum_x p(x \mid S)\, v
= 1 + \kappa\, \mathbb{E}[v \mid S].$$

Then $\log(1 + \kappa\, \mathbb{E}[v \mid S]) \approx \kappa\, \mathbb{E}[v \mid S]$,
and dividing by $\kappa$ leaves $\mathbb{E}[v \mid S]$.

Keeping one more term of the expansion shows what the tilt adds:

$$\bar v_\kappa = \mathbb{E}[v \mid S] + \frac{\kappa}{2}\, \operatorname{Var}[v \mid S] + \dots$$

A positive tilt rewards variance, a negative tilt penalizes it. This is why
the tilted mean is used to model the attitude toward risk: $\kappa > 0$ is
risk-seeking and $\kappa < 0$ is risk-averse.
:::

:::{dropdown} Why does it become the maximum as the tilt goes to infinity?
Let $v_{\max}$ be the largest value among the outcomes with nonzero
probability, and $p_{\max}$ the probability of the outcome that attains it.
The sum is dominated by its largest term:

$$p_{\max}\, e^{\kappa v_{\max}} \;\le\; \sum_x p(x \mid S)\, e^{\kappa v}
\;\le\; e^{\kappa v_{\max}}.$$

Taking the logarithm and dividing by $\kappa > 0$,

$$v_{\max} + \frac{\log p_{\max}}{\kappa} \;\le\; \bar v_\kappa \;\le\; v_{\max}.$$

The term $\log p_{\max} / \kappa$ vanishes as $\kappa$ grows. The same argument
with $\kappa \to -\infty$ gives the minimum.
:::

**Stored form.** Like the expectation semiring, the tilted semiring is awkward
in the form $(p, v)$ and simple in a stored form. Store the probability
together with the *tilted value* $m = p\, e^{\kappa v}$. Then

$$\begin{aligned}
\text{product:} \quad & m = p_1 p_2\, e^{\kappa (v_1 + v_2)}
  = \big(p_1 e^{\kappa v_1}\big)\big(p_2 e^{\kappa v_2}\big) = m_1\, m_2, \\
\text{sum:} \quad & m = (p_1 + p_2)\, e^{\kappa \bar v_\kappa}
  = p_1 e^{\kappa v_1} + p_2 e^{\kappa v_2} = m_1 + m_2.
\end{aligned}$$

Both stored numbers follow the plain sum-product rules, each on its own:

$$(p_1, m_1) \otimes (p_2, m_2) = (p_1 p_2,\; m_1 m_2), \qquad
(p_1, m_1) \oplus (p_2, m_2) = (p_1 + p_2,\; m_1 + m_2).$$

The value is read back at the end as $v = \frac{1}{\kappa} \log (m / p)$.

:::{dropdown} How is the stored form of Chapter 1 related to this one?
The stored form of Chapter 1 is the pair $(p, w)$ with the weighted value
$w = p\, v$. It is the small-tilt limit of $(p, m)$. For a small $\kappa$,

$$m = p\, e^{\kappa v} \approx p + \kappa\, p\, v = p + \kappa\, w.$$

Compare with the dual number $p + w\,\varepsilon$ of Chapter 1, Section 5: the
unit $\varepsilon$, with $\varepsilon^2 = 0$, plays the role of an
infinitesimal tilt. Multiplying two such entries and dropping the terms in
$\kappa^2$ gives back the product rule of Chapter 1,

$$(p_1 + \kappa w_1)(p_2 + \kappa w_2) \approx p_1 p_2 + \kappa\,(p_1 w_2 + p_2 w_1).$$

So the expectation semiring is the first derivative of the tilted semiring
with respect to the tilt, taken at $\kappa = 0$.
:::

### Log-sum-exp, the soft maximum

Write the tilt as $\kappa = 1/\eta$, with a *temperature* $\eta > 0$. The
tilted mean becomes

$$\bar v_\eta(S) = \eta \log \sum_x p(x \mid S)\, e^{v(x, S) / \eta},$$

which is called log-sum-exp, or the **soft maximum**: as $\eta \to 0$ it tends
to $\max_x v$, and as $\eta \to \infty$ to the average.

:::{dropdown} Is the soft maximum the "softmax" of machine learning?
They are two halves of the same elimination step.

Machine learning calls *softmax* the function that turns a list of scores
$v_1, \dots, v_n$ into probabilities,

$$\operatorname{softmax}(v)_i = \frac{e^{v_i / \eta}}{\sum_j e^{v_j / \eta}}.$$

It answers "which entry is the largest?", softly: the largest score gets the
largest probability, and as $\eta \to 0$ it gets all of it. The soft maximum
answers "how large is the largest entry?", softly:

$$\bar v_\eta = \eta \log \sum_j e^{v_j / \eta} \;\;\xrightarrow{\;\eta \to 0\;}\;\; \max_j v_j.$$

So the softmax is a soft version of the *arg max*, and the soft maximum, also
called log-sum-exp, is a soft version of the *max*. The first is the
derivative of the second:
$\partial \bar v_\eta / \partial v_i = \operatorname{softmax}(v)_i$.

In elimination both appear at once. Summing out $x$ with the soft maximum
gives the new factor, whose value is $\bar v_\eta(S)$. The conditional that
is left on $x$, reweighted by its value channel as in Section 5, is

$$q(x \mid S) = \frac{p(x \mid S)\, e^{v(x, S) / \eta}}{\sum_{x'} p(x' \mid S)\, e^{v(x', S) / \eta}},$$

which is the softmax of the values, with $p(x \mid S)$ as a prior weight. With
a uniform $p$ over $n$ outcomes it is exactly the formula above, and the soft
maximum is the log-sum-exp minus the constant $\eta \log n$.

| Maximum, exact | Soft version | In elimination |
|---|---|---|
| $\max_x v$: the best value | soft maximum, log-sum-exp | the value of the new factor $\phi(S)$ |
| $\arg\max_x v$: the best choice | softmax | the reweighted conditional $q(x \mid S)$ |
:::

It is the same operator as the tilted mean. It has its own name because it is
used for a different purpose, and the purpose depends on *which* variables it
is applied to:

| The tilt is applied to | What is tilted | Meaning | Where |
|---|---|---|---|
| state variables | the outcome of the dynamics, which the agent does not control | an attitude toward **luck**: risk-seeking or risk-averse | [Chapter 8](chapter08.md) |
| action variables | the choice of action, which the agent controls | a **soft choice**: prefer good actions without committing to the best | Chapters [8](chapter08.md), [10](chapter10.md) and [17](chapter17.md) |

### The family at a glance

All members that carry a value share one product: probabilities multiply and
values add. They differ only in how the values of two exclusive outcomes, with
probabilities $p_1$ and $p_2$, merge:

| Semiring | The merged value | In words |
|---|---|---|
| expectation | $\dfrac{p_1 v_1 + p_2 v_2}{p_1 + p_2}$ | the average |
| tilted, or soft maximum | $\dfrac{1}{\kappa} \log \dfrac{p_1 e^{\kappa v_1} + p_2 e^{\kappa v_2}}{p_1 + p_2}$ | between the average and the better (or the worse) |
| max-sum | $\max(v_1, v_2)$ | the better |

## 3. What elimination needs

Why is elimination valid with all of these, and what would make it fail? The
answer is three properties. They are stated for factors, following the
*valuation algebras* of Shenoy and Shafer [\[1\]](#ch02-shenoy1990) and Kohlas
[\[2\]](#ch02-kohlas2003).

Write $f \otimes g$ for the product of two factors and $f^{\downarrow S}$ for
the factor $f$ with all variables except those in $S$ summed out by $\oplus$.

**Axiom 1: the product is commutative and associative.**

$$f \otimes g = g \otimes f, \qquad (f \otimes g) \otimes h = f \otimes (g \otimes h).$$

The factors of a bucket can be multiplied in any order.

**Axiom 2: summing out in stages is the same as summing out at once.** For
$T \subseteq S$,

$$\big(f^{\downarrow S}\big)^{\downarrow T} = f^{\downarrow T}.$$

Variables can be eliminated one at a time, and the order among them does not
change the result.

**Axiom 3: a factor that does not contain the variable can be taken out of the
sum.** If $f$ has no variable outside $S$,

$$(f \otimes g)^{\downarrow S} = f \otimes g^{\downarrow S}.$$

This is the distributive law at the level of factors. It is what makes
elimination cheap: when a variable is summed out, only the factors in its
bucket take part, and all others wait outside the sum.

**The three axioms hold whenever the entries form a commutative semiring**
[\[3\]](#ch02-kohlas2008), that is, whenever $\otimes$ and $\oplus$ on entries are commutative and
associative and satisfy

$$a \otimes (b \oplus c) = (a \otimes b) \oplus (a \otimes c).$$

So it is enough to check this one law on entries:

| Semiring | The distributive law | Why it holds |
|---|---|---|
| sum-product | $a\,(b + c) = a b + a c$ | arithmetic |
| expectation | the same, for dual numbers $p + w\,\varepsilon$ | arithmetic with $\varepsilon^2 = 0$ (Chapter 1, Section 5) |
| tilted, soft maximum | the same, for each of $p$ and $m$ separately | arithmetic, twice |
| max-sum | $a + \max(b, c) = \max(a + b,\; a + c)$ | adding a constant does not change which is larger |

**What the axioms do not allow.** They assume *one* sum $\oplus$ for all
variables. Control needs two: the average over the next state, which the agent
does not choose, and the maximum over the action, which it does. Two different
sums do not commute, so Axiom 2 fails and the elimination order starts to
matter.

A bet on a fair coin shows it. The action is the side to bet on, the outcome
is the side that comes up, and a correct bet pays 10:

$$\max_a \sum_{s'} p(s')\, v(a, s') = \max(5, 5) = 5,
\qquad
\sum_{s'} p(s') \max_a v(a, s') = 0.5 \cdot 10 + 0.5 \cdot 10 = 10.$$

The first is a bet placed before the toss. The second is a bet placed after
seeing the coin. [Chapter 4](chapter04.md) shows which order is the right one
for an MDP.

## 4. Division and conditionals

With $\otimes$ and $\oplus$ alone, elimination returns one result, the entry
left at the root. GTSAM returns more: for each variable a conditional
$c(x \mid S)$, and together a Bayes net. That requires a division $\oslash$,
defined as the inverse of the product:

$$c = \psi \oslash \phi \quad\text{is the factor for which}\quad c \otimes \phi = \psi.$$

In each semiring the product multiplies probabilities and adds values, so the
division divides probabilities and *subtracts* values:

| Semiring | The conditional $c(x \mid S) = \psi \oslash \phi$ | Its value channel is called |
|---|---|---|
| sum-product | $\dfrac{p(x, S)}{p(S)} = p(x \mid S)$ | (none) |
| expectation | $\big(p(x \mid S),\;\; v(x, S) - \bar v(S)\big)$ | the **surprise**: advantage or TD residual (Chapter 1) |
| tilted, soft maximum | $\big(p(x \mid S),\;\; v(x, S) - \bar v_\kappa(S)\big)$ | the *soft* advantage, when $x$ is an action |
| max-sum | $v(x, S) - \max_x v(x, S)$ | the **regret**: how much worse $x$ is than the best choice |

In all four, the value channel of the conditional says how the value changes
when $x$ becomes known, relative to the summary $\phi(S)$ that elimination
passed on. Only the summary differs: the average, the tilted mean, or the
maximum.

*On the track,* take the bucket of the last action $a_1$ under the coin-flip
policy. The value of its product is $Q_1(s, a)$, and the three conditionals
are:

| cell | $Q_1(s, L)$, $Q_1(s, R)$ | advantage (expectation) | regret (max-sum) | soft advantage ($\kappa = 1$) |
|---|---|---|---|---|
| 0 | $0$, $-1$ | $+0.5$, $-0.5$ | $0$, $-1$ | $+0.38$, $-0.62$ |
| 1 | $0$, $7$ | $-3.5$, $+3.5$ | $-7$, $0$ | $-6.31$, $+0.69$ |
| 2 | $2$, $9$ | $-3.5$, $+3.5$ | $-7$, $0$ | $-6.31$, $+0.69$ |

Division needs the divisor's probability to be nonzero. Where $p(S) = 0$ the
conditional is not defined, and GTSAM's convention $0 / 0 = 0$ is used, as in
its ordinary discrete elimination.

## 5. The normalization invariant

Every conditional produced by elimination satisfies one identity, in every
semiring:

$$\bigoplus_x c(x \mid S) = \mathbf{1} \qquad \text{for every value of } S.$$

In words: adding a conditional over its own variable gives the *one* of the
semiring.

:::{dropdown} Why does it hold?
The new factor $\phi(S)$ does not depend on $x$, so dividing by it can be
taken out of the sum over $x$, by the same distributive law that Axiom 3 uses
for the product:

$$\bigoplus_x c(x \mid S) = \bigoplus_x \big(\psi(x, S) \oslash \phi(S)\big)
= \Big(\bigoplus_x \psi(x, S)\Big) \oslash \phi(S)
= \phi(S) \oslash \phi(S) = \mathbf{1}.$$
:::

What it says depends on what the one is:

| Semiring | $\mathbf{1}$ | The invariant, written out | Meaning |
|---|---|---|---|
| sum-product | $1$ | $\sum_x p(x \mid S) = 1$ | a conditional is normalized |
| expectation | $(1, 0)$ | $\sum_x p(x \mid S) = 1$ and $\sum_x p(x \mid S)\, c_v(x \mid S) = 0$ | it is normalized, and its surprises average to zero |
| tilted, soft maximum | $(1, 0)$ | $\sum_x p(x \mid S) = 1$ and $\sum_x p(x \mid S)\, e^{\kappa\, c_v(x \mid S)} = 1$ | it is normalized, and so is its tilted version |
| max-sum | $0$ | $\max_x c(x \mid S) = 0$ | the best choice has zero regret |

Here $c_v$ is the value channel of the conditional. The second row is the
zero-mean property of the advantage and of the TD residual from Chapter 1,
Section 2. It did not need a separate proof there: it is the normalization of
a conditional, in a semiring whose one is $(1, 0)$.

The third row deserves a closer look. It says that

$$q(x \mid S) = p(x \mid S)\, e^{\kappa\, c_v(x \mid S)}$$

is itself a normalized distribution over $x$: the original conditional,
reweighted toward the outcomes with a high value. It is the *softmax* of the
values, with the original conditional as a prior weight (Section 2). For the last action on the
track, with $\kappa = 1$:

| cell | coin flip $\pi(L \mid s)$, $\pi(R \mid s)$ | tilted $q(L \mid s)$, $q(R \mid s)$ |
|---|---|---|
| 0 | $0.5$, $0.5$ | $0.731$, $0.269$ |
| 1 | $0.5$, $0.5$ | $0.001$, $0.999$ |
| 2 | $0.5$, $0.5$ | $0.001$, $0.999$ |

When $x$ is an action, $q$ is an *improved policy*: it prefers the actions
with a positive advantage, smoothly. This reweighting,
$\pi_{\text{new}} \propto \pi\, e^{A / \eta}$, is the policy update of a whole
class of RL algorithms ([Chapter 17](chapter17.md)). When $x$ is a state, $q$
is a version of the dynamics bent toward lucky outcomes, which is the source
of the optimism discussed in [Chapter 8](chapter08.md).

**Why the invariant matters.**

- **It is a test.** Any implementation of a semiring factor can be checked by
  adding each conditional over its variable and comparing with $\mathbf{1}$.
  The unit tests of the module do this ([Appendix A](appendix_a.md)).
- **It makes the Bayes net relative.** The product of all conditionals is the
  product of all factors divided by the result at the root. For the
  expectation semiring that is $(p(\tau),\; R(\tau) - J)$: the surprises along
  a trajectory add up to its return relative to the average (Chapter 1,
  Section 7). For max-sum it is $R(\tau) - \max_\tau R$: the regrets along a
  trajectory add up to how far it falls short of the best one.
- **It is the baseline of RL.** Subtracting $\bar v(S)$ from the value is what
  RL calls subtracting a baseline. Chapters [5](chapter05.md) and
  [11](chapter11.md) use the fact that it comes for free with every
  conditional.

## 6. The log-dual form

Chapter 1 presented the pair in two forms: $(p, v)$, for thinking, and
$(p, w)$ with $w = p\, v$, for storing. There is a third, obtained by taking
the logarithm of the probability:

$$(\ell, v), \qquad \ell = \log p.$$

In this form the product **adds both channels**, and the division subtracts
both:

$$(\ell_1, v_1) \otimes (\ell_2, v_2) = (\ell_1 + \ell_2,\;\; v_1 + v_2),
\qquad
(\ell, v) \oslash (\ell_S, \bar v) = (\ell - \ell_S,\;\; v - \bar v).$$

The sum is where the work is. The probability channel becomes a log-sum-exp,
and the value channel an average with the normalized probabilities as weights:

$$\ell(S) = \log \sum_x e^{\ell(x, S)},
\qquad
\bar v(S) = \sum_x e^{\ell(x, S) - \ell(S)}\; v(x, S).$$

**Why "dual".** The stored form of Chapter 1 is the dual number
$p + w\,\varepsilon$, with $\varepsilon^2 = 0$. Its logarithm is

$$\log(p + w\,\varepsilon) = \log\big(p\,(1 + v\,\varepsilon)\big)
= \log p + \log(1 + v\,\varepsilon) = \ell + v\,\varepsilon,$$

because $\log(1 + z) = z$ exactly when $z^2 = 0$. So $(\ell, v)$ is the
logarithm of the stored pair. It turns the semiring product into addition for
the same reason that log-probabilities turn a product of probabilities into a
sum.

| Form | Entry | Product | Used for |
|---|---|---|---|
| $(p, v)$ | probability, value | $(p_1 p_2,\; v_1 + v_2)$ | reasoning |
| $(p, w)$, $w = p\,v$ | probability, weighted value | $(p_1 p_2,\; p_1 w_2 + p_2 w_1)$ | discrete tables: the sum is a plain sum |
| $(\ell, v)$, $\ell = \log p$ | log-probability, value | $(\ell_1 + \ell_2,\; v_1 + v_2)$ | Gaussian factors, and long products |

The log-dual form has two uses.

**Numerical range.** A product of many small probabilities underflows to zero,
and $v = w / p$ is then $0 / 0$. The notebook multiplies 400 factors with
probabilities of order $10^{-3}$: the form $(p, w)$ returns `nan`, and the
form $(\ell, v)$ returns the correct value with $\ell \approx -2486$.

**Gaussian factors.** For a Gaussian, $\ell$ is a quadratic function: it is
minus the error that GTSAM stores for a Gaussian factor, up to a constant,

$$\ell(x) = -\tfrac{1}{2}\, \lVert A x - b \rVert^2_\Sigma + \text{const}.$$

If the value $v$ is a quadratic too, both channels are quadratics, and the
product of two factors just adds the quadratics of each channel. Summing out a
variable is an integral: the $\ell$ channel is ordinary Gaussian elimination,
and the $v$ channel is the expectation of a quadratic under the resulting
conditional, the "substitute the mean, add the variance" rule of Chapter 1,
Section 8. This is exactly how `SemiringGaussianFactor` is stored: a
`GaussianFactorGraph` for $\ell$ and a `HessianFactor` for $v$.
[Chapter 6](chapter06.md) builds on it.

## 7. In the module

- The module implements the **expectation** semiring, in two representations:
  $(p, w)$ tables for the discrete family and $(\ell, v)$ quadratics for the
  Gaussian family.
- The other members of the family are run in the
  [companion notebook](chapter02_examples.ipynb) by a generic elimination
  routine in numpy. The routine takes a semiring as an argument: four small
  functions that lift a probability, lift a reward, multiply, and add.
- A maximum over action variables only, with an average over the states, is
  not a single semiring. It is done by hand with the factor interface in
  [Chapter 4](chapter04.md).

## 8. Where each member is used

| Sum over the actions | Sum over the states | What it computes | Chapters |
|---|---|---|---|
| average under the policy | average | the value of a given policy | 1, 3, 5, 7, 11 to 14 |
| maximum | average | the best policy | 4, 6, 15, 16 |
| soft maximum | average | a soft-optimal policy | 8, 10, 17 |
| maximum | tilted mean | a risk-sensitive best policy | 8 |
| maximum | maximum | the best trajectory, optimistic about the dynamics | 9 |

The first column is the first choice every algorithm in this book makes. The
[taxonomy table](appendix_c.md) lists it for each of them.

(chapter02-references)=
## 9. References

(ch02-shenoy1990)=
\[1\] P. P. Shenoy and G. Shafer, "Axioms for probability and belief-function
propagation", *Uncertainty in Artificial Intelligence 4*, 1990. The three
axioms, and message passing derived from them.

(ch02-kohlas2003)=
\[2\] J. Kohlas, *Information Algebras: Generic Structures for Inference*,
Springer, 2003. Valuation algebras, including those with division.

(ch02-kohlas2008)=
\[3\] J. Kohlas and N. Wilson, "Semiring induced valuation algebras: exact and
approximate local computation algorithms", *Artificial Intelligence*, 2008. A
commutative semiring on entries gives a valuation algebra on factors.

(ch02-aji2000)=
\[4\] S. M. Aji and R. J. McEliece, "The generalized distributive law", *IEEE
Transactions on Information Theory*, 2000. One message-passing algorithm for
any commutative semiring, with sum-product and max-sum as instances.

(ch02-dechter1999)=
\[5\] R. Dechter, "Bucket elimination: a unifying framework for reasoning",
*Artificial Intelligence*, 1999. Variable elimination for sums, maxima and
mixtures of the two.

(ch02-kschischang2001)=
\[6\] F. R. Kschischang, B. J. Frey and H.-A. Loeliger, "Factor graphs and the
sum-product algorithm", *IEEE Transactions on Information Theory*, 2001.

(ch02-eisner2002)=
\[7\] J. Eisner, "Parameter estimation for probabilistic finite-state
transducers", *ACL*, 2002. The expectation semiring.

(ch02-li2009)=
\[8\] Z. Li and J. Eisner, "First- and second-order expectation semirings with
applications to minimum-risk training on translation forests", *EMNLP*, 2009.

(ch02-viterbi1967)=
\[9\] A. J. Viterbi, "Error bounds for convolutional codes and an
asymptotically optimum decoding algorithm", *IEEE Transactions on Information
Theory*, 1967. The best path through a chain by keeping the better of two
partial paths: max-sum elimination.

(ch02-howard1972)=
\[10\] R. A. Howard and J. E. Matheson, "Risk-sensitive Markov decision
processes", *Management Science*, 1972. The exponential tilt of the return in
a tabular decision process.

(ch02-jacobson1973)=
\[11\] D. H. Jacobson, "Optimal stochastic linear systems with exponential
performance criteria and their relation to deterministic differential games",
*IEEE Transactions on Automatic Control*, 1973. The tilt in the linear-
Gaussian case.

(ch02-whittle1990)=
\[12\] P. Whittle, *Risk-Sensitive Optimal Control*, Wiley, 1990. The
exponential tilt as a model of the attitude toward risk.

(ch02-kappen2005)=
\[13\] H. J. Kappen, "Path integrals and symmetry breaking for optimal control
theory", *Journal of Statistical Mechanics*, 2005. Control whose value is a
log-sum-exp over trajectories.

(ch02-todorov2006)=
\[14\] E. Todorov, "Linearly-solvable Markov decision problems", *NeurIPS*,
2006. With a soft maximum over the actions the backup becomes linear.

(ch02-levine2018)=
\[15\] S. Levine, "Reinforcement learning and control as probabilistic
inference: tutorial and review", arXiv:1805.00909, 2018. The soft maximum in
RL, with its temperature.

---

Previous: [Chapter 1: MDPs as factor graphs: evaluating a policy by variable elimination](chapter01.md).
Next: [Chapter 3: Infinite horizon and discounting](chapter03.md).
