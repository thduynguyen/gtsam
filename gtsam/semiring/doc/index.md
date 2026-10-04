# Semiring factor graphs

This book expresses optimal control and reinforcement learning (RL) as
operations on factor graphs, for readers who know factor graphs from SLAM
and have not met control or RL before.

It accompanies the `gtsam/semiring` module of GTSAM, in which every factor
entry carries two numbers, a probability and a value. Ordinary variable
elimination on such factors computes the quantities that control and RL are
built on: the expected return of a policy, its value functions, and its
advantages.

The book has one organizing idea, the **two-stage framework** of
[Chapter 5](chapter05.md). Stage 1 eliminates the states and actions of the
graph with the policy held fixed, which produces *backward messages* (values
and advantages) and *forward messages* (how often each state is visited).
Stage 2 uses those messages to improve the policy. Every algorithm in the
book is a choice of how to carry out each stage, and each chapter after the
fifth follows the same template:

1. the graph: its variables, its factors and the policy parameters;
2. Stage 1: which semiring, how the dynamics factor is accessed, how the
   messages are computed;
3. Stage 2: how the policy is updated;
4. an exact special case that serves as a correctness test;
5. an implementation, with the module or in numpy;
6. what breaks.



## Part I: Exact elimination on decision graphs

The factor graph of a decision problem, the semirings that elimination can run on, and the two-stage framework that organizes the rest.

1. [MDPs as factor graphs: evaluating a policy by variable elimination](chapter01.md). A Markov decision process as a factor graph; why ordinary elimination cannot evaluate it; factor entries that carry a probability and a value; the correspondence between RL quantities and elimination. [Run the examples](chapter01_examples.ipynb).
2. [The semiring family](chapter02.md). Sum-product, expectation, max-sum, tilted and soft-maximum semirings; the axioms elimination needs; division, conditionals and their normalization invariant; the log-dual form. [Run the examples](chapter02_examples.ipynb).
3. [Infinite horizon and discounting](chapter03.md). The discount as a termination factor; a chain of identical steps; the Bellman equation as a fixed point of the backward message; the forward message. [Run the examples](chapter03_examples.ipynb).
4. [Decision nodes and elimination order](chapter04.md). Actions as decisions eliminated by a maximum; why the elimination order then matters; dynamic programming, value iteration and policy iteration; a shared parameter as one global decision. [Run the examples](chapter04_examples.ipynb).
5. [Gradients by elimination: the two-stage framework](chapter05.md). The policy gradient as forward message times local derivative times backward message; the second-order semiring; the Fisher matrix; the formal definition of Stage 1 and Stage 2. [Run the examples](chapter05_examples.ipynb).

## Appendices

- [Appendix A: The GTSAM implementation](appendix_a.md). The classes of the `gtsam/semiring` module, how they plug into GTSAM's elimination, and the tests of the semiring laws.
- [Appendix B: Notation](appendix_b.md). Every symbol of the book, with the names used elsewhere.

## Running the examples

Each chapter has a companion notebook with the code of its examples, executed,
with every number quoted in the chapter checked by an assertion. The notebooks
open in Google Colab from the badge at the top of each chapter. Those that use
the module install a GTSAM build that contains it in their first cell; the
others need only numpy.

The source of the module, the chapters and the notebooks is in
[`gtsam/semiring`](https://github.com/thduynguyen/gtsam/tree/feature/semiringfactor/gtsam/semiring).
