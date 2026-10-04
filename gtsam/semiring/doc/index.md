# Semiring factor graphs

:::{div}
:class: in-progress
**This book is a work in progress.** It is still being written and revised: its content is incomplete and may contain errors.
:::

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

The [taxonomy table](appendix_c.md) lists every algorithm with its choices.

## Part I: Exact elimination on decision graphs

The factor graph of a decision problem, the semirings that elimination can run on, and the two-stage framework that organizes the rest.

1. [MDPs as factor graphs: evaluating a policy by variable elimination](chapter01.md). A Markov decision process as a factor graph; why ordinary elimination cannot evaluate it; factor entries that carry a probability and a value; the correspondence between RL quantities and elimination. [Run the examples](chapter01_examples.ipynb).
2. [The semiring family](chapter02.md). Sum-product, expectation, max-sum, tilted and soft-maximum semirings; the axioms elimination needs; division, conditionals and their normalization invariant; the log-dual form. [Run the examples](chapter02_examples.ipynb).
3. [Infinite horizon and discounting](chapter03.md). The discount as a termination factor; a chain of identical steps; the Bellman equation as a fixed point of the backward message; the forward message. [Run the examples](chapter03_examples.ipynb).
4. [Decision nodes and elimination order](chapter04.md). Actions as decisions eliminated by a maximum; why the elimination order then matters; dynamic programming, value iteration and policy iteration; a shared parameter as one global decision. [Run the examples](chapter04_examples.ipynb).
5. [Gradients by elimination: the two-stage framework](chapter05.md). The policy gradient as forward message times local derivative times backward message; the second-order semiring; the Fisher matrix; the formal definition of Stage 1 and Stage 2. [Run the examples](chapter05_examples.ipynb).

## Part II: Gaussian elimination and stochastic optimal control

Known dynamics. Linear-quadratic control as Gaussian elimination, then nonlinear dynamics by relinearization and by sampling.

6. [LQR and LQG](chapter06.md). The Riccati recursion as Gaussian elimination with a maximum over the action; the Kalman filter as the forward pass; LQG and the separation principle. [Run the examples](chapter06_examples.ipynb).
7. [Policy optimization with exact messages](chapter07.md). A linear-Gaussian policy evaluated exactly, its exact gradient, and Stage 2 as gradient, Gauss-Newton and trust-region steps. [Run the examples](chapter07_examples.ipynb).
8. [Softness and risk](chapter08.md). Risk-sensitive control, linearly solvable MDPs and control as inference; the optimism problem, and how a fixed dynamics factor avoids it. [Run the examples](chapter08_examples.ipynb).
9. [Nonlinear dynamics](chapter09.md). iLQR and DDP as relinearize-then-eliminate; AICO; the contrast with MAP trajectory optimization; model predictive control. [Run the examples](chapter09_examples.ipynb).
10. [Sampling-based control](chapter10.md). MPPI and path-integral control as a sampled soft maximum, and CEM as a sampled maximum. [Run the examples](chapter10_examples.ipynb).

## Part III: Sampled and learned messages: model-free RL

Unknown dynamics. The same two stages, with the forward and backward messages estimated from sampled transitions.

11. [Monte Carlo messages](chapter11.md). Rollouts as particles for the forward message and returns for the backward message; REINFORCE; baselines as control variates. [Run the examples](chapter11_examples.ipynb).
12. [Bootstrapped messages](chapter12.md). TD(0), TD(lambda) and GAE as local consistency of the backward message; function approximation and the deadly triad. [Run the examples](chapter12_examples.ipynb).
13. [Actor-critic](chapter13.md). A learned backward message, the critic, combined with a gradient step on the policy. [Run the examples](chapter13_examples.ipynb).
14. [Stale messages and trust regions](chapter14.md). The performance difference lemma; natural policy gradient, TRPO and PPO as ways to reuse messages safely; the analogy with relinearization. [Run the examples](chapter14_examples.ipynb).
15. [Value-based control](chapter15.md). SARSA, Q-learning, fitted Q iteration and DQN; data collected by another policy. [Run the examples](chapter15_examples.ipynb).
16. [Off-policy actor-critic](chapter16.md). Deterministic policy gradients: DPG, DDPG and TD3. [Run the examples](chapter16_examples.ipynb).
17. [Soft and EM methods](chapter17.md). SAC, REPS, MPO and AWR: the soft maximum over actions, and policy improvement as expectation-maximization. [Run the examples](chapter17_examples.ipynb).

## Part IV: Learned factors: model-based RL

The dynamics factor itself is learned from data, and elimination runs on the learned factor.

18. [Learning the dynamics factor](chapter18.md). System identification as factor-graph learning; uncertainty about the model through ensembles and Gaussian processes. [Run the examples](chapter18_examples.ipynb).
19. [PILCO](chapter19.md). Moment-matched Gaussian forward messages through a Gaussian-process dynamics model. [Run the examples](chapter19_examples.ipynb).
20. [Guided policy search](chapter20.md). Local trajectory optimizers and one global policy, coordinated by dual decomposition. [Run the examples](chapter20_examples.ipynb).
21. [Planning in learned models](chapter21.md). PETS, MBPO, Dreamer and TD-MPC2: planning and learning inside a learned model, and compounding model error. [Run the examples](chapter21_examples.ipynb).

## Part V: Beyond the clean story

Hidden state, unknown rewards, exploration and return distributions: where elimination needs more structure, or is not enough.

22. [Partial observability](chapter22.md). POMDPs; the belief as a forward message; the separation principle revisited. [Run the examples](chapter22_examples.ipynb).
23. [Inverse problems](chapter23.md). Maximum-entropy inverse RL as learning the reward factors. [Run the examples](chapter23_examples.ipynb).
24. [Exploration and dual control](chapter24.md). Why deciding where to sample is not an elimination; the Bayes-optimal policy on a small problem, and practical heuristics. [Run the examples](chapter24_examples.ipynb).
25. [Distributional RL](chapter25.md). The full distribution of the return, carried by a convolution semiring. [Run the examples](chapter25_examples.ipynb).
26. [Robotics case studies](chapter26.md). PPO for legged locomotion, SAC and TD3 on hardware, MPC with learned components, and residual RL, each mapped onto the framework. [Run the examples](chapter26_examples.ipynb).

## Appendices

- [Appendix A: The GTSAM implementation](appendix_a.md). The classes of the `gtsam/semiring` module, how they plug into GTSAM's elimination, and the tests of the semiring laws.
- [Appendix B: Notation](appendix_b.md). Every symbol of the book, with the names used elsewhere.
- [Appendix C: The taxonomy table](appendix_c.md). Every algorithm of the book, with its choices along the five axes of the framework.

## Running the examples

Each chapter has a companion notebook with the code of its examples, executed,
with every number quoted in the chapter checked by an assertion. The notebooks
open in Google Colab from the badge at the top of each chapter. Those that use
the module install a GTSAM build that contains it in their first cell; the
others need only numpy.

The source of the module, the chapters and the notebooks is in
[`gtsam/semiring`](https://github.com/thduynguyen/gtsam/tree/feature/semiringfactor/gtsam/semiring).

## Copyright and license

Copyright © 2026 Duy Ta. All rights reserved. The text and figures of this book may not be reproduced or
redistributed without permission. The example code in the notebooks and tools is
under the BSD license of GTSAM. See [Copyright and license](LICENSE.md).
