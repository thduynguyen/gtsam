# Semiring factor graphs

These chapters express optimal control and reinforcement learning (RL) as
operations on factor graphs, for readers who know factor graphs from SLAM.

They accompany the `gtsam/semiring` module of GTSAM, in which every factor
entry carries two numbers, a probability and a value. Ordinary variable
elimination on such factors computes the quantities that control and RL are
built on: the expected return of a policy, its value functions, and its
advantages.

## Chapters

1. [MDPs as factor graphs: evaluating a policy by variable elimination](chapter01.md).
   A Markov decision process as a factor graph; why ordinary elimination cannot
   evaluate it; semiring factors and their operators; the correspondence
   between RL quantities and elimination; a discrete and a continuous (LQR)
   worked example. [Run the examples](chapter01_examples.ipynb).
2. [Finding the best policy in one pass](chapter02.md). A teaser for policy
   optimization: for tabular problems and for LQR, replacing the expectation
   over actions by a maximum yields the best policy in one backward pass. For
   LQR this is, line by line, the Riccati recursion of classic control.
   [Run the examples](chapter02_examples.ipynb).

Planned chapters:

- **Optimal control with known dynamics.** A generic two-stage framework on
  semiring factor graphs, an inner elimination that evaluates a policy inside
  an outer optimization that improves it, with the algorithms of control mapped
  to optimization on the graph.
- **Reinforcement learning.** The case where the dynamics are unknown or only
  available through a simulator, with RL algorithms mapped to the same two
  stages, the exact sums and integrals replaced by sample approximations.

## Running the examples

Each chapter has a companion notebook with the code of its examples. The
notebooks open in Google Colab from the badge at the top of each chapter; their
first cell installs a GTSAM build that contains the semiring module.

The source of the module, the chapters and the notebooks is in
[`gtsam/semiring`](https://github.com/thduynguyen/gtsam/tree/feature/semiringfactor/gtsam/semiring).
