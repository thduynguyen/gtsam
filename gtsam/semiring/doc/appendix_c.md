# Appendix C: The taxonomy table

:::{div}
:class: in-progress
**This book is a work in progress.** It is still being written and revised: its content is incomplete and may contain errors.
:::

This appendix is the index of the book. It lists every algorithm with the
choices it makes along the five axes of the two-stage framework
([Chapter 5](chapter05.md), Section 6). Each row is the *framework card* of
the chapter that treats the algorithm, shortened to fit.

The axes, once more:

| Axis | The question |
|---|---|
| 1. Sum over the actions | Which semiring sum eliminates the action in Stage 1: the average under the policy, the maximum, or the soft maximum? |
| 2. Dynamics factor | How is $p(s' \mid s, a)$ accessed: in closed form, linearized, by samples, or as a learned model? |
| 3. Backward messages | How are the values $Q$, $V$ and the advantages $A$ computed? |
| 4. Forward messages | How is the state visitation $d_t$ computed? |
| 5. Stage 2 update | How are the policy parameters $\theta$ changed? |

Reading down a column shows how one choice varies across the field. Reading
along a row shows what an algorithm is. Two algorithms that differ in one cell
differ in exactly that respect: Q-learning and SARSA differ in the first
column only, TRPO and PPO in the last, iLQR and LQR in the second.

## Part I: Exact elimination

| Algorithm | Ch. | 1. Sum over the actions | 2. Dynamics factor | 3. Backward messages | 4. Forward messages | 5. Stage 2 update |
|---|---|---|---|---|---|---|
| Policy evaluation | [1](chapter01.md), [3](chapter03.md) | average under $\pi$ | closed form (tables) | exact | exact, if asked for | none |
| Dynamic programming, value iteration | [4](chapter04.md) | maximum | closed form (tables) | exact | not needed | none: the policy is read from the conditionals |
| Policy iteration | [4](chapter04.md) | average under $\pi$ | closed form (tables) | exact | not needed | greedy per state |
| Exact policy gradient | [5](chapter05.md) | average under $\pi_\theta$ | closed form (tables) | exact | exact | gradient or natural gradient |

## Part II: Gaussian elimination and stochastic optimal control

| Algorithm | Ch. | 1. Sum over the actions | 2. Dynamics factor | 3. Backward messages | 4. Forward messages | 5. Stage 2 update |
|---|---|---|---|---|---|---|
| LQR | [6](chapter06.md) | maximum | closed form: linear-Gaussian | exact: a quadratic (Riccati) | not needed | none: the policy is read from the conditionals |
| LQG | [6](chapter06.md), [22](chapter22.md) | maximum | closed form, with an observation factor | exact: the same quadratic | exact: Gaussian given the observations (Kalman filter) | none: the gain is applied to the estimate |
| Exact policy gradient, linear-Gaussian policy | [7](chapter07.md) | average under $\pi_\theta$ | closed form | exact: a quadratic (Lyapunov) | exact: Gaussian state marginals | gradient |
| Natural gradient, linear-Gaussian policy | [7](chapter07.md) | average under $\pi_\theta$ | closed form | exact | exact | natural gradient (Gauss-Newton with the Fisher matrix), damped (Levenberg-Marquardt) or with a trust region on the KL divergence |
| Soft (maximum-entropy) control | [8](chapter08.md) | soft maximum; average at the states | closed form | exact | not needed | none: the tilted conditional of the action |
| Control as inference, linearly solvable MDPs | [8](chapter08.md) | soft maximum; the same tilt at the states | closed form | exact; linear in $e^{V / \eta}$ | not needed | none: the posterior conditional of the action |
| LEQG (risk-sensitive) | [8](chapter08.md) | maximum; tilted mean at the states | closed form: linear-Gaussian | exact: Riccati with a tilted matrix | not needed | none |
| iLQR, DDP | [9](chapter09.md) | maximum; average at the states | closed form, linearized around the current trajectory (DDP: to second order) | a local quadratic, exact for the linearized graph | one trajectory: the rollout of the current policy | relinearize and repeat, with a line search |
| MAP trajectory optimization | [9](chapter09.md) | maximum; maximum at the states too | closed form, linearized at every iteration | none kept: one joint least-squares solve | none kept | Gauss-Newton or Levenberg-Marquardt on all states and controls |
| AICO | [9](chapter09.md) | posterior of the graph with factors $e^{r}$ | closed form, linearized at the mean of each belief | a Gaussian per state | a Gaussian per state | repeat the sweeps to a fixed point |
| MPPI, path-integral control | [10](chapter10.md) | soft maximum over whole plans, from $M$ samples | a simulator: the model is run, never differentiated | Monte Carlo: the return of each sampled rollout | particles: the sampled rollouts | refit the mean of the sampling distribution to the weighted samples |
| CEM | [10](chapter10.md) | maximum over whole plans, from the best of $M$ samples | a simulator | Monte Carlo returns | particles | refit mean and variance to the best samples |

Model predictive control (MPC) is a wrapper around any row of this part: run
both stages on a short horizon from the current state, apply the first action,
and repeat.

## Part III: Sampled and learned messages (model-free RL)

| Algorithm | Ch. | 1. Sum over the actions | 2. Dynamics factor | 3. Backward messages | 4. Forward messages | 5. Stage 2 update |
|---|---|---|---|---|---|---|
| REINFORCE, with a baseline | [11](chapter11.md) | average under $\pi_\theta$, by sampling | samples: whole rollouts | Monte Carlo return $R_t$ minus a baseline | particles: the states visited by rollouts | gradient, sampled |
| TD(0) | [12](chapter12.md) | average under $\pi$, by sampling | samples: single transitions | bootstrapped: $\hat V(s)$ toward $r + \gamma \hat V(s')$ | particles: visited states | none (evaluation only) |
| TD($\lambda$), GAE | [12](chapter12.md) | average under $\pi$, by sampling | samples: short sequences | bootstrapped $n$-step targets mixed by $\lambda$ | particles: visited states | none (evaluation only) |
| Actor-critic (A2C) | [13](chapter13.md) | average under $\pi_\theta$, by sampling | samples: single transitions | learned critic $\hat V$ by TD; advantage from the TD residual or GAE | particles from the current policy | gradient, sampled |
| Natural policy gradient | [14](chapter14.md) | average under $\pi_\theta$, by sampling | samples | learned critic, GAE | particles from the current policy | natural gradient |
| TRPO | [14](chapter14.md) | the same, with importance weights when reusing | samples | learned critic, GAE, frozen at $\pi_{\text{old}}$ | particles from $\pi_{\text{old}}$, frozen | trust region: maximize the surrogate subject to $\mathrm{KL} \le D_{\max}$ |
| PPO | [14](chapter14.md) | the same, with importance weights | samples | the same, reused for several passes | particles from $\pi_{\text{old}}$, frozen and reused | several gradient steps on the surrogate, clipped at $1 \pm \epsilon_{\text{clip}}$ |
| SARSA | [15](chapter15.md) | average under the current policy, sampled | sampled transitions, from the current policy | bootstrapped table, one-step target | particles of the current policy | greedy per state, with some random actions |
| Q-learning | [15](chapter15.md) | maximum | sampled transitions, from any policy | bootstrapped table, one-step target | particles of any data policy | greedy per state |
| Fitted Q iteration | [15](chapter15.md) | maximum | a fixed batch of transitions | bootstrapped, refitted by regression at every sweep | the batch | greedy per state |
| DQN | [15](chapter15.md) | maximum | sampled transitions, kept in a replay buffer | learned critic, bootstrapped from a frozen copy | replay buffer | greedy per state, with some random actions |
| DPG, with exact messages | [16](chapter16.md) | average under a deterministic policy: substitute its action | closed form (linear-Gaussian) | exact quadratic $Q$ | exact | gradient through the action |
| DDPG | [16](chapter16.md) | the same | sampled transitions, replay buffer | learned critic, bootstrapped from frozen copies | replay buffer | gradient through the action, with the learned critic |
| TD3 | [16](chapter16.md) | the same, at a slightly noisy action | sampled transitions, replay buffer | two learned critics; the target uses the smaller | replay buffer | the same, once per two critic updates |
| SAC | [17](chapter17.md) | soft maximum, with a fixed reference policy | sampled transitions, replay buffer | learned soft critic, bootstrapped; two critics | replay buffer | gradient steps fitting $\pi_\theta$ to the tilted conditional; the temperature follows a target entropy |
| REPS | [17](chapter17.md) | soft maximum, relative to the current policy | sampled transitions from the current policy | learned $V$; weights from TD residuals | particles of the current policy | EM: weighted maximum likelihood; the temperature from a KL bound |
| MPO | [17](chapter17.md) | soft maximum, relative to the current policy | sampled transitions, replay buffer | learned critic, bootstrapped | replay buffer | EM: weighted maximum likelihood with a KL bound |
| AWR | [17](chapter17.md) | soft maximum, relative to the current policy | sampled transitions, replay buffer | returns from the data minus a learned $V$ | replay buffer | EM: weighted maximum likelihood, fixed temperature |

## Part IV: Learned factors (model-based RL)

| Algorithm | Ch. | 1. Sum over the actions | 2. Dynamics factor | 3. Backward messages | 4. Forward messages | 5. Stage 2 update |
|---|---|---|---|---|---|---|
| Certainty-equivalent control with a learned model | [18](chapter18.md) | maximum | learned: a linear-Gaussian factor fitted by least squares | exact on the learned factor (Riccati) | not needed | none; an outer loop collects data and refits |
| PILCO | [19](chapter19.md) | average under $\pi_\theta$ (deterministic: substitution) | learned: a Gaussian process | not computed: derivatives go through the forward recursion | a moment-matched Gaussian | gradient of the closed-form return; an outer loop collects data |
| Guided policy search | [20](chapter20.md) | maximum per local problem (soft maximum in the original) | learned: a local linear model per trajectory | a local quadratic, exact on the local model (iLQR) | the local trajectories, one per start | supervised fit of $\pi_\theta$, coordinated by dual variables |
| PETS | [21](chapter21.md) | maximum, sampled (CEM) | learned: an ensemble of networks | none: the returns of sampled plans | particles through the ensemble, from the real state | none; receding horizon |
| MBPO | [21](chapter21.md) | soft maximum (as SAC) | learned: an ensemble of networks | learned critic, on real and imagined transitions | replay buffer, extended by short model rollouts | gradient (the SAC update) |
| Dreamer | [21](chapter21.md) | average under $\pi_\theta$ | learned, in a learned latent space | learned critic on imagined rollouts | particles in the latent space | gradient of the predicted value through the model |
| TD-MPC2 | [21](chapter21.md) | soft maximum, sampled (MPPI) | learned, in a learned latent space | learned critic closing a short horizon | particles in the latent space, from the real state | gradient on the critic for a guiding policy; receding horizon |

## Part V: Beyond the clean story

| Algorithm | Ch. | 1. Sum over the actions | 2. Dynamics factor | 3. Backward messages | 4. Forward messages | 5. Stage 2 update |
|---|---|---|---|---|---|---|
| Exact belief-space solution of a POMDP | [22](chapter22.md) | maximum, one per belief | tables, with a sensor factor | exact, a function of the belief; exponential in the horizon | the belief $d_t(s \mid y_{0:t})$, by an exact filter | none |
| QMDP | [22](chapter22.md) | maximum, one per belief | tables, with a sensor factor | exact for the fully observed problem | the belief, by an exact filter | none |
| Maximum-entropy inverse RL | [23](chapter23.md) | soft maximum; average at the states | tables, kept fixed | exact soft $Q$ and $V$ | exact | gradient on the reward parameters $\theta_r$: counted minus expected features |
| Bayes-adaptive exact solution | [24](chapter24.md) | maximum, one per belief about the model | unknown: its parameters are summed out under the belief | exact, a function of the belief | the belief about the model | none |
| Thompson sampling | [24](chapter24.md) | maximum, for one sampled model | one model drawn from the belief | exact or approximate, for the sampled model | the belief, used to draw the model | none, or that of the underlying method |
| Bonus-based exploration (UCB, novelty) | [24](chapter24.md) | maximum, with a bonus added to the reward | the current estimate | as in the underlying method | counts or prediction errors | that of the underlying method |
| Exact return distribution | [25](chapter25.md) | average under $\pi$, in the convolution semiring | tables | exact: a return distribution per state | not needed | none |
| C51 | [25](chapter25.md) | maximum of the mean | sampled transitions | learned, bootstrapped: probabilities on a fixed grid | replay buffer | greedy with respect to the mean |
| QR-DQN | [25](chapter25.md) | maximum of the mean | sampled transitions | learned, bootstrapped: quantiles | replay buffer | greedy with respect to the mean |
| PPO for sim-to-real locomotion | [26](chapter26.md) | average under $\pi_\theta$ | simulator samples, with randomized model parameters | learned $V$ with GAE | particles: fresh parallel rollouts | clipped trust-region steps |
| SAC or TD3 on hardware | [26](chapter26.md) | soft maximum, or maximum by gradient | real transitions | learned $Q$, bootstrapped, two copies | replay buffer | gradient through the critic |
| MPC with learned components | [26](chapter26.md) | maximum or soft maximum, sampled or by a local quadratic | a learned model, or a known model with learned parts | short model rollouts, and a learned critic at the horizon | a point: the current state | re-plan at every control step |
| Residual RL | [26](chapter26.md) | that of the training method | real or simulated transitions | as in the training method | as in the training method | that of the training method, on a correction to a nominal controller |

## Reading the table

- **Down column 1,** the three sums of [Chapter 2](chapter02.md) divide the
  field: the average evaluates, the maximum optimizes, and the soft maximum
  does something in between that keeps a distribution over actions.
- **Down column 2,** access to the dynamics factor weakens from Part I to
  Part III, closed form, then linearized, then samples only, and Part IV puts
  a learned factor in its place.
- **Columns 3 and 4** are where approximation enters. Exact messages need the
  dynamics in closed form. Without it the backward message is estimated from
  returns or by bootstrapping, and the forward message is a set of visited
  states, fresh or stored.
- **Column 5** runs from "none", when the maximum can be taken inside the
  elimination, through greedy and gradient steps to trust regions and EM.
  These are the outer loops a SLAM reader knows as Gauss-Newton,
  Levenberg-Marquardt and Dogleg.
- **Part V** lists the cases where a cell needs more than the framework
  offers: a backward message that is a function of a belief, a model that is
  itself uncertain, or a value that is a whole distribution.

---

Previous: [Appendix B: Notation](appendix_b.md).
