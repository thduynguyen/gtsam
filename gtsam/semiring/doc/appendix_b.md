# Appendix B: Notation

Every symbol in this book has one meaning. This appendix lists them, grouped
by where they are introduced. Where the control or RL literature uses another
letter, the last column says so.

## The problem

| Symbol | Meaning | Introduced | Elsewhere written |
|---|---|---|---|
| $s_t$, $a_t$ | state and action at step $t$, discrete | Ch. 1 | |
| $x_t$, $u_t$ | state and action at step $t$, continuous | Ch. 1 | |
| $s'$, $x'$ | the next state, $s_{t+1}$ or $x_{t+1}$ | Ch. 1 | |
| $T$ | the last step of a problem with a fixed number of moves | Ch. 1 | $H$, $N$ |
| $\tau$ | a trajectory: one assignment to all states and actions | Ch. 1 | |
| $p(s_0)$, $p_0$ | the distribution of the first state, and the same as a vector | Ch. 1, 3 | $\rho_0$, $\mu$ |
| $p(s' \mid s, a)$ | the dynamics | Ch. 1 | $P$, $T$ |
| $\pi(a \mid s)$ | the policy | Ch. 1 | |
| $r(s, a)$, $r(s_T)$ | the reward of a step, and the final reward | Ch. 1 | $-c$, $-\ell$ for costs |
| $R(\tau)$ | the return: the sum of the rewards along a trajectory | Ch. 1 | $G$ |
| $R_t$ | the return from step $t$ on, along one trajectory | Ch. 11 | $G_t$ |
| $J$ | the expected return, $\mathbb{E}[R]$ | Ch. 1 | $\eta$, $V^\pi(\rho_0)$ |
| $\gamma$ | the discount: the probability that the episode continues | Ch. 3 | |
| $\varnothing$ | the state "ended" | Ch. 3 | |
| $y_t$ | an observation of the state | Ch. 6 | $o_t$, $z_t$ |

## Values

| Symbol | Meaning | Introduced | Elsewhere written |
|---|---|---|---|
| $Q_t(s, a)$, $Q(s, a)$ | action value, at step $t$ or stationary | Ch. 1, 3 | |
| $V_t(s)$, $V(s)$ | state value | Ch. 1, 3 | |
| $A_t(s, a) = Q_t - V_t$ | advantage | Ch. 1 | |
| $\delta_t$ | TD residual | Ch. 1, 3 | |
| $Q^*$, $V^*$, $\pi^*$, $J^*$ | the same, for the best policy | Ch. 4 | |
| $V^{(k)}$ | the value with $k$ moves left | Ch. 3 | $V_k$ |
| $P_\pi$, $r_\pi$ | state-to-state table and expected reward of a policy | Ch. 3 | |
| $d_t(s)$ | state visitation: the marginal of $s_t$, the forward message | Ch. 1, 3 | $\rho_t$, $\mu_t$ |
| $d(s)$ | discounted visitation, $\sum_t \gamma^t d_t(s)$ | Ch. 3 | $\rho^\pi$, $d^\pi$ |
| $d_t(s \mid y_{0:t})$ | the belief: the forward message given the observations so far | Ch. 22 | $b_t$ |

## Semirings and elimination

| Symbol | Meaning | Introduced |
|---|---|---|
| $(p, v)$ | an entry of a semiring factor: probability and value | Ch. 1 |
| $(p, w)$, $w = p\,v$ | the stored form: probability and weighted value | Ch. 1 |
| $(\ell, v)$, $\ell = \log p$ | the log-dual form | Ch. 2 |
| $(p, m)$, $m = p\, e^{\kappa v}$ | the stored form of the tilted semiring | Ch. 2 |
| $\otimes$, $\oplus$, $\oslash$ | the product, sum and division of a semiring | Ch. 1, 2 |
| $\mathbf{0}$, $\mathbf{1}$ | its zero and its one | Ch. 2 |
| $x$, $S$ | the variable being eliminated, and its separator | Ch. 1 |
| $\psi(x, S)$ | the product of the factors in the bucket of $x$ | Ch. 1 |
| $\phi(S)$ | the new factor that elimination leaves on the separator | Ch. 1 |
| $c(x \mid S)$ | the conditional that elimination leaves on $x$ | Ch. 1 |
| $\bar v(S)$ | the value of $\phi$: the mean of $v$ given $S$ | Ch. 1 |
| $\bar v_\kappa(S)$, $\bar v_\eta(S)$ | the tilted mean, and the soft maximum | Ch. 2 |
| $\kappa$ | the tilt; $\kappa > 0$ is risk-seeking | Ch. 2 |
| $\eta$ | the temperature of the soft maximum, $\eta = 1 / \kappa$ | Ch. 2 |
| $\varepsilon$ | the unit of the dual number $p + w\,\varepsilon$, with $\varepsilon^2 = 0$ | Ch. 1 |
| $\varepsilon_\theta$ | the second unit, marking a derivative with respect to $\theta$ | Ch. 5 |
| $f^{\downarrow S}$ | a factor with all variables outside $S$ summed out | Ch. 2 |

## Parameters and the two stages

| Symbol | Meaning | Introduced | Elsewhere written |
|---|---|---|---|
| $\theta$ | the parameters of the policy $\pi_\theta$ | Ch. 4 | |
| $\theta_i$ | one entry of the parameter vector | Ch. 5 | |
| $\theta_k$ | the parameters at iteration $k$ of the outer loop | Ch. 5 | |
| $\theta_V$, $\theta_Q$ | the parameters of a learned value function, the critic | Ch. 12 | $w$, $\phi$ |
| $\theta_Q^-$ | the parameters of a slowly updated copy of the critic | Ch. 15 | $\theta'$ |
| $\theta_p$ | the parameters of a learned dynamics model | Ch. 18 | |
| $\theta_r$ | the parameters of a learned reward | Ch. 23 | |
| $\nabla_\theta J$ | the gradient of the expected return | Ch. 5 | $g$ |
| $g(\tau)$, $g_t$ | the score $\nabla_\theta \log p_\theta(\tau)$, and its term at step $t$, $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | Ch. 5 | |
| $\mathcal{I}(\theta)$ | the Fisher matrix | Ch. 5 | $F$ |
| $\alpha$ | a step size | Ch. 1 | |
| $\lambda_{\text{LM}}$ | the damping of a Levenberg-Marquardt step | Ch. 5 | $\lambda$ |
| $\Delta\theta$ | a change of the parameters | Ch. 5 | |
| $D_{\max}$ | the size of a trust region, as a bound on a KL divergence | Ch. 5, 14 | $\delta$ |
| $\epsilon_{\text{clip}}$ | the clipping range of PPO | Ch. 14 | $\epsilon$ |
| $\mathrm{KL}(q \,\|\, p)$ | the Kullback-Leibler divergence | Ch. 5 | |
| $\mathcal{H}$ | entropy | Ch. 8 | $H$ |

## Linear-Gaussian problems

| Symbol | Meaning | Introduced | Elsewhere written |
|---|---|---|---|
| $N(\mu, \Sigma)$ | a Gaussian with mean $\mu$ and covariance $\Sigma$ | Ch. 1 | |
| $F$, $B$ | the matrices of linear dynamics, $x' = F x + B u + \text{noise}$ | Ch. 6 | $A$, $B$ |
| $\Sigma_w$ | the covariance of the noise of the dynamics | Ch. 6 | $W$ |
| $C_x$, $C_u$, $C_T$ | the cost matrices: $r(x, u) = -(x^\top C_x x + u^\top C_u u)$, $r(x_T) = -x_T^\top C_T x_T$ | Ch. 6 | $Q$, $R$, $Q_f$ |
| $K_t$, $K$ | the feedback gain of a linear policy, $u = -K x$ | Ch. 6 | |
| $\Sigma_e$ | the covariance of the noise of a Gaussian policy | Ch. 6 | $\Sigma$ |
| $P_t$, $\beta_t$ | the quadratic value, $V_t(x) = -(x^\top P_t x + \beta_t)$ | Ch. 6 | $S_t$ |
| $H_{uu}$, $H_{ux}$, $H_{xx}$ | the blocks of the quadratic action value $Q_t(x, u)$ | Ch. 6 | $Q_{uu}$, ... |
| $\mu_0$, $\Sigma_0$ | the mean and covariance of the first state | Ch. 6 | |
| $G$, $\Sigma_y$ | the observation matrix and noise covariance, $y = G x + \text{noise}$ | Ch. 6 | $C$ or $H$, $V$ |
| $f(x, u)$ | nonlinear dynamics, $x' = f(x, u) + \text{noise}$ | Ch. 9 | |

The control literature writes the dynamics and cost matrices $A$, $B$, $Q$,
$R$. In this book $A$, $Q$ and $R$ stand for the advantage, the action value
and the return.

## Samples and learning

| Symbol | Meaning | Introduced | Elsewhere written |
|---|---|---|---|
| $M$ | the number of samples | Ch. 10 | $N$, $K$ |
| a superscript $(i)$ | the index of a sample, as in $\tau^{(i)}$ | Ch. 10 | |
| $\hat{\ }$ | an estimate, as in $\hat V$, $\hat Q$, $\widehat{\nabla_\theta J}$ | Ch. 1 | |
| $\lambda$ | the mixing parameter of TD($\lambda$) and GAE | Ch. 12 | |
| $n$ | the number of steps of an $n$-step return | Ch. 12 | |
| $\pi_{\text{old}}$ | the policy that the current messages were computed for | Ch. 14 | $\pi_k$ |
| $\pi_D$ | the policy that collected the data | Ch. 15 | $\mu$, $\beta$ |
| $\mathcal{D}$ | a set of stored transitions: the replay buffer | Ch. 15 | |
| $\rho$ | an importance weight, a ratio of two probabilities | Ch. 14 | |

Other symbols are local to one chapter and are defined where they are used.

---

Previous: [Appendix A: The GTSAM implementation](appendix_a.md).
