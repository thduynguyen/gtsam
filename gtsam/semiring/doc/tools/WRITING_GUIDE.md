# Guide for writing chapters of the semiring book

This guide records the conventions of the book in `gtsam/semiring/doc`, so
that new and edited chapters read like the existing ones. Before writing, read
`chapter05.md` Section 6 (the two-stage framework) and `appendix_b.md` (the
notation), and skim `chapter01.md` Sections 1 to 4 for the style.

Paths below are relative to `gtsam/semiring/doc`; `tools/README.md` explains
how to run the tools.

## 1. Reader and style

The reader knows factor graphs and GTSAM from SLAM, and knows NO control or RL.

- Give the answer first, then the derivation. Open each section by saying
  what comes out.
- Back every claim with a formula. If a sentence says "X is the average of
  Y", the formula follows.
- Define every symbol before using it, and never reuse a symbol for a second
  meaning. The notation is fixed by `appendix_b.md`; use it. Symbols already
  taken include A (advantage), Q (action value), R (return), S (separator),
  V, J, d (visitation), c (conditional), w (weighted value), g (score),
  alpha (step size), gamma (discount), kappa (tilt), eta (temperature),
  lambda (TD-lambda only), epsilon (dual unit), delta (TD residual).
  LQR matrices are F, B, C_x, C_u, C_T, never A, Q, R. A new symbol must be
  free, must be defined where it is introduced, and must be added to
  `appendix_b.md`.
- Short, plain sentences. No rhetorical questions in running text (they are
  fine as dropdown titles). Bold "**NOT**" where a wrong reading is likely.
- Material that would break the main flow (a proof, a why, a variant) goes in
  a collapsible MyST dropdown:

      :::{dropdown} Why does the baseline not change the gradient?
      ...
      :::

  Never use raw HTML: MyST does not typeset maths inside it.
- Use worked numbers only on an example that has been introduced. Every
  number in the text must be printed and asserted by the chapter's notebook.
- Each elimination or message step that matters gets a graph figure (SVG),
  not ASCII art.
- Maths is plain TeX in `$...$` and `$$...$$`; multi-line displays use
  `\begin{aligned}` inside `$$`. In table cells write `\mid`, never `|`.
  Write the semiring's one and zero as `\mathbf{1}` and `\mathbf{0}`: the
  site renders `\mathbb{1}` as a plain digit. Equations are not numbered.
- Cross-reference chapters as `[Chapter 7](chapter07.md)`.
- Wrap prose at about 78 columns.

## 2. Chapter skeleton

    # Chapter NN: Title

    Two or three short paragraphs: what the previous chapter left open, what
    this chapter does, and "The short version:" as a bullet list of results.

    **Run the examples.** The code of this chapter's examples is in the companion
    notebook [chapterNN_examples.ipynb](chapterNN_examples.ipynb), which runs
    online:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thduynguyen/gtsam/blob/feature/semiringfactor/gtsam/semiring/doc/chapterNN_examples.ipynb)

    ## 1. The graph
    ## 2. Stage 1: ...
    ## 3. Stage 2: ...
    ## 4. An exact special case
    ## 5. Implementation
    ## 6. What breaks
    ## 7. Framework card
    (chapterNN-references)=
    ## 8. References

The six parts of the template, which every chapter from 6 on follows:

1. **The graph**: the variables, the factors, and theta. A figure.
2. **Stage 1**: which semiring sum at the actions, how the dynamics factor is
   accessed, how the backward messages (Q, V, A) and the forward messages
   (d_t) are computed.
3. **Stage 2**: how theta is updated.
4. **An exact special case that serves as a correctness test**: a case where
   the chapter's method must reproduce a number computed exactly in an earlier
   chapter (Section 4 below lists the reference numbers). Run it.
5. **A GTSAM or numpy implementation**: the essential lines, quoted from the
   notebook. Use the `gtsam/semiring` module where it applies (expectation
   semiring, discrete or linear-Gaussian); otherwise numpy.
6. **What breaks**: where the approximation hurts, as a bullet list, with
   pointers to the chapters that address each point.

Sections may be added or titled more specifically, but the six parts stay
recognizable and in this order. When a chapter covers several algorithms, each
gets its own subsection and its own column in the framework card.

**Framework card.** A chapter ends with a table of its algorithm(s) along the
five axes of Chapter 5, Section 6, in exactly this row order:

    | | Algorithm name |
    |---|---|
    | 1. Sum over the actions | ... |
    | 2. Dynamics factor | ... |
    | 3. Backward messages | ... |
    | 4. Forward messages | ... |
    | 5. Stage 2 update | ... |

The same row goes into `appendix_c.md`, the taxonomy table.

Do not write a "Previous / Next" footer; `tools/assemble.py` adds it.

Aim for 350 to 600 lines per chapter. Depth beats breadth: one algorithm
derived step by step on a running example is worth more than five described
in a paragraph each. For chapters that survey large systems, derive the core
idea on a small example and map the rest onto the framework in tables.

## 3. Vocabulary

- "Stage 1" / "Stage 2", "forward message" (d_t, the marginal of the state),
  "backward message" (the value factor: Q, V; and the surprise A in the
  conditional), "dynamics factor", "policy factor", "reward factor",
  "value factor", "the sum over the actions" (average under the policy,
  maximum, soft maximum), "tilt", "surprise", "eliminate".
- A sampled quantity is "a sampled message" or "particles"; a learned value
  function is "a learned backward message" or "the critic"; a replay buffer is
  "a store of old forward messages".
- SLAM analogies are welcome when they are exact (Gauss-Newton, Levenberg-
  Marquardt, Dogleg, relinearization, iSAM2, Schur complement, marginals).

## 4. Running examples and reference numbers

Reuse these. Every notebook defines them itself, since it runs alone on Colab.

**The track** (Chapter 1, Section 7): cells 0, 1, 2; actions L = 0, R = 1; two
moves.

    prior = np.array([0.5, 0.5, 0.0])                  # p(s0)
    dynamics = np.zeros((3, 2, 3))                     # p(s' | s, a)
    dynamics[0, L], dynamics[0, R] = [1, 0, 0], [0.2, 0.8, 0]
    dynamics[1, L], dynamics[1, R] = [0.8, 0.2, 0], [0, 0.2, 0.8]
    dynamics[2, L], dynamics[2, R] = [0, 0.8, 0.2], [0, 0, 1]
    move_reward = np.array([[0.0, -1.0]] * 3)          # r(s, a)
    final_reward = np.array([0.0, 0.0, 10.0])          # r(s2)

  Reference numbers: coin flip J = 1.4; Q_1 = [[0,-1],[0,7],[2,9]];
  V_1 = (-0.5, 3.5, 5.5); A_0 = [[-1.1,1.1],[-1.9,1.9],[-0.3,0.3]];
  A_1 = [[0.5,-0.5],[-3.5,3.5],[-3.5,3.5]]; d_0 = (0.5,0.5,0),
  d_1 = (0.5,0.3,0.2); best policy J* = 6.1 (R everywhere except L in cell 0 at
  the last move); best stationary policy J = 6 (R everywhere); max over
  everything = 9; logistic policy pi_theta(R|s) = sigmoid(theta_s) at
  theta = 0: gradient (0.15, 1.0, 0.35), Fisher diag(0.25, 0.2, 0.05), natural
  gradient (0.6, 5, 7).

**The endless track** (Chapter 3): the same dynamics and start, gamma = 0.9,

    reward = np.array([[0.0, -1.0], [0.0, -1.0], [2.0, 1.0]])   # r(s, a)

  Reference numbers: coin flip V = (-0.2248, 1.1017, 4.1231), J = 0.4385,
  Q = [[-0.2023,-0.2472],[0.0365,2.1669],[3.5354,4.7108]], discounted
  visitation d = (3.8062, 3.4746, 2.7192); best policy R everywhere,
  V* = (5.4194, 7.5610, 10), Q* = [[4.8775,5.4194],[5.2629,7.5610],[9.2439,10]],
  J* = 6.4902. Policy iteration from the coin flip: greedy (L,R,R) with
  J = 3.7805, then (R,R,R).

**The line** (Chapter 1, Section 8): x' = x + u + w, w ~ N(0, 0.5);
r(x,u) = -(x^2 + u^2); final reward -x_2^2; x_0 ~ N(2, 1); two moves. In the
matrix notation F = B = C_x = C_u = C_T = 1, Sigma_w = 0.5, mu_0 = 2,
Sigma_0 = 1.

  Reference numbers: policy u = -0.5 x + e, e ~ N(0, 0.1): J = -9.825; the same
  without jitter: J = -9.375; optimal (Riccati): K_1 = 0.5, K_0 = 0.6,
  P_2 = 1, P_1 = 1.5, P_0 = 1.6, beta_1 = 0.5, beta_0 = 1.25, J* = -9.25.
  Endless, gamma = 0.9, one gain: K* = 0.5884, J* = -15.0898.

Later chapters add **the weak motor** (Chapter 9: x' = x + tanh(u) + w), **the
hill** (Chapter 19: x' = x + u + sin x + w) and **the lost robot** (Chapter
22: the track with a noisy cell sensor).

Reusable code: the generic elimination routine with pluggable semirings is in
`tools/src/chapter02_examples.py`; Stage 1 with the module, the gradient, the
Fisher matrix and the second-order semiring are in
`tools/src/chapter05_examples.py`.

A new example is introduced as Chapter 1, Section 7, does it: the problem in
words, then its tables or formulas, then the graph.

## 5. The companion notebook

Write `tools/src/chapterNN_examples.py` in percent format (see the docstring
of `tools/nb.py`) and build it with `tools/nb.py`.

- The first cell is the markdown title cell `# Chapter NN examples: ...` with a
  one-paragraph description linking to
  `https://thduynguyen.github.io/gtsam/chapterNN`. The second cell is the
  imports. Markdown cells in between name the chapter section they belong to.
- Only numpy, scipy, plotly and gtsam. Plots, if any, with plotly.
- **Use the module for every exact computation.** Build the problem as a
  `SemiringFactorGraph` and eliminate it, with a `SemiringRules` object when
  some variables are summed out by a maximum or a tilted mean (see
  `appendix_a.md`, and `tools/src/chapter02_examples.py` and
  `chapter04_examples.py` for the idiom). Read values, advantages, regrets and
  policies from the factors and conditionals (`value()`, `surprise()`,
  `greedy()`, `tilted()`, `marginalFactor`). Use numpy only for what the
  module cannot do: drawing samples, fitting a learned function, array
  bookkeeping, and independent checks of the module's numbers.
- Fix every random seed (`np.random.default_rng(0)`). Keep the run time under
  about a minute.
- Pin every number quoted in the chapter with an `assert` (`np.isclose` /
  `np.allclose`; for sampled estimates, assert a tolerance that the fixed seed
  satisfies and say in the text that the number is from one seeded run).
- The exact special case of the template is an assert against the reference
  numbers of Section 4.
- With the module: see `chapter01.md` Section 10, `appendix_a.md` and
  `python/gtsam/tests/test_SemiringFactorGraph.py` for the Python API.

## 6. Figures

Write `tools/src/fig_chapterNN.py` using `tools/bookfig.py` (read its
docstring). Look at each PNG preview and fix overlaps. Keep the visual
language: blue state, green action, purple parameter, yellow observation,
black probability factor, orange value factor, teal learned factor, white
sampled factor, dashed red ring for the variable being eliminated. At least
one figure per chapter (the graph of part 1). Reference them as
`![Alt text](figures/Name.svg)`.

## 7. Checking

`python3 tools/lint.py chapterNN.md` must print OK. Then re-read the chapter
once as the reader of Section 1 and fix what that reader would stumble on: an
undefined symbol, a claim without a formula, a number that is not in the
notebook output. After adding a chapter, run `tools/assemble.py`, build the
site and run `tools/check_site.py`.

## 8. References

End each chapter with a short reference list: the original papers of the
algorithms, and one textbook pointer where useful, with authors, title, venue
and year. Label the section `(chapterNN-references)=`.
