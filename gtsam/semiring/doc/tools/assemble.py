"""Assemble the semiring book: table of contents, landing page, footers.

Run after chapters change:  python3 gtsam/semiring/doc/tools/assemble.py

It reads the title of every chapter from its first heading and rewrites
  - myst.yml (the table of contents, grouped in parts),
  - index.md (the landing page),
  - the work-in-progress notice under the title of every page,
  - the "Previous / Next" footer of every chapter and appendix,
  - the chapter table of gtsam/semiring/README.md (between two markers).
Chapters that do not exist yet are left out.

The author, the copyright line, the credit for AI assistance and the
work-in-progress notice are the constants AUTHOR, COPYRIGHT, AI_CREDIT and
NOTICE below. To remove the notice from every page, set NOTICE to None and
run this script.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.dirname(HERE) + "/"
README = os.path.join(os.path.dirname(os.path.dirname(HERE)), "README.md")
SITE = "https://thduynguyen.github.io/gtsam/"

PARTS = [
    ("Part I: Exact elimination on decision graphs", range(1, 6),
     "The factor graph of a decision problem, the semirings that elimination "
     "can run on, and the two-stage framework that organizes the rest."),
    ("Part II: Gaussian elimination and stochastic optimal control",
     range(6, 11),
     "Known dynamics. Linear-quadratic control as Gaussian elimination, then "
     "nonlinear dynamics by relinearization and by sampling."),
    ("Part III: Sampled and learned messages: model-free RL", range(11, 18),
     "Unknown dynamics. The same two stages, with the forward and backward "
     "messages estimated from sampled transitions."),
    ("Part IV: Learned factors: model-based RL", range(18, 22),
     "The dynamics factor itself is learned from data, and elimination runs "
     "on the learned factor."),
    ("Part V: Beyond the clean story", range(22, 27),
     "Hidden state, unknown rewards, exploration and return distributions: "
     "where elimination needs more structure, or is not enough."),
]
APPENDICES = ["appendix_a.md", "appendix_b.md", "appendix_c.md"]
LICENSE = "LICENSE.md"

AUTHOR = "Duy Ta"
COPYRIGHT = f"Copyright © 2026 {AUTHOR}. All rights reserved."
# The disclosure of AI assistance, shown in the footer of every page.
AI_CREDIT = "Written with the assistance of Claude (Anthropic)."
# Shown in bold red under the title of every page (see custom.css).
NOTICE = ("**This book is a work in progress.** It is still being written "
          "and revised: its content is incomplete and may contain errors.")
NOTICE_BLOCK = re.compile(r"\n:::\{div\}\n:class: in-progress\n.*?\n:::\n", re.S)


def notice():
    return f"\n:::{{div}}\n:class: in-progress\n{NOTICE}\n:::\n"

SUMMARY = {
    1: "A Markov decision process as a factor graph; why ordinary elimination "
       "cannot evaluate it; factor entries that carry a probability and a "
       "value; the correspondence between RL quantities and elimination.",
    2: "Sum-product, expectation, max-sum, tilted and soft-maximum semirings; "
       "the axioms elimination needs; division, conditionals and their "
       "normalization invariant; the log-dual form.",
    3: "The discount as a termination factor; a chain of identical steps; the "
       "Bellman equation as a fixed point of the backward message; the "
       "forward message.",
    4: "Actions as decisions eliminated by a maximum; why the elimination "
       "order then matters; dynamic programming, value iteration and policy "
       "iteration; a shared parameter as one global decision.",
    5: "The policy gradient as forward message times local derivative times "
       "backward message; the second-order semiring; the Fisher matrix; the "
       "formal definition of Stage 1 and Stage 2.",
    6: "The Riccati recursion as Gaussian elimination with a maximum over the "
       "action; the Kalman filter as the forward pass; LQG and the separation "
       "principle.",
    7: "A linear-Gaussian policy evaluated exactly, its exact gradient, and "
       "Stage 2 as gradient, Gauss-Newton and trust-region steps.",
    8: "Risk-sensitive control, linearly solvable MDPs and control as "
       "inference; the optimism problem, and how a fixed dynamics factor "
       "avoids it.",
    9: "iLQR and DDP as relinearize-then-eliminate; AICO; the contrast with "
       "MAP trajectory optimization; model predictive control.",
    10: "MPPI and path-integral control as a sampled soft maximum, and CEM as "
        "a sampled maximum.",
    11: "Rollouts as particles for the forward message and returns for the "
        "backward message; REINFORCE; baselines as control variates.",
    12: "TD(0), TD(lambda) and GAE as local consistency of the backward "
        "message; function approximation and the deadly triad.",
    13: "A learned backward message, the critic, combined with a gradient "
        "step on the policy.",
    14: "The performance difference lemma; natural policy gradient, TRPO and "
        "PPO as ways to reuse messages safely; the analogy with "
        "relinearization.",
    15: "SARSA, Q-learning, fitted Q iteration and DQN; data collected by "
        "another policy.",
    16: "Deterministic policy gradients: DPG, DDPG and TD3.",
    17: "SAC, REPS, MPO and AWR: the soft maximum over actions, and policy "
        "improvement as expectation-maximization.",
    18: "System identification as factor-graph learning; uncertainty about "
        "the model through ensembles and Gaussian processes.",
    19: "Moment-matched Gaussian forward messages through a Gaussian-process "
        "dynamics model.",
    20: "Local trajectory optimizers and one global policy, coordinated by "
        "dual decomposition.",
    21: "PETS, MBPO, Dreamer and TD-MPC2: planning and learning inside a "
        "learned model, and compounding model error.",
    22: "POMDPs; the belief as a forward message; the separation principle "
        "revisited.",
    23: "Maximum-entropy inverse RL as learning the reward factors.",
    24: "Why deciding where to sample is not an elimination; the "
        "Bayes-optimal policy on a small problem, and practical heuristics.",
    25: "The full distribution of the return, carried by a convolution "
        "semiring.",
    26: "PPO for legged locomotion, SAC and TD3 on hardware, MPC with learned "
        "components, and residual RL, each mapped onto the framework.",
}
APPENDIX_SUMMARY = {
    "appendix_a.md": "The classes of the `gtsam/semiring` module, how they "
                     "plug into GTSAM's elimination, and the tests of the "
                     "semiring laws.",
    "appendix_b.md": "Every symbol of the book, with the names used "
                     "elsewhere.",
    "appendix_c.md": "Every algorithm of the book, with its choices along "
                     "the five axes of the framework.",
}


def title_of(name):
    with open(DOC + name) as f:
        return f.readline().strip().lstrip("# ").strip()


def chapter(n):
    return f"chapter{n:02d}.md"


def exists(name):
    return os.path.exists(DOC + name)


def notebook(n):
    name = f"chapter{n:02d}_examples.ipynb"
    return name if exists(name) else None


def write_toc():
    lines = ["# The book of semiring factor graphs: build with",
             "#   myst build --html     (or: myst start)",
             "# The table of contents below is generated by",
             "# tools/assemble.py from the chapter headings.",
             "version: 1", "project:",
             "  title: Semiring factor graphs",
             "  description: Optimal control and reinforcement learning as "
             "variable elimination on factor graphs, with GTSAM.",
             "  github: https://github.com/thduynguyen/gtsam",
             "  authors:", f"    - name: {AUTHOR}",
             "  # Formulas are not referred to by number, so do not number them.",
             "  numbering:", "    equation:", "      enabled: false",
             "  toc:", "    - file: index.md"]
    for part, numbers, _ in PARTS:
        present = [n for n in numbers if exists(chapter(n))]
        if not present:
            continue
        lines += [f'    - title: "{part}"', "      children:"]
        for n in present:
            lines.append(f"        - file: {chapter(n)}")
            if notebook(n):
                lines += ["          children:",
                          f"            - file: {notebook(n)}"]
    present = [name for name in APPENDICES if exists(name)]
    if present:
        lines += ["    - title: Appendices", "      children:"]
        lines += [f"        - file: {name}" for name in present]
    if exists(LICENSE):
        lines.append(f"    - file: {LICENSE}")
    lines += ["site:", "  template: book-theme",
              "  title: Semiring factor graphs", "  options:",
              "    style: custom.css", "  parts:",
              "    footer: footer.md", ""]
    with open(DOC + "footer.md", "w") as f:
        f.write(f"{COPYRIGHT} {AI_CREDIT} The example code is under the "
                f"BSD license of GTSAM. See [Copyright, license and "
                f"authorship]({LICENSE}).\n")
    with open(DOC + "myst.yml", "w") as f:
        f.write("\n".join(lines))


def short(title):
    """'Chapter 3: Infinite horizon' -> 'Infinite horizon'."""
    return re.sub(r"^(Chapter \d+|Appendix [A-C]): ", "", title)


def write_index():
    out = ["# Semiring factor graphs", ""]
    if NOTICE:
        out += notice().strip("\n").split("\n") + [""]
    out += ["This book expresses optimal control and reinforcement learning "
           "(RL) as\noperations on factor graphs, for readers who know factor "
           "graphs from SLAM\nand have not met control or RL before.", "",
           "It accompanies the `gtsam/semiring` module of GTSAM, in which "
           "every factor\nentry carries two numbers, a probability and a "
           "value. Ordinary variable\nelimination on such factors computes "
           "the quantities that control and RL are\nbuilt on: the expected "
           "return of a policy, its value functions, and its\nadvantages.", "",
           "The book has one organizing idea, the **two-stage framework** of\n"
           "[Chapter 5](chapter05.md). Stage 1 eliminates the states and "
           "actions of the\ngraph with the policy held fixed, which produces "
           "*backward messages* (values\nand advantages) and *forward "
           "messages* (how often each state is visited).\nStage 2 uses those "
           "messages to improve the policy. Every algorithm in the\nbook is "
           "a choice of how to carry out each stage, and each chapter after "
           "the\nfifth follows the same template:", "",
           "1. the graph: its variables, its factors and the policy "
           "parameters;",
           "2. Stage 1: which semiring, how the dynamics factor is accessed, "
           "how the\n   messages are computed;",
           "3. Stage 2: how the policy is updated;",
           "4. an exact special case that serves as a correctness test;",
           "5. an implementation, with the module or in numpy;",
           "6. what breaks.", "",
           "The [taxonomy table](appendix_c.md) lists every algorithm with "
           "its choices."
           if exists("appendix_c.md") else "", ""]
    for part, numbers, blurb in PARTS:
        present = [n for n in numbers if exists(chapter(n))]
        if not present:
            continue
        out += [f"## {part}", "", blurb, ""]
        for n in present:
            line = (f"{n}. [{short(title_of(chapter(n)))}]({chapter(n)}). "
                    f"{SUMMARY[n]}")
            if notebook(n):
                line += f" [Run the examples]({notebook(n)})."
            out.append(line)
        out.append("")
    present = [name for name in APPENDICES if exists(name)]
    if present:
        out += ["## Appendices", ""]
        for name in present:
            out.append(f"- [{title_of(name)}]({name}). "
                       f"{APPENDIX_SUMMARY[name]}")
        out.append("")
    out += ["## Running the examples", "",
            "Each chapter has a companion notebook with the code of its "
            "examples, executed,\nwith every number quoted in the chapter "
            "checked by an assertion. The notebooks\nopen in Google Colab "
            "from the badge at the top of each chapter. Those that use\nthe "
            "module install a GTSAM build that contains it in their first "
            "cell; the\nothers need only numpy.", "",
            "The source of the module, the chapters and the notebooks is in\n"
            "[`gtsam/semiring`](https://github.com/thduynguyen/gtsam/tree/"
            "feature/semiringfactor/gtsam/semiring).", "",
            "## Author, copyright and license", "",
            f"This book is written by {AUTHOR}, with the assistance of "
            "Claude, an AI model made by\nAnthropic. The author defined its "
            "scope and structure, directed and revised its\nexplanations, "
            "and is responsible for its content.", "",
            f"{COPYRIGHT} The text and figures of this book may not be "
            "reproduced or\nredistributed without permission. The example "
            "code in the notebooks and tools is\nunder the BSD license of "
            f"GTSAM. See [Copyright, license and authorship]({LICENSE}).", ""]
    with open(DOC + "index.md", "w") as f:
        f.write("\n".join(out))


FOOTER = re.compile(r"\n+---\n+(Previous|Next): .*\Z", re.S)


def write_footers():
    pages = [chapter(n) for _, numbers, _ in PARTS for n in numbers
             if exists(chapter(n))]
    pages += [name for name in APPENDICES if exists(name)]
    for i, name in enumerate(pages):
        text = open(DOC + name).read()
        # The work-in-progress notice, directly under the title.
        text = NOTICE_BLOCK.sub("", text, count=1)
        if NOTICE:
            title, rest = text.split("\n", 1)
            text = title + "\n" + notice() + rest
        text = FOOTER.sub("", text).rstrip("\n")
        links = []
        if i > 0:
            links.append(f"Previous: [{title_of(pages[i - 1])}]"
                         f"({pages[i - 1]}).")
        if i + 1 < len(pages):
            links.append(f"Next: [{title_of(pages[i + 1])}]({pages[i + 1]}).")
        with open(DOC + name, "w") as f:
            f.write(text + "\n\n---\n\n" + "\n".join(links) + "\n")


BEGIN, END = "<!-- chapters:begin -->", "<!-- chapters:end -->"


def write_readme():
    text = open(README).read()
    if BEGIN not in text:
        return
    rows = []
    for part, numbers, _ in PARTS:
        present = [n for n in numbers if exists(chapter(n))]
        if not present:
            continue
        rows.append(f"**{part}**\n")
        for n in present:
            row = f"- {n}. [{short(title_of(chapter(n)))}](doc/{chapter(n)})"
            if notebook(n):
                row += f" ([notebook](doc/{notebook(n)}))"
            rows.append(row)
        rows.append("")
    present = [name for name in APPENDICES if exists(name)]
    if present:
        rows.append("**Appendices**\n")
        rows += [f"- [{short(title_of(name))}](doc/{name})"
                 for name in present]
        rows.append("")
    head, rest = text.split(BEGIN)
    tail = rest.split(END)[1]
    with open(README, "w") as f:
        f.write(head + BEGIN + "\n" + "\n".join(rows) + END + tail)


if __name__ == "__main__":
    write_toc()
    write_index()
    write_footers()
    write_readme()
    print("assembled:", sum(exists(chapter(n)) for n in range(1, 27)),
          "chapters,", sum(exists(a) for a in APPENDICES), "appendices")
