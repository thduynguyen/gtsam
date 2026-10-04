# Tools for the semiring book

The scripts that generate and check the book in `gtsam/semiring/doc`. They are
not part of the book itself: `myst.yml` lists the pages, and nothing in this
directory is among them.

| File | What it does |
|---|---|
| `WRITING_GUIDE.md` | The style rules, the chapter template and the running examples. Read it before writing or editing a chapter. |
| `src/chapterNN_examples.py` | The source of the companion notebook of chapter NN, in "percent" format. |
| `src/fig_chapterNN*.py` | The scripts that draw the figures of chapter NN. |
| `bookfig.py` | A small library for drawing factor graphs as SVG, in the visual language of the book. |
| `nb.py` | Builds a notebook from its source, adds the standard preamble, executes it and prints its outputs. |
| `lint.py` | Checks a chapter without building the site: every formula in KaTeX, dropdowns, links, figures. |
| `assemble.py` | Regenerates `myst.yml`, `index.md`, `footer.md`, the work-in-progress notice and the footer of every page, and the chapter list of the module's README, from the chapter headings. The copyright line and the notice are constants at its top. |
| `check_site.py` | Checks every page of a built site for maths errors and leftovers of raw TeX. |

All commands below are run from the repository root.

## Figures

```
python gtsam/semiring/doc/tools/src/fig_chapter05.py
```

writes the SVGs into `doc/figures/` and, if `pixi` is available, PNG previews
into `tools/png/` (ignored by git). Some scripts need numpy and scipy.

The figures of chapter 1 are drawn by three standalone scripts,
`fig_chapter01_backup.py`, `fig_chapter01_examples.py` and
`fig_chapter01_line.py`; the three "optimal backup" figures used in chapters 4
and 6 by `fig_chapter04_backup.py`. `MdpFactorGraph.svg` was written by hand
and has no script.

## Notebooks

The notebooks are generated: edit `src/chapterNN_examples.py`, never the
`.ipynb`. With a Python that has `nbformat`, `nbclient`, `ipykernel`, `numpy`,
`scipy` and `plotly`, and a GTSAM build that contains the semiring module:

```
PYTHONPATH=build/python python gtsam/semiring/doc/tools/nb.py chapter05_examples
```

Each notebook asserts every number its chapter quotes, so a successful run is
the check that the text and the code agree. Notebooks that do not import
`gtsam` need only numpy, scipy and plotly.

## Checking a chapter

Once:

```
pixi exec --spec nodejs -- npm install --prefix gtsam/semiring/doc/tools katex
```

Then:

```
python3 gtsam/semiring/doc/tools/lint.py chapter05.md
```

## Building the book

```
python3 gtsam/semiring/doc/tools/assemble.py
cd gtsam/semiring/doc
pixi exec --spec nodejs --spec mystmd -- myst build --html
python3 tools/check_site.py
```

`assemble.py` must be run after a chapter is added or its title changes. The
build output is in `doc/_build/`, which git ignores.
