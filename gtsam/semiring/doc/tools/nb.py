"""Build and execute a companion notebook of the semiring book.

Usage, from the repository root, with a Python that has nbformat, nbclient,
ipykernel, numpy, scipy and plotly, and with a GTSAM build that contains the
semiring module on the path:

    PYTHONPATH=build/python python gtsam/semiring/doc/tools/nb.py chapter05_examples

The source is tools/src/chapter05_examples.py in "percent" format:

    # %% [markdown]
    # # Chapter 5 examples: gradients by elimination
    #
    # One or two sentences saying what the notebook runs, with a link to the
    # chapter: [Chapter 5](https://thduynguyen.github.io/gtsam/chapter05).

    # %%
    import numpy as np

    # %% [markdown]
    # ## The exact gradient (Section 3)

    # %%
    ...code...

The first cell must be the markdown title cell and the second the imports.
This script inserts the repository's standard preamble between them (copyright
cell and Colab install cell tagged remove-cell, Colab badge), executes the
notebook, writes it to gtsam/semiring/doc/, and prints every text output and
error so the numbers can be checked.
"""
import os
import re
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.dirname(HERE) + "/"
REPO, BRANCH = "thduynguyen/gtsam", "feature/semiringfactor"
WHEELS = f"https://github.com/{REPO}/releases/expanded_assets/semiring-wheels"

COPYRIGHT = """GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information"""

INSTALL_GTSAM = f"""try:
    import google.colab
    # The semiring module is not in a released GTSAM yet, so install a wheel
    # built from the feature branch.
    %pip install --quiet numpy pyparsing plotly
    %pip install --quiet --no-index --no-deps --find-links {WHEELS} gtsam-develop
except ImportError:
    pass  # Not running on Colab, do nothing"""

INSTALL_NUMPY = """try:
    import google.colab
    # This notebook needs only numpy, scipy and plotly, which Colab provides.
    %pip install --quiet numpy scipy plotly
except ImportError:
    pass  # Not running on Colab, do nothing"""


def badge(name):
    url = (f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/"
           f"gtsam/semiring/doc/{name}")
    return (f'<a href="{url}" target="_parent"><img '
            'src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Open In Colab"/></a>')


def parse(source):
    """Split percent-format source into (kind, text) cells."""
    cells, kind, lines = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", line)
                             for line in text.split("\n"))
        if text.strip():
            cells.append((kind, text))

    for line in source.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        else:
            lines.append(line)
    flush()
    return cells


def build(stem):
    source = open(f"{HERE}/src/{stem}.py").read()
    cells = parse(source)
    assert cells[0][0] == "markdown" and cells[0][1].startswith("# "), \
        "the first cell must be the markdown title cell"
    assert cells[1][0] == "code", "the second cell must be the imports"
    name = stem + ".ipynb"
    uses_gtsam = bool(re.search(r"^\s*(import gtsam|from gtsam)", source, re.M))
    copyright_cell = new_markdown_cell(COPYRIGHT)
    copyright_cell.metadata["tags"] = ["remove-cell"]
    install = new_code_cell(INSTALL_GTSAM if uses_gtsam else INSTALL_NUMPY)
    install.metadata["tags"] = ["remove-cell"]
    body = [new_markdown_cell(text) if kind == "markdown"
            else new_code_cell(text) for kind, text in cells]
    notebook = new_notebook(cells=[
        body[0], copyright_cell, new_markdown_cell(badge(name)), install,
        *body[1:]])
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    for i, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{i:03d}"

    failed = False
    try:
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": DOC}}).execute()
    except Exception as error:  # report below, with the outputs so far
        failed = True
        print("EXECUTION FAILED:", str(error)[-1500:])
    # Drop execution timestamps, so that rebuilding does not change the file.
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.write(notebook, DOC + name)
    print("=====", name, len(notebook.cells), "cells")
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text.rstrip())
            elif output.output_type == "error":
                failed = True
                print("ERROR", output.ename, output.evalue)
            elif "text/plain" in output.get("data", {}) and \
                    len(output["data"]) == 1:
                print(output["data"]["text/plain"])
            else:
                print(f"[{output.output_type}: "
                      f"{', '.join(output.get('data', {}).keys())}]")
    return failed


if __name__ == "__main__":
    results = [build(stem.replace(".py", "").replace(".ipynb", ""))
               for stem in sys.argv[1:]]
    sys.exit(1 if any(results) else 0)
