"""Check chapters of the semiring book without building the site.

Usage:  python3 gtsam/semiring/doc/tools/lint.py chapter05.md [more.md ...]

Needs KaTeX next to this script, installed once with
    pixi exec --spec nodejs -- npm install --prefix gtsam/semiring/doc/tools katex
and `pixi` on the path to run node.

Checks, for each file in gtsam/semiring/doc:
  - every formula ($...$ and $$...$$) typesets in KaTeX without an error;
  - dropdowns (:::{dropdown} ... :::) and code fences are closed;
  - no raw HTML (<details>, <br>, <div>), which MyST does not typeset;
  - every figure and every linked chapter or notebook exists;
  - no "|" inside inline maths in a table row (it would split the cell;
    write \\mid instead).
Prints "OK" or the problems with line numbers. Exit status 1 on problems.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.dirname(HERE) + "/"
REPO = os.path.abspath(os.path.join(HERE, *[os.pardir] * 4))

KATEX_JS = r"""
const katex = require(process.argv[2] + '/node_modules/katex');
const items = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const errors = [];
for (const [line, display, tex] of items) {
  try {
    katex.renderToString(tex, {displayMode: display, throwOnError: true, strict: false});
  } catch (e) { errors.push([line, tex.slice(0, 90), String(e.message).slice(0, 160)]); }
}
console.log(JSON.stringify(errors));
"""


def formulas(lines):
    """Yield (line number, display, tex) for every formula outside code."""
    text, fenced, out = [], False, []
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            text.append((number, ""))
            continue
        text.append((number, "" if fenced else re.sub(r"`[^`]*`", "", line)))
    joined = "\n".join(line for _, line in text)
    starts = [0]
    for _, line in text:
        starts.append(starts[-1] + len(line) + 1)

    def line_of(offset):
        low = 0
        while low + 1 < len(starts) and starts[low + 1] <= offset:
            low += 1
        return low + 1

    spans = []
    for match in re.finditer(r"\$\$(.+?)\$\$", joined, re.S):
        out.append((line_of(match.start()), True, match.group(1)))
        spans.append(match.span())
    blanked = list(joined)
    for a, b in spans:
        for i in range(a, b):
            if blanked[i] != "\n":
                blanked[i] = " "
    blanked = "".join(blanked)
    inline = r"(?<![\\$])\$((?:[^$\n]|\n(?!\s*\n))+?)\$"
    for match in re.finditer(inline, blanked):
        out.append((line_of(match.start()), False, match.group(1)))
    leftover = re.sub(inline, lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                      blanked)
    stray = [line_of(m.start()) for m in re.finditer(r"(?<!\\)\$", leftover)]
    return out, stray


def check(name):
    path = os.path.join(DOC, name)
    lines = open(path).read().split("\n")
    problems, pending = [], set()
    maths, stray = formulas(lines)
    for number in stray:
        problems.append((number, "unpaired $"))
    depth, fenced = 0, False
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if re.match(r"^:::+\{", stripped):
            depth += 1
        elif re.match(r"^:::+$", stripped):
            depth -= 1
            if depth < 0:
                problems.append((number, "closing ::: without an opening"))
                depth = 0
        if re.search(r"<(details|summary|br|div|span)\b", line):
            problems.append((number, "raw HTML: use MyST syntax"))
        if "\t" in line:
            problems.append((number, "tab character"))
        if stripped.startswith("|"):
            for tex in re.findall(r"(?<![\\$])\$([^$\n]+?)\$", line):
                if "|" in tex.replace(r"\|", ""):
                    problems.append((number, f"'|' in table maths: ${tex}$ "
                                     "(write \\mid or \\vert)"))
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", line):
            if re.match(r"^(https?:|mailto:)", target):
                continue
            if not os.path.exists(os.path.join(DOC, target)):
                planned = re.fullmatch(
                    r"(chapter(0[1-9]|1\d|2[0-6])(_examples\.ipynb|\.md)"
                    r"|appendix_[abc]\.md)", target)
                if planned:
                    pending.add(target)
                else:
                    problems.append((number, f"missing file: {target}"))
    if depth:
        problems.append((len(lines), f"{depth} dropdown(s) not closed"))
    if fenced:
        problems.append((len(lines), "code fence not closed"))

    items = os.path.join(HERE, f".lint_{os.path.basename(name)}.json")
    script = os.path.join(HERE, "katex_check.js")
    with open(items, "w") as f:
        json.dump(maths, f)
    if not os.path.exists(script):
        with open(script, "w") as f:
            f.write(KATEX_JS)
    result = subprocess.run(
        ["pixi", "exec", "--spec", "nodejs", "--", "node", script, HERE, items],
        cwd=REPO, capture_output=True, text=True)
    os.remove(items)
    try:
        for number, tex, message in json.loads(result.stdout.strip().split("\n")[-1]):
            problems.append((number, f"KaTeX: {message} in: {tex}"))
    except Exception:
        problems.append((0, "could not run KaTeX: " + result.stderr[-300:]))

    display = sum(1 for _, d, _ in maths if d)
    print(f"{name}: {len(lines)} lines, {len(maths)} formulas "
          f"({display} display)", "OK" if not problems else "",
          f"| links to files not written yet: {len(pending)}" if pending else "")
    for number, message in sorted(problems):
        print(f"  line {number}: {message}")
    return bool(problems)


if __name__ == "__main__":
    sys.exit(1 if any([check(name) for name in sys.argv[1:]]) else 0)
