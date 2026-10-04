"""Draw factor-graph figures for the semiring book as SVG.

Same visual language as the figures of Chapter 1:
  circles are variables (blue state, green action, purple parameter, yellow
  observation, grey dashed eliminated), squares are factors (black probability,
  orange value, teal learned, white-with-border sampled), a dashed red ring
  marks the variable being eliminated, grey arrows are a conditional.

Usage:
    from bookfig import Figure
    fig = Figure(860, 330)
    fig.variable("s0", (100, 120), "s_0", "state")
    fig.variable("a0", (250, 250), "a_0", "action")
    fig.variable("s1", (400, 120), "s_1", "state")
    fig.factor((250, 120), "probability", ["s0", "a0", "s1"],
               "p(s_1 | s_0, a_0)", "above")
    fig.factor((100, 250), "value", ["s0", "a0"], "r(s_0, a_0)", "below")
    fig.save("MyFigure.svg", "Title", "One-paragraph description for readers "
             "who cannot see the figure.")

Labels may contain subscripts written _x or _{t+1} and superscripts written ^x
or ^{*}; they are drawn with tspans. Use real Unicode for Greek letters and
primes, e.g. "π_θ(a | s)", "s′".

save() writes the SVG to gtsam/semiring/doc/figures and, if `pixi` is
available, a PNG preview to tools/png (not tracked by git) for checking the
layout.
"""
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
PNG = os.path.join(HERE, "png")
REPO = os.path.abspath(os.path.join(HERE, *[os.pardir] * 4))
INK, EDGE, MUTED = "#24292f", "#57606a", "#8c959f"
VALUE, VALUE_TEXT = "#d9730d", "#a85a08"
LEARNED, LEARNED_TEXT = "#0a7ea4", "#075e7a"
RED = "#cf222e"

VARIABLE_STYLE = {  # fill, stroke, text
    "state": ("#ddebff", "#1a4fa0", INK),
    "action": ("#dff3e4", "#1e7b34", INK),
    "parameter": ("#eadcff", "#6639ba", INK),
    "observation": ("#fff4c2", "#9a6700", INK),
    "done": ("#f3f4f6", MUTED, MUTED),
    "terminal": ("#e6e8eb", EDGE, EDGE),
}
FACTOR_STYLE = {  # fill, stroke, label colour
    "probability": (INK, INK, INK),
    "value": (VALUE, VALUE, VALUE_TEXT),
    "learned": (LEARNED, LEARNED, LEARNED_TEXT),
    "sampled": ("#ffffff", INK, INK),
}
LEGEND_TEXT = {
    "state": "state",
    "action": "action",
    "parameter": "parameter",
    "observation": "observation",
    "done": "eliminated; arrows: its conditional",
    "terminal": "ended",
    "probability": "(p, 0): probability",
    "value": "(1, v): value",
    "learned": "learned factor",
    "sampled": "factor known only through samples",
    "ring": "being eliminated",
}


def markup(label, size):
    """Convert _x, _{..}, ^x, ^{..} in a label to SVG tspans."""
    label = (label.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))
    small = round(size * 0.72, 1)
    shift = round(size * 0.3, 1)

    def script(match):
        kind, body = match.group(1), match.group(2) or match.group(3)
        dy = shift if kind == "_" else -shift * 1.4
        return (f'<tspan dy="{dy}" font-size="{small}">{body}</tspan>'
                f'<tspan dy="{-dy}" font-size="{small}">​</tspan>')

    return re.sub(r"([_^])(?:\{([^}]*)\}|(\S))", script, label)


class Figure:
    def __init__(self, width=860, height=330, legend=True):
        self.width, self.height, self.legend = width, height, legend
        self.pos, self.kind = {}, {}
        self.edges, self.squares, self.arrows = [], [], []
        self.circles, self.rings, self.texts, self.extra = [], [], [], []
        self.used = []

    def _use(self, kind):
        if kind not in self.used:
            self.used.append(kind)

    def variable(self, name, center, label, kind="state", radius=24):
        """A variable node.

        kind: state, action, parameter, observation, done (eliminated),
        terminal (an absorbing "ended" state).
        """
        fill, stroke, color = VARIABLE_STYLE[kind]
        self.pos[name], self.kind[name] = center, kind
        self._use(kind)
        dash = ' stroke-dasharray="4 3"' if kind == "done" else ""
        self.circles.append(
            f'<circle cx="{center[0]}" cy="{center[1]}" r="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
            f'<text x="{center[0]}" y="{center[1] + 6}" font-size="18" '
            f'font-style="italic" text-anchor="middle" fill="{color}">'
            f'{markup(label, 18)}</text>')

    def factor(self, center, kind, connect, label=None, where="above",
               offset=None, size=16):
        """A factor square joined to the variables named in connect.

        where: above, below, left, right; or pass offset=(dx, dy) and an
        anchor through where="start" / "middle" / "end".
        """
        fill, stroke, color = FACTOR_STYLE[kind]
        self._use(kind)
        for name in connect:
            self.edge(center, self.pos[name])
        x, y = center[0] - size / 2, center[1] - size / 2
        self.squares.append(
            f'<rect x="{x}" y="{y}" width="{size}" height="{size}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        if label:
            placement = {
                "above": (0, -16, "middle"), "below": (0, 28, "middle"),
                "left": (-16, 5, "end"), "right": (16, 5, "start"),
            }
            if offset is None:
                dx, dy, anchor = placement[where]
            else:
                dx, dy = offset
                anchor = where if where in ("start", "middle", "end") else "middle"
            self.text(center[0] + dx, center[1] + dy, label, color, anchor)

    def edge(self, p, q, dashed=False):
        """An undirected line between two points or variable names."""
        p, q = self.pos.get(p, p), self.pos.get(q, q)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.edges.append(
            f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" '
            f'stroke="{EDGE}" stroke-width="2"{dash}/>')

    def curve(self, p, control, q, dashed=False):
        """An undirected curved line from p to q, pulled toward control."""
        p, q = self.pos.get(p, p), self.pos.get(q, q)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.edges.append(
            f'<path d="M {p[0]} {p[1]} Q {control[0]} {control[1]} {q[0]} '
            f'{q[1]}" fill="none" stroke="{EDGE}" stroke-width="2"{dash}/>')

    def arrow(self, start, end, bend=0.0, color=MUTED, gap=27, dashed=False,
              start_gap=None, end_gap=None):
        """A directed arrow between two variables (names) or points.

        bend > 0 curves the arrow to its right-hand side on the screen (for
        an arrow pointing right, downward), by that fraction of its length.
        gap is the clearance left at both ends (27 clears a variable circle);
        start_gap and end_gap set them separately.
        """
        p, q = self.pos.get(start, start), self.pos.get(end, end)
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        mid = ((p[0] + q[0]) / 2 + bend * length * uy,
               (p[1] + q[1]) / 2 - bend * length * ux)
        # Shorten both ends along the tangent of the curve at that end.
        def toward(a, b, distance):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            return (a[0] + distance * (b[0] - a[0]) / d,
                    a[1] + distance * (b[1] - a[1]) / d)
        a = toward(p, mid, gap if start_gap is None else start_gap)
        tip = toward(q, mid, gap if end_gap is None else end_gap)
        tx, ty = q[0] - mid[0], q[1] - mid[1]
        norm = math.hypot(tx, ty)
        tx, ty = tx / norm, ty / norm
        back = (tip[0] - 11 * tx, tip[1] - 11 * ty)
        left = (back[0] - 4.5 * ty, back[1] + 4.5 * tx)
        right = (back[0] + 4.5 * ty, back[1] - 4.5 * tx)
        head = " ".join(f"{x:.1f},{y:.1f}" for x, y in (tip, left, right))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.arrows.append(
            f'<path d="M {a[0]:.1f} {a[1]:.1f} Q {mid[0]:.1f} {mid[1]:.1f} '
            f'{back[0]:.1f} {back[1]:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
            f'<polygon points="{head}" fill="{color}"/>')

    def ring(self, name, radius=33):
        """Mark a variable as the one being eliminated."""
        center = self.pos[name]
        self._use("ring")
        self.rings.append(
            f'<circle cx="{center[0]}" cy="{center[1]}" r="{radius}" '
            f'fill="none" stroke="{RED}" stroke-width="2" '
            f'stroke-dasharray="6 4"/>')

    def text(self, x, y, content, color=INK, anchor="middle", size=13,
             bold=False):
        weight = ' font-weight="bold"' if bold else ""
        self.texts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}"{weight}>{markup(content, size)}</text>')

    def box(self, x, y, width, height, label=None, color=MUTED, dashed=True):
        """A rounded frame around part of the graph, e.g. "Stage 1"."""
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.extra.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" '
            f'fill="none" stroke="{color}" stroke-width="1.5"{dash}/>')
        if label:
            self.text(x + 10, y + 18, label, color, "start", 12, bold=True)

    def raw(self, svg):
        """Any other SVG, drawn under the nodes."""
        self.extra.append(svg)

    def _legend(self):
        items, x, y = [], 24, self.height - 29
        for kind in self.used:
            label = LEGEND_TEXT[kind]
            if kind in VARIABLE_STYLE:
                fill, stroke, _ = VARIABLE_STYLE[kind]
                dash = ' stroke-dasharray="3 2"' if kind == "done" else ""
                items.append(f'<circle cx="{x + 8}" cy="{y}" r="8" fill="{fill}" '
                             f'stroke="{stroke}" stroke-width="2"{dash}/>')
            elif kind in FACTOR_STYLE:
                fill, stroke, _ = FACTOR_STYLE[kind]
                items.append(f'<rect x="{x + 2}" y="{y - 6}" width="12" '
                             f'height="12" fill="{fill}" stroke="{stroke}" '
                             f'stroke-width="1.5"/>')
            else:
                items.append(f'<circle cx="{x + 8}" cy="{y}" r="8" fill="none" '
                             f'stroke="{RED}" stroke-width="2" '
                             f'stroke-dasharray="4 3"/>')
            items.append(f'<text x="{x + 23}" y="{y + 4}" font-size="12" '
                         f'fill="{EDGE}">{label}</text>')
            x += 23 + 6.3 * len(label) + 22
        rule = (f'<line x1="16" y1="{self.height - 50}" x2="{self.width - 16}" '
                f'y2="{self.height - 50}" stroke="#d0d7de"/>')
        return rule + "\n  " + "\n  ".join(items)

    def save(self, name, title, desc):
        parts = (self.extra + self.edges + self.arrows + self.squares +
                 self.rings + self.circles + self.texts)
        legend = self._legend() if self.legend else ""
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}" role="img" aria-labelledby="title desc" font-family="Helvetica, Arial, sans-serif">
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <rect x="0.5" y="0.5" width="{self.width - 1}" height="{self.height - 1}" rx="10" fill="#ffffff" stroke="#d0d7de"/>
  {chr(10).join("  " + part for part in parts)}
  {legend}
</svg>
'''
        path = os.path.join(OUT, name)
        with open(path, "w") as f:
            f.write(re.sub(r"[ \t]+\n", "\n", svg))
        os.makedirs(PNG, exist_ok=True)
        png = os.path.join(PNG, name.replace(".svg", ".png"))
        try:
            subprocess.run(
                ["pixi", "exec", "--spec", "librsvg", "--", "rsvg-convert",
                 "-o", png, path], cwd=REPO, check=False, capture_output=True)
        except FileNotFoundError:  # no pixi: skip the preview
            png = "(no preview)"
        print("written", path, "| preview", png)


def chain(fig, steps, y_state=120, y_action=250, x0=100, dx=300,
          state="s", action="a", policy=True, reward=True, prior=True,
          final=True, policy_label="π(a_{t} | s_{t})", last=None):
    """Draw the standard MDP chain of Chapter 1 for the given number of steps.

    Creates variables s0..s{steps}, a0..a{steps-1} and their factors, and
    returns the positions of the factor squares by name, e.g. "dynamics0".
    Pass policy=False to leave the policy factors out.
    """
    where = {}
    for t in range(steps + 1):
        fig.variable(f"{state}{t}", (x0 + dx * t, y_state),
                     f"{state}_{t}", "state")
    for t in range(steps):
        fig.variable(f"{action}{t}", (x0 + dx * t + dx / 2, y_action),
                     f"{action}_{t}", "action")
    if prior:
        where["prior"] = (x0, y_state - 80)
        fig.factor(where["prior"], "probability", [f"{state}0"],
                   f"p({state}_0)", "right")
    for t in range(steps):
        s, a, n = f"{state}{t}", f"{action}{t}", f"{state}{t + 1}"
        centre = (x0 + dx * t + dx / 2, y_state)
        where[f"dynamics{t}"] = centre
        fig.factor(centre, "probability", [s, a, n],
                   f"p({state}_{t + 1} | {state}_{t}, {action}_{t})", "above")
        if policy:
            centre = (x0 + dx * t + dx / 4, (y_state + y_action) / 2)
            where[f"policy{t}"] = centre
            fig.factor(centre, "probability", [s, a],
                       policy_label.replace("{t}", str(t)), "left",
                       offset=(-14, 22))
        if reward:
            centre = (x0 + dx * t, y_action)
            where[f"reward{t}"] = centre
            fig.factor(centre, "value", [s, a],
                       f"r({state}_{t}, {action}_{t})", "below")
    if final:
        centre = (x0 + dx * steps + 90, y_state)
        where["final"] = centre
        fig.factor(centre, "value", [f"{state}{steps}"],
                   f"r({state}_{steps})", "above")
    return where
