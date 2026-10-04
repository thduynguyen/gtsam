"""Generate the figures of the worked example (a robot on a three-cell track)."""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, os.pardir, "figures")
W, H = 860, 400
INK, EDGE, MUTED = "#24292f", "#57606a", "#8c959f"
VALUE, VALUE_TEXT = "#d9730d", "#a85a08"
FONT = 'font-family="Helvetica, Arial, sans-serif"'

POS = {
    "s0": (100, 130), "s1": (400, 130), "s2": (700, 130),
    "a0": (250, 290), "a1": (550, 290),
}
PRIOR, FINAL = (100, 50), (790, 130)
DYNAMICS = {0: (250, 130), 1: (550, 130)}
POLICY = {0: (175, 210), 1: (475, 210)}
REWARD = {0: (100, 290), 1: (400, 290)}
SUB = str.maketrans("012", "₀₁₂")
DISPLAY = {"s": "s", "a": "a"}  # letters shown for states and actions


def disp(name):
    """Displayed name of a variable, e.g. s1 becomes x₁ in the line example."""
    return DISPLAY[name[0]] + name[1:].translate(SUB)


def line(p, q):
    return (f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" '
            f'stroke="{EDGE}" stroke-width="2"/>')


def square(center, color, size=16):
    x, y = center[0] - size / 2, center[1] - size / 2
    return f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{color}"/>'


def variable(name, kind):
    center = POS[name]
    fill, stroke, color = {
        "state": ("#ddebff", "#1a4fa0", INK),
        "action": ("#dff3e4", "#1e7b34", INK),
        "done": ("#f3f4f6", MUTED, MUTED),
    }[kind]
    dash = ' stroke-dasharray="4 3"' if kind == "done" else ""
    return (f'<circle cx="{center[0]}" cy="{center[1]}" r="24" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"{dash}/>'
            f'<text x="{center[0]}" y="{center[1] + 6}" font-size="18" '
            f'font-style="italic" text-anchor="middle" fill="{color}">'
            f'{disp(name)}</text>')


def ring(name):
    center = POS[name]
    return (f'<circle cx="{center[0]}" cy="{center[1]}" r="33" fill="none" '
            f'stroke="#cf222e" stroke-width="2" stroke-dasharray="6 4"/>')


def text(x, y, content, color=INK, anchor="middle", size=13):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}">{content}</text>')


def head(tip, direction):
    norm = math.hypot(*direction)
    ux, uy = direction[0] / norm, direction[1] / norm
    back = (tip[0] - 11 * ux, tip[1] - 11 * uy)
    left = (back[0] - 4.5 * uy, back[1] + 4.5 * ux)
    right = (back[0] + 4.5 * uy, back[1] - 4.5 * ux)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (tip, left, right))
    return f'<polygon points="{points}" fill="{MUTED}"/>'


def straight_arrow(source, target):
    """Arrow between the rims of two variable nodes."""
    (x0, y0), (x1, y1) = POS[source], POS[target]
    norm = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / norm, (y1 - y0) / norm
    start = (x0 + 24 * ux, y0 + 24 * uy)
    tip = (x1 - 25 * ux, y1 - 25 * uy)
    end = (tip[0] - 6 * ux, tip[1] - 6 * uy)
    return (f'<path d="M {start[0]:.1f} {start[1]:.1f} L {end[0]:.1f} {end[1]:.1f}" '
            f'fill="none" stroke="{MUTED}" stroke-width="2"/>' + head(tip, (ux, uy)))


def arc_arrow(source, target):
    """Arrow arching over the factor between two neighboring states."""
    (x0, y0), (x1, y1) = POS[source], POS[target]
    control = ((x0 + x1) / 2, y0 - 90)
    tip = (x1 - 19, y1 - 15)
    direction = (tip[0] - control[0], tip[1] - control[1])
    norm = math.hypot(*direction)
    end = (tip[0] - 6 * direction[0] / norm, tip[1] - 6 * direction[1] / norm)
    return (f'<path d="M {x0 + 22} {y0 - 16} Q {control[0]} {control[1]} '
            f'{end[0]:.1f} {end[1]:.1f}" fill="none" stroke="{MUTED}" '
            f'stroke-width="2"/>' + head(tip, direction))


LEGEND = f'''
  <line x1="16" y1="350" x2="{W - 16}" y2="350" stroke="#d0d7de"/>
  <g font-size="12" fill="{EDGE}">
    <rect x="24" y="365" width="12" height="12" fill="{INK}"/>
    <text x="42" y="375">(p, 0): probability factor</text>
    <rect x="206" y="365" width="12" height="12" fill="{VALUE}"/>
    <text x="224" y="375">(1, v): value factor</text>
    <circle cx="362" cy="371" r="8" fill="none" stroke="#cf222e" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="377" y="375">eliminated next</text>
    <circle cx="492" cy="371" r="8" fill="#f3f4f6" stroke="{MUTED}" stroke-width="2" stroke-dasharray="3 2"/>
    <text x="507" y="375">eliminated; arrows come from the parents of its conditional</text>
  </g>'''


def stage(eliminated):
    """Draw the graph after eliminating the first `eliminated` variables.

    The elimination order is s2, a1, s1, a0, s0.
    """
    order = ["s2", "a1", "s1", "a0", "s0"]
    done = set(order[:eliminated])
    edges, nodes, labels = [], [], []

    def factor(center, color, neighbors, label, label_at, anchor="middle"):
        for name in neighbors:
            edges.append(line(center, POS[name]))
        nodes.append(square(center, color))
        labels.append(text(label_at[0], label_at[1], label,
                           VALUE_TEXT if color == VALUE else INK, anchor))

    # Original factors that have not been absorbed yet.
    if "s0" not in done:
        factor(PRIOR, INK, ["s0"], f"(p({disp('s0')}), 0)", (116, 55), "start")
    for t in (0, 1):
        s, a, n = f"s{t}", f"a{t}", f"s{t + 1}"
        S, A, N = disp(s), disp(a), disp(n)
        if n not in done:
            factor(DYNAMICS[t], INK, [s, a, n], f"(p({N} | {S}, {A}), 0)",
                   (DYNAMICS[t][0], 108))
        if a not in done:
            factor(POLICY[t], INK, [s, a], f"(π({A} | {S}), 0)",
                   (POLICY[t][0] + 11, POLICY[t][1] + 34), "end")
            factor(REWARD[t], VALUE, [s, a], f"(1, r({S}, {A}))",
                   (REWARD[t][0], 322))
    if "s2" not in done:
        factor(FINAL, VALUE, ["s2"], f"(1, r({disp('s2')}))", (790, 108))

    # New factors produced by elimination and not yet absorbed.
    for t in (1, 0):
        s, a, n = f"s{t}", f"a{t}", f"s{t + 1}"
        S, A = disp(s), disp(a)
        if n in done and a not in done:
            factor(DYNAMICS[t], VALUE, [s, a], f"φ({S}, {A})",
                   (DYNAMICS[t][0], 108))
        if a in done and s not in done:
            factor(REWARD[t], VALUE, [s],
                   f"φ({S}) = (1, V{str(t).translate(SUB)})",
                   (REWARD[t][0], 322))
    if "s0" in done:
        nodes.append(square(REWARD[0], VALUE))
        labels.append(text(100, 322, "(1, J): the expected return", VALUE_TEXT))

    # Conditionals of the eliminated variables, drawn as arrows.
    arrows = []
    for t in (1, 0):
        s, a, n = f"s{t}", f"a{t}", f"s{t + 1}"
        if n in done:
            arrows += [arc_arrow(s, n), straight_arrow(a, n)]
            labels.append(text(POS[n][0], 82,
                               f"c({disp(n)} | {disp(s)}, {disp(a)})", EDGE))
        if a in done:
            arrows.append(straight_arrow(s, a))
            labels.append(text(POS[a][0], 337, f"c({disp(a)} | {disp(s)})",
                               EDGE))
    if "s0" in done:
        labels.append(text(100, 82, f"c({disp('s0')})", EDGE))

    variables = [variable(name, "done" if name in done else
                          ("state" if name[0] == "s" else "action"))
                 for name in POS]
    marker = [ring(order[eliminated])] if eliminated < len(order) else []
    return "\n  ".join(edges + arrows + nodes + marker + variables + labels)


def write(name, title, desc, body, height=H, legend=LEGEND):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" role="img" aria-labelledby="title desc" {FONT}>
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="10" fill="#ffffff" stroke="#d0d7de"/>
  {body}{legend}
</svg>
'''
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)


for prefix, letters in (("Track", {"s": "s", "a": "a"}),
                        ("Line", {"s": "x", "a": "u"})):
    DISPLAY = letters
    order = [disp(name) for name in ("s2", "a1", "s1", "a0", "s0")]
    for k in range(6):
        title = ("The factor graph of the example" if k == 0 else
                 f"After eliminating {order[k - 1]}")
        desc = ("Two-step factor graph with a prior, and for each step a "
                "policy, a reward and a dynamics factor, plus a final reward."
                if k == 0 else
                f"The graph after eliminating {', '.join(order[:k])}: each "
                "eliminated variable keeps a conditional, drawn as arrows from "
                "its parents, and the newest value factor is shown in orange.")
        write(f"{prefix}Elimination{k}.svg", title, desc, stage(k))
DISPLAY = {"s": "s", "a": "a"}


# The world itself.
def cell(i, label):
    x = 170 + 170 * i
    return (f'<rect x="{x}" y="90" width="170" height="110" fill="#f6f8fa" '
            f'stroke="{INK}" stroke-width="2"/>'
            + text(x + 85, 222, f"cell {i}", EDGE, size=13)
            + "".join(text(x + 85, 135 + 20 * j, part, color, size=13)
                      for j, (part, color) in enumerate(label)))


world = "\n  ".join([
    cell(0, [("start here", INK), ("with probability 0.5", EDGE)]),
    cell(1, [("start here", INK), ("with probability 0.5", EDGE)]),
    cell(2, [("charger", VALUE_TEXT), ("+10 if here at the end", VALUE_TEXT)]),
    # walls
    f'<line x1="170" y1="80" x2="170" y2="210" stroke="{INK}" stroke-width="6"/>',
    f'<line x1="680" y1="80" x2="680" y2="210" stroke="{INK}" stroke-width="6"/>',
    # actions
    f'<path d="M 340 55 L 250 55" stroke="{INK}" stroke-width="2" fill="none"/>',
    f'<polygon points="238,55 252,49 252,61" fill="{INK}"/>',
    text(230, 60, "Left: reward 0", INK, "end"),
    f'<path d="M 510 55 L 600 55" stroke="{INK}" stroke-width="2" fill="none"/>',
    f'<polygon points="612,55 598,49 598,61" fill="{INK}"/>',
    text(620, 60, "Right: reward −1", INK, "start"),
    text(425, 60, "two moves", EDGE),
    text(425, 258, "A move succeeds with probability 0.8. "
         "Otherwise, or against a wall, the robot stays where it is.", INK),
])
write("TrackWorld.svg", "A robot on a three-cell track",
      "Three cells in a row between two walls. The robot starts in cell 0 or "
      "cell 1 with equal probability and makes two moves, left or right. "
      "Moving right costs 1. A move succeeds with probability 0.8. Ending in "
      "cell 2, the charger, pays 10.", world, height=285, legend="")
print("written")
