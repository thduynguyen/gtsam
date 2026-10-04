"""Generate the world picture and the plots of the continuous (LQR) example."""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, os.pardir, "figures")
INK, EDGE, MUTED = "#24292f", "#57606a", "#8c959f"
BLUE, GREEN, ORANGE, RED = "#1a4fa0", "#1e7b34", "#d9730d", "#cf222e"
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def text(x, y, content, color=INK, anchor="middle", size=13, style=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}"{style}>{content}</text>')


def write(name, width, height, title, desc, body):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title desc" {FONT}>
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" fill="#ffffff" stroke="#d0d7de"/>
  {body}
</svg>
'''
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)


# ---------------------------------------------------------------------------
# The world: a robot on a line that wants to be at the origin.
W, H = 860, 300
X0, SCALE, AXIS = 230, 150, 190  # pixel of x = 0, pixels per unit, axis height


def px(x):
    return X0 + SCALE * x


parts = [f'<line x1="{px(-1.2)}" y1="{AXIS}" x2="{px(3.9)}" y2="{AXIS}" '
         f'stroke="{INK}" stroke-width="2"/>']
for tick in range(-1, 4):
    parts.append(f'<line x1="{px(tick)}" y1="{AXIS - 6}" x2="{px(tick)}" '
                 f'y2="{AXIS + 6}" stroke="{INK}" stroke-width="2"/>')
    parts.append(text(px(tick), AXIS + 24, str(tick), EDGE))
parts.append(text(px(3.9) + 14, AXIS + 5, "x", INK, style=' font-style="italic"'))

# The goal at the origin.
parts.append(f'<line x1="{px(0)}" y1="{AXIS - 62}" x2="{px(0)}" y2="{AXIS}" '
             f'stroke="{ORANGE}" stroke-width="3"/>')
parts.append(f'<polygon points="{px(0)},{AXIS - 62} {px(0) + 30},{AXIS - 52} '
             f'{px(0)},{AXIS - 42}" fill="{ORANGE}"/>')
parts.append(text(px(0), AXIS + 44, "goal: stay near x = 0", "#a85a08"))

# The start distribution, a bell curve around x = 2.
points = []
for i in range(81):
    x = -0.5 + 4.5 * i / 80
    density = math.exp(-0.5 * (x - 2.0) ** 2)
    points.append(f"{px(x):.1f},{AXIS - 2 - 70 * density:.1f}")
parts.append(f'<polyline points="{" ".join(points)}" fill="none" '
             f'stroke="{BLUE}" stroke-width="2" stroke-dasharray="5 4"/>')
parts.append(text(px(2.75), AXIS - 78, "start: x₀ ~ N(2, 1)", BLUE, "start"))

# The robot at x = 2 and one move toward the goal.
parts.append(f'<circle cx="{px(2)}" cy="{AXIS - 18}" r="16" fill="#ddebff" '
             f'stroke="{BLUE}" stroke-width="2"/>')
parts.append(f'<path d="M {px(2) - 20} {AXIS - 18} L {px(1) + 14} {AXIS - 18}" '
             f'stroke="{GREEN}" stroke-width="3" fill="none"/>')
parts.append(f'<polygon points="{px(1)},{AXIS - 18} {px(1) + 16},{AXIS - 25} '
             f'{px(1) + 16},{AXIS - 11}" fill="{GREEN}"/>')
parts.append(text(px(1.45), AXIS - 30, "move u", GREEN))

parts.append(text(W / 2, 40, "Two moves. Each move: x′ = x + u + w, with wheel "
                  "slip w ~ N(0, 0.5).", INK))
parts.append(text(W / 2, 62, "Policy: go half way back, with jitter: "
                  "u = −0.5 x + e, e ~ N(0, 0.1).", INK))
parts.append(text(W / 2, AXIS + 76, "Rewards: −(x² + u²) at each move, for being "
                  "far from the goal and for the effort; −x₂² at the end.", INK))
write("LineWorld.svg", W, H, "A robot on a line",
      "A robot on a number line starts around x = 2 and makes two moves toward "
      "the origin. Each move adds the action u and Gaussian noise to its "
      "position. It is penalized for its distance from the origin and for the "
      "size of its moves.", "\n  ".join(parts))


# ---------------------------------------------------------------------------
# Plots: the value functions, and the advantage of the first move.
W, H = 860, 350


def panel(left, top, width, height, xlim, ylim, xticks, yticks, xlabel, title):
    def sx(x):
        return left + width * (x - xlim[0]) / (xlim[1] - xlim[0])

    def sy(y):
        return top + height * (ylim[1] - y) / (ylim[1] - ylim[0])

    items = [f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
             f'fill="#f6f8fa" stroke="#d0d7de"/>',
             text(left + width / 2, top - 14, title, INK, size=14)]
    for tick in xticks:
        items.append(f'<line x1="{sx(tick):.1f}" y1="{top}" x2="{sx(tick):.1f}" '
                     f'y2="{top + height}" stroke="#e1e4e8"/>')
        items.append(text(sx(tick), top + height + 18, f"{tick:g}", EDGE, size=12))
    for tick in yticks:
        items.append(f'<line x1="{left}" y1="{sy(tick):.1f}" x2="{left + width}" '
                     f'y2="{sy(tick):.1f}" stroke="#e1e4e8"/>')
        items.append(text(left - 8, sy(tick) + 4, f"{tick:g}", EDGE, "end", 12))
    items.append(text(left + width / 2, top + height + 38, xlabel, INK))
    return sx, sy, items


def curve(function, sx, sy, xlim, color, dash=""):
    points = []
    for i in range(121):
        x = xlim[0] + (xlim[1] - xlim[0]) * i / 120
        points.append(f"{sx(x):.1f},{sy(function(x)):.1f}")
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"{extra}/>')


body = []

# Left: V2, V1, V0 as functions of the position.
xlim, ylim = (-3, 3), (-17, 1)
sx, sy, items = panel(60, 50, 330, 240, xlim, ylim, range(-3, 4),
                      (0, -5, -10, -15), "position x", "Value functions")
body += items
body.append(curve(lambda x: -x * x, sx, sy, xlim, MUTED, "6 4"))
body.append(curve(lambda x: -(1.5 * x * x + 0.7), sx, sy, xlim, BLUE))
body.append(curve(lambda x: -(1.625 * x * x + 1.7), sx, sy, xlim, ORANGE))
body.append(text(sx(0), sy(-9.8), "V₂(x) = −x²", EDGE, size=12))
body.append(text(sx(0), sy(-11.6), "V₁(x) = −(1.5 x² + 0.7)", BLUE, size=12))
body.append(text(sx(0), sy(-13.4), "V₀(x) = −(1.625 x² + 1.7)", "#a85a08",
                 size=12))

# Right: the advantage of the first move, starting at x0 = 2.
advantage = lambda u: -2.5 * (u + 1.2) ** 2 + 0.35
xlim, ylim = (-2.6, 0.2), (-5, 1)
sx, sy, items = panel(480, 50, 330, 240, xlim, ylim,
                      (-2.5, -2, -1.5, -1, -0.5, 0), (1, 0, -1, -2, -3, -4, -5),
                      "first move u₀", "Advantage of the first move, from x₀ = 2")
body += items
body.append(f'<line x1="{sx(xlim[0]):.1f}" y1="{sy(0):.1f}" x2="{sx(xlim[1]):.1f}" '
            f'y2="{sy(0):.1f}" stroke="{INK}" stroke-width="1"/>')
body.append(curve(advantage, sx, sy, xlim, GREEN))
for u, color, label, dx, anchor in (
        (-1.2, RED, "best move: u₀ = −1.2", -10, "end"),
        (-1.0, BLUE, "policy's mean: u₀ = −1", 10, "start")):
    body.append(f'<circle cx="{sx(u):.1f}" cy="{sy(advantage(u)):.1f}" r="5" '
                f'fill="{color}"/>')
    body.append(text(sx(u) + dx, sy(advantage(u)) - 8, label, color, anchor, 12))
body.append(text(sx(-1.2), sy(-3.4), "A₀(2, u) = 0.35 − 2.5 (u + 1.2)²", GREEN,
                 size=12))

write("LineValueFunctions.svg", W, H,
      "Value functions and advantage of the line example",
      "Left: the value functions V2, V1 and V0 are downward parabolas in the "
      "position, each lower than the next because more moves remain. Right: "
      "the advantage of the first move from x0 = 2 is a downward parabola in "
      "the move, highest at u = -1.2, while the policy's average move is -1.",
      "\n  ".join(body))
print("written")
