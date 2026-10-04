"""Generate the three figures of one Bellman backup for the semiring README."""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, os.pardir, "figures")
W, H = 780, 316
S, SP, A = (110, 100), (410, 100), (260, 220)  # state, next state, action
INK, EDGE, MUTED = "#24292f", "#57606a", "#8c959f"
VALUE, VALUE_TEXT = "#d9730d", "#a85a08"


SUBSCRIPTS = str.maketrans("t+1", "\u209c\u208a\u2081")


def sub(base, script, rest=""):
    """Text with a subscript, e.g. V_{t+1}(s'), using Unicode subscripts."""
    return base + script.translate(SUBSCRIPTS) + rest


def line(p, q):
    return f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}"/>'


def square(center, color, size=16):
    x, y = center[0] - size / 2, center[1] - size / 2
    return f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{color}"/>'


def variable(center, label, kind):
    fill, stroke, text = {
        "state": ("#ddebff", "#1a4fa0", INK),
        "action": ("#dff3e4", "#1e7b34", INK),
        "done": ("#f3f4f6", MUTED, MUTED),
    }[kind]
    dash = ' stroke-dasharray="4 3"' if kind == "done" else ""
    return (f'<circle cx="{center[0]}" cy="{center[1]}" r="24" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"{dash}/>'
            f'<text x="{center[0]}" y="{center[1] + 6}" font-size="18" '
            f'font-style="italic" text-anchor="middle" fill="{text}">{label}</text>')


def ring(center):
    """Mark the variable being eliminated."""
    return (f'<circle cx="{center[0]}" cy="{center[1]}" r="33" fill="none" '
            f'stroke="#cf222e" stroke-width="2" stroke-dasharray="6 4"/>')


def text(x, y, content, color=INK, anchor="middle", size=13):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}">{content}</text>')


def arrow(d, tip, direction):
    """A path ending in an explicit arrowhead at tip, pointing along direction."""
    norm = math.hypot(*direction)
    ux, uy = direction[0] / norm, direction[1] / norm
    back = (tip[0] - 11 * ux, tip[1] - 11 * uy)
    left = (back[0] - 4.5 * uy, back[1] + 4.5 * ux)
    right = (back[0] + 4.5 * uy, back[1] - 4.5 * ux)
    head = " ".join(f"{x:.1f},{y:.1f}" for x, y in (tip, left, right))
    return (f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="2"/>'
            f'<polygon points="{head}" fill="{MUTED}"/>')


LEGEND = f'''
  <g font-size="12" fill="{EDGE}">
    <rect x="24" y="281" width="12" height="12" fill="{INK}"/>
    <text x="42" y="291">(p, 0): probability</text>
    <rect x="166" y="281" width="12" height="12" fill="{VALUE}"/>
    <text x="184" y="291">(1, v): value</text>
    <circle cx="282" cy="287" r="8" fill="none" stroke="#cf222e" stroke-width="2" stroke-dasharray="4 3"/>
    <text x="297" y="291">being eliminated</text>
    <circle cx="418" cy="287" r="8" fill="#f3f4f6" stroke="{MUTED}" stroke-width="2" stroke-dasharray="3 2"/>
    <text x="433" y="291">eliminated; arrows: its conditional</text>
  </g>'''


def figure(name, title, desc, edges, body):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="title desc" font-family="Helvetica, Arial, sans-serif">
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="#ffffff" stroke="#d0d7de"/>
  <g stroke="{EDGE}" stroke-width="2" fill="none">
    {' '.join(edges)}
  </g>
  {body}
  <line x1="16" y1="266" x2="{W - 16}" y2="266" stroke="#d0d7de"/>{LEGEND}
</svg>
'''
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)


DYNAMICS, POLICY, REWARD, FUTURE = (260, 100), (185, 160), (110, 220), (520, 100)
V_NEXT = sub("V", "t+1", "(s′)")
EXPECTED = "E[" + sub("V", "t+1", "(s′) | s, a]")

policy_and_reward_edges = [line(S, A), line(S, REWARD), line(REWARD, A)]
policy_and_reward = "\n  ".join([
    square(POLICY, INK), square(REWARD, VALUE),
    text(200, 194, "(π(a | s), 0)", anchor="end"),
    text(110, 250, "(1, r(s, a))", VALUE_TEXT),
])
conditional_of_next_state = "\n  ".join([
    arrow("M 128 84 Q 260 14 384 82", (391, 86), (131, 72)),
    arrow("M 279 205 L 386 120", (392, 115), (113, -90)),
    variable(SP, "s′", "done"),
    text(448, 105, f"c(s′ | s, a) = (p(s′ | s, a), {sub('δ', 't', '(s, a, s′)')})",
         EDGE, anchor="start"),
])

# 1. Before: the bucket of the next state.
figure(
    "BellmanBackupStart.svg",
    "One time step before elimination",
    "State s, action a and next state s prime. The dynamics factor joins all "
    "three, the policy and the reward join s and a, and a value factor holding "
    "the future value attaches to s prime, which is about to be eliminated.",
    policy_and_reward_edges + [line(S, FUTURE), line(DYNAMICS, A)],
    "\n  ".join([
        square(DYNAMICS, INK), square(FUTURE, VALUE), policy_and_reward,
        ring(SP),
        variable(S, "s", "state"), variable(SP, "s′", "state"),
        variable(A, "a", "action"),
        text(260, 78, "(p(s′ | s, a), 0)"),
        text(520, 78, f"(1, {V_NEXT})", VALUE_TEXT),
    ]))

# 2. After eliminating the next state: a new value factor on (s, a).
figure(
    "BellmanBackupNextState.svg",
    "After eliminating the next state",
    "The next state is eliminated and keeps a conditional given s and a. The "
    "dynamics and future value factors are replaced by one new value factor "
    "joining s and a, holding the expected future value. The action is about "
    "to be eliminated.",
    policy_and_reward_edges + [line(S, DYNAMICS), line(DYNAMICS, A)],
    "\n  ".join([
        square(DYNAMICS, VALUE), policy_and_reward,
        conditional_of_next_state,
        ring(A),
        variable(S, "s", "state"), variable(A, "a", "action"),
        text(260, 78, f"φ(s, a) = (1, {EXPECTED})", VALUE_TEXT),
    ]))

# 3. After eliminating the action: a new value factor on s.
figure(
    "BellmanBackupAction.svg",
    "After eliminating the action",
    "The action is eliminated and keeps a conditional given s, the policy "
    "with the advantage. The policy, reward and expected future value factors "
    "are replaced by one new value factor on s, holding the state value.",
    [line(S, REWARD)],
    "\n  ".join([
        square(REWARD, VALUE),
        arrow("M 128 116 L 236 200", (242, 205), (114, 89)),
        conditional_of_next_state,
        variable(A, "a", "done"),
        variable(S, "s", "state"),
        text(300, 225, f"c(a | s) = (π(a | s), {sub('A', 't', '(s, a)')})", EDGE,
             anchor="start"),
        text(110, 250, f"φ(s) = (1, {sub('V', 't', '(s)')})", VALUE_TEXT),
    ]))
print("written")
