"""Generate the figures of one optimal backup: expectation over x', max over u."""
import fig_chapter01_backup as m

X, XN, U = m.S, m.SP, m.A          # positions of x, x' and u
DYNAMICS, REWARD, FUTURE = (260, 100), (110, 220), (520, 100)
RED = "#cf222e"
V_NEXT = m.sub("V", "t+1", "(x′)")
EXPECTED = "E[" + m.sub("V", "t+1", "(x′) | x, u]")

reward_edges = [m.line(X, REWARD), m.line(REWARD, U)]
reward = "\n  ".join([m.square(REWARD, m.VALUE),
                      m.text(110, 250, "(1, r(x, u))", m.VALUE_TEXT)])
conditional_of_next_state = "\n  ".join([
    m.arrow("M 128 84 Q 260 14 384 82", (391, 86), (131, 72)),
    m.arrow("M 279 205 L 386 120", (392, 115), (113, -90)),
    m.variable(XN, "x′", "done"),
    m.text(448, 105, "c(x′ | x, u): the dynamics, with its surprise", m.EDGE,
           anchor="start"),
])

m.figure(
    "OptimalBackupStart.svg",
    "One time step before elimination, without a policy factor",
    "State x, action u and next state x prime. The dynamics factor joins all "
    "three, the reward joins x and u, and a value factor holding the future "
    "value attaches to x prime. There is no policy factor. x prime is "
    "eliminated first, by expectation.",
    reward_edges + [m.line(X, FUTURE), m.line(DYNAMICS, U)],
    "\n  ".join([
        m.square(DYNAMICS, m.INK), m.square(FUTURE, m.VALUE), reward,
        m.ring(XN),
        m.variable(X, "x", "state"), m.variable(XN, "x′", "state"),
        m.variable(U, "u", "action"),
        m.text(260, 78, "(p(x′ | x, u), 0)"),
        m.text(520, 78, f"(1, {V_NEXT})", m.VALUE_TEXT),
        m.text(410, 152, "eliminate by expectation", RED),
    ]))

m.figure(
    "OptimalBackupNextState.svg",
    "After eliminating the next state by expectation",
    "The next state keeps the dynamics as its conditional. A new value factor "
    "joins x and u, holding the expected future value. The action is "
    "eliminated next, by maximization.",
    reward_edges + [m.line(X, DYNAMICS), m.line(DYNAMICS, U)],
    "\n  ".join([
        m.square(DYNAMICS, m.VALUE), reward,
        conditional_of_next_state,
        m.ring(U),
        m.variable(X, "x", "state"), m.variable(U, "u", "action"),
        m.text(260, 78, f"φ(x, u) = (1, {EXPECTED})", m.VALUE_TEXT),
        m.text(304, 226, "eliminate by max", RED, anchor="start"),
    ]))

m.figure(
    "OptimalBackupAction.svg",
    "After eliminating the action by maximization",
    "The action keeps the best action for each state as its conditional, the "
    "optimal policy. A new value factor on x holds the optimal value.",
    [m.line(X, REWARD)],
    "\n  ".join([
        m.square(REWARD, m.VALUE),
        m.arrow("M 128 116 L 236 200", (242, 205), (114, 89)),
        conditional_of_next_state,
        m.variable(U, "u", "done"),
        m.variable(X, "x", "state"),
        m.text(300, 225, "c(u | x): the best action for each x, the optimal "
               "policy", m.EDGE, anchor="start"),
        m.text(110, 250, f"φ(x) = (1, {m.sub('V', 't', '(x)')})", m.VALUE_TEXT),
    ]))
print("written")
