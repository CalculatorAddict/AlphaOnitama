"""Monte Carlo Tree Search — the conceptual heart of the engine.

MCTS builds a search tree by repeating four stages per iteration. Read it as a
bandit-driven, anytime planning algorithm:

  1. **Select** (selection) — from the root, repeatedly descend to the child that
     maximises a *tree policy* (a multi-armed-bandit rule balancing exploitation
     of high-value children against exploration of rarely-visited ones), until we
     reach a node that is not yet fully expanded or is terminal.
  2. **Expand** (expansion) — add one new child for an untried move.
  3. **Simulate** (simulation / evaluation) — estimate the value of the new leaf.
     Phase 1 does a uniform-random *rollout* to a terminal state (or a ply cap).
     Phase 2 replaces this with a neural value head — hence the pluggable
     ``evaluate_leaf`` hook.
  4. **Backpropagate** (backpropagation) — propagate the leaf value up to the
     root, **flipping its sign at every level** because the players alternate in a
     zero-sum game.

The whole search acts as a *policy-improvement operator*: the visit counts at the
root form a better policy than the raw tree policy, which is exactly what Phase 2
distills into the network. The chosen move is the *robust* child — the most
visited, not the highest mean — because visit count is the more stable statistic.

Value / sign convention
-----------------------
``evaluate_leaf`` returns a value in ``[-1, 1]`` **from the perspective of the
side to move at the leaf**: +1 if that side ultimately wins, -1 if it loses, 0 on
a ply-cap draw. A node's accumulated ``W`` is stored from the perspective of the
player who *moved into* that node (equivalently: the side to move at its parent).
That alignment is what makes the UCB1 exploitation term ``W/N`` directly the
value of the move *for the parent's side to move*, so the parent simply maximises
it. Getting this sign flip wrong is the second classic silent MCTS bug (after a
broken win-check); the backprop loop documents exactly where it happens.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

from .game import MAX_PLIES, GameState, Move

# Default UCB1 exploration constant. sqrt(2) is the textbook value for rewards in
# [-1, 1] / [0, 1]; larger explores more, smaller exploits more.
DEFAULT_C = math.sqrt(2)


class Node:
    """A node in the search tree, representing one game position.

    Attributes:
        state: the :class:`GameState` at this node.
        parent: the parent node, or ``None`` at the root.
        move: the move that led from ``parent`` to here (``None`` at the root).
        children: ``{Move: Node}`` for moves already expanded.
        N: visit count.
        W: total accumulated value, from the perspective of the player who *moved
            into* this node (see the module docstring).
        P: prior probability of the move into this node — Phase 2 only; unused
            (left 0) by the Phase 1 UCB1 policy.
        untried_moves: legal moves from ``state`` not yet expanded into children.
    """

    __slots__ = ("state", "parent", "move", "children", "N", "W", "P", "untried_moves")

    def __init__(self, state: GameState, parent: Optional["Node"] = None,
                 move: Optional[Move] = None,
                 rng: Optional[random.Random] = None) -> None:
        self.state = state
        self.parent = parent
        self.move = move
        self.children: dict[Move, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.P = 0.0
        # A terminal node has no actions; otherwise shuffle so expansion order
        # (and thus early play at low iteration counts) is unbiased.
        self.untried_moves = [] if state.is_terminal() else list(state.legal_moves())
        if rng is not None:
            rng.shuffle(self.untried_moves)

    def is_fully_expanded(self) -> bool:
        """True once every legal move from this node has a child."""
        return not self.untried_moves

    def is_terminal(self) -> bool:
        """True if the position at this node is won by either side."""
        return self.state.is_terminal()


# --- Tree policy (pluggable; Phase 2 swaps UCB1 for PUCT) ---------------------

def ucb1_select(node: Node, c: float) -> Node:
    """Select the child of ``node`` maximising the UCB1 score.

    ::

        UCB1(child) = W/N                         # exploitation: mean value so far
                    + c * sqrt(ln(parent.N)/N)    # exploration: bonus for rarely tried

    An unvisited child (N == 0) scores +inf and is returned immediately — every
    move is tried at least once before any is deepened. ``c`` trades the two
    terms off; the default is sqrt(2).

    Phase 2 replaces this with the PUCT rule
    ``Q + c_puct * P * sqrt(parent.N) / (1 + N)``; keeping selection behind this
    one function is what lets that swap happen without touching the search loop.
    """
    log_parent_n = math.log(node.N)
    best_child: Optional[Node] = None
    best_score = -math.inf
    for child in node.children.values():
        if child.N == 0:
            return child  # must-visit
        exploitation = child.W / child.N
        exploration = c * math.sqrt(log_parent_n / child.N)
        score = exploitation + exploration
        if score > best_score:
            best_score, best_child = score, child
    assert best_child is not None  # node was fully expanded => has children
    return best_child


# --- Leaf evaluation (pluggable; Phase 2 swaps rollout for the net) -----------

def rollout(node: Node, rng: random.Random, max_plies: int = MAX_PLIES) -> float:
    """Stage 3 — Simulate (Phase-1 leaf evaluation): a uniform-random playout.

    Returns the outcome **from the perspective of the side to move at ``node``**:
    +1 if that side wins, -1 if it loses, 0 if the playout hits ``max_plies``
    without a winner (no signal). At low iteration counts most playouts hit the
    cap and the signal is near-zero, so play is near-random; signal — and strength
    — emerge as playouts start reaching real terminals.
    """
    leaf = node.state
    if leaf.is_terminal():
        # The side to move at a terminal node is the one who was just beaten.
        return -1.0
    me = leaf.to_move
    sim = leaf
    for _ in range(max_plies):
        if sim.is_terminal():
            break
        sim = sim.apply(rng.choice(sim.legal_moves()))
    winner = sim.winner()
    if winner is None:
        return 0.0  # ply cap reached: a draw / no signal
    return 1.0 if winner == me else -1.0


# --- The four-phase loop -----------------------------------------------------

def _select(root: Node, tree_policy: Callable[[Node, float], Node], c: float) -> Node:
    """Stage 1 — Select: descend by the tree policy to an expandable/terminal node."""
    node = root
    while node.is_fully_expanded() and not node.is_terminal():
        node = tree_policy(node, c)
    return node


def _expand(node: Node, rng: random.Random) -> Node:
    """Stage 2 — Expand: realise one untried move as a fresh child, and return it.

    A terminal (or already fully expanded) node is returned unchanged so it can be
    evaluated directly.
    """
    if node.is_terminal() or not node.untried_moves:
        return node
    move = node.untried_moves.pop()
    child = Node(node.state.apply(move), parent=node, move=move, rng=rng)
    node.children[move] = child
    return child


def _backprop(node: Optional[Node], value: float) -> None:
    """Stage 4 — Backpropagate: carry ``value`` to the root, flipping sign per level.

    ``value`` enters from the perspective of the side to move at the leaf. The
    sign is flipped *before* adding at each step so that each node's ``W`` ends up
    in the perspective of the player who moved into it — the zero-sum,
    alternating-players adjustment that keeps ``W/N`` meaningful for selection.
    """
    while node is not None:
        node.N += 1
        value = -value  # alternate perspective: this node's mover is the opponent
        node.W += value
        node = node.parent


def mcts_search(
    root_state: GameState,
    iterations: int = 1000,
    *,
    evaluate_leaf: Optional[Callable[[Node], float]] = None,
    tree_policy: Callable[[Node, float], Node] = ucb1_select,
    c: float = DEFAULT_C,
    rng: Optional[random.Random] = None,
) -> tuple[Move, dict[Move, int]]:
    """Run MCTS from ``root_state`` and return ``(best_move, visit_counts)``.

    Args:
        root_state: the position to choose a move for.
        iterations: number of select/expand/evaluate/backprop cycles.
        evaluate_leaf: leaf value function ``node -> value in [-1, 1]`` (side-to-
            move perspective). Defaults to a random :func:`rollout`. Phase 2 passes
            the network's value head here.
        tree_policy: selection rule ``(node, c) -> child``. Defaults to
            :func:`ucb1_select`; Phase 2 passes a PUCT variant.
        c: exploration constant handed to ``tree_policy``.
        rng: RNG for rollouts and expansion order (reproducibility).

    Returns:
        ``best_move``: the *robust* (most-visited) root child — the move to play.
        ``visit_counts``: ``{Move: N}`` over the root's children. Phase 1 uses only
        the argmax; Phase 2 uses the full distribution as the policy target.
    """
    rng = rng or random.Random()
    if evaluate_leaf is None:
        def evaluate_leaf(node: Node) -> float:  # default: random rollout
            return rollout(node, rng)

    root = Node(root_state, rng=rng)
    for _ in range(iterations):
        leaf = _select(root, tree_policy, c)   # Stage 1 — Select
        leaf = _expand(leaf, rng)              # Stage 2 — Expand
        value = evaluate_leaf(leaf)            # Stage 3 — Simulate (rollout / net)
        _backprop(leaf, value)                 # Stage 4 — Backpropagate

    if not root.children:
        # Degenerate (e.g. a terminal root); fall back to any legal move.
        return root_state.legal_moves()[0], {}
    visit_counts = {move: child.N for move, child in root.children.items()}
    best_move = max(visit_counts, key=visit_counts.__getitem__)
    return best_move, visit_counts
