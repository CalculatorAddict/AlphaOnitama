"""CLI driver: deal a game, run agent pairings, headless batch eval, or launch the GUI.

Modes
-----
  human-vs-mcts      GUI: you (Red) vs MCTS (Blue).
  mcts-vs-random     headless batch; prints win rate (the debugging oracle).
  random-vs-random   headless batch; sanity baseline (~50/50).
  mcts-vs-mcts       watch two MCTS agents (GUI, or headless with --ascii).
  human-vs-neural    Phase 2 (requires a trained checkpoint).

Examples::

    uv run onitama --mode human-vs-mcts --iterations 2000
    uv run onitama --mode mcts-vs-random --games 50
    uv run onitama --mode random-vs-random --games 200    # expect ~50/50

The headless batch is the Phase 1 correctness oracle: two RandomAgents should be
~50/50, and MCTS at >=1000 iterations should crush random (~95%+). A skewed
random-vs-random result, or MCTS failing to dominate, points at a win-check or
backprop-sign bug.
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

from .cards import CARD_NAMES
from .game import BLUE, RED, GameState, MAX_PLIES, Move


def deal(rng: Optional[random.Random] = None) -> GameState:
    """Deal a fresh start position: five distinct cards, two per hand plus neutral."""
    rng = rng or random
    five = rng.sample(CARD_NAMES, 5)
    return GameState.initial(tuple(five))  # type: ignore[arg-type]


def play_game(agent_red, agent_blue, *, max_plies: int = MAX_PLIES,
              verbose: bool = False, rng: Optional[random.Random] = None,
              start: Optional[GameState] = None) -> Optional[str]:
    """Play one game to terminal (or the ply cap) and return the winner colour.

    Args:
        agent_red, agent_blue: objects with ``choose(state) -> Move``.
        max_plies: safety cap; on reaching it the game is a draw (``None``).
        verbose: if True, print an ASCII board and a notated move log per ply.
        start: optional pre-dealt start state (else a random deal).

    Returns:
        ``RED``, ``BLUE``, or ``None`` for a draw at the ply cap.
    """
    state = start or deal(rng)
    agents = {RED: agent_red, BLUE: agent_blue}
    for ply in range(max_plies):
        if state.is_terminal():
            break
        mover = state.to_move
        move = agents[mover].choose(state)
        if verbose:
            tag = "(captures)" if _is_capture(state, move) else ""
            print(f"{ply // 2 + 1}. {mover[0]} {state.move_to_str(move)} {tag}".rstrip())
        state = state.apply(move)
    winner = state.winner()
    if verbose:
        print(state.render_ascii())
        print(f"Result: {winner or 'draw (ply cap)'}\n")
    return winner


def _is_capture(state: GameState, move: Move) -> bool:
    """True if ``move`` lands on an occupied (enemy) square in ``state``."""
    if move.is_pass:
        return False
    from .game import sq_to_rc, EMPTY
    r, c = sq_to_rc(move.dst)
    return state.board[r][c] != EMPTY


def run_batch(make_red, make_blue, games: int, label: str,
              rng: Optional[random.Random] = None) -> None:
    """Play ``games`` games (alternating start state) and print a win-rate summary.

    ``make_red`` / ``make_blue`` are zero-arg factories so each game gets fresh
    agents. The same dealt position is used by both colourings is *not* required;
    we simply deal independently per game.
    """
    rng = rng or random.Random()
    tally = {RED: 0, BLUE: 0, None: 0}
    for _ in range(games):
        winner = play_game(make_red(), make_blue(), rng=rng)
        tally[winner] += 1
    n = float(games)
    print(f"{label}: {games} games")
    print(f"  RED  {tally[RED]:4d}  ({tally[RED] / n:5.1%})")
    print(f"  BLUE {tally[BLUE]:4d}  ({tally[BLUE] / n:5.1%})")
    print(f"  draw {tally[None]:4d}  ({tally[None] / n:5.1%})")


# --- Mode wiring -------------------------------------------------------------

def _make_agent_factories(mode: str, iterations: int, checkpoint: Optional[str]):
    """Return ``(make_red, make_blue)`` factories for a headless ``mode``."""
    from .agents import MCTSAgent, RandomAgent

    if mode == "random-vs-random":
        return (RandomAgent, RandomAgent)
    if mode == "mcts-vs-random":
        return (lambda: MCTSAgent(iterations), RandomAgent)
    if mode == "mcts-vs-mcts":
        return (lambda: MCTSAgent(iterations), lambda: MCTSAgent(iterations))
    raise ValueError(f"mode {mode!r} is not a headless batch mode")


def _run_gui(mode: str, iterations: int, checkpoint: Optional[str]) -> None:
    """Launch the pygame GUI for an interactive / watch mode."""
    from . import gui  # lazy: pygame only needed for GUI modes
    gui.run(mode=mode, iterations=iterations, checkpoint=checkpoint)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the ``onitama`` console script."""
    parser = argparse.ArgumentParser(prog="onitama", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="human-vs-mcts",
                        choices=["human-vs-mcts", "human-vs-neural", "mcts-vs-random",
                                 "random-vs-random", "mcts-vs-mcts"])
    parser.add_argument("--iterations", type=int, default=1000,
                        help="MCTS iterations per move (default 1000)")
    parser.add_argument("--games", type=int, default=50,
                        help="number of games for headless batch modes")
    parser.add_argument("--checkpoint", default=None,
                        help="Phase 2 net checkpoint for human-vs-neural")
    parser.add_argument("--ascii", action="store_true",
                        help="force text rendering / headless play")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    headless_modes = {"random-vs-random", "mcts-vs-random", "mcts-vs-mcts"}

    # Headless batch evaluation (the oracle), or any mode forced to --ascii.
    if args.mode in headless_modes and (args.ascii or args.mode != "mcts-vs-mcts"):
        make_red, make_blue = _make_agent_factories(args.mode, args.iterations,
                                                    args.checkpoint)
        run_batch(make_red, make_blue, args.games, args.mode)
        return 0

    if args.ascii:
        return _run_ascii(args)

    _run_gui(args.mode, args.iterations, args.checkpoint)
    return 0


def _run_ascii(args) -> int:
    """Play a single interactive/watch game in the terminal."""
    from .agents import HumanAgent, MCTSAgent, RandomAgent

    def pick(role: str):
        if role == "human":
            return HumanAgent()
        if role == "mcts":
            return MCTSAgent(args.iterations)
        return RandomAgent()

    roles = {
        "human-vs-mcts": ("human", "mcts"),
        "mcts-vs-mcts": ("mcts", "mcts"),
        "mcts-vs-random": ("mcts", "random"),
        "random-vs-random": ("random", "random"),
    }.get(args.mode, ("human", "mcts"))
    play_game(pick(roles[0]), pick(roles[1]), verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
