"""Agents: objects that choose a move given a position.

Every agent exposes the same one-method contract::

    choose(state: GameState) -> Move

so the drivers (``play.py``, ``gui.py``) can pit any pairing against any other.

Phase 1 ships :class:`RandomAgent`, :class:`MCTSAgent` (UCB1 + random rollouts),
and :class:`HumanAgent` (CLI numbered entry). Phase 2 adds ``NeuralMCTSAgent``.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .game import GameState, Move


class RandomAgent:
    """Picks uniformly at random among the legal moves. The baseline opponent."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random

    def choose(self, state: GameState) -> Move:
        """Return a uniformly random legal move."""
        return self._rng.choice(state.legal_moves())


class MCTSAgent:
    """Plays the move chosen by pure MCTS (UCB1 tree policy, random rollouts).

    Args:
        iterations: number of select/expand/simulate/backprop iterations per move.
        c: UCB1 exploration constant (defaults to sqrt(2)).
        rng: optional RNG for reproducible rollouts.
    """

    def __init__(self, iterations: int = 1000, c: float = math.sqrt(2),
                 rng: Optional[random.Random] = None) -> None:
        self.iterations = iterations
        self.c = c
        self._rng = rng or random

    def choose(self, state: GameState) -> Move:
        """Run MCTS from ``state`` and return the most-visited root move."""
        # Imported lazily so the engine + RandomAgent have no dependency on search.
        from .mcts import mcts_search

        move, _visits = mcts_search(
            state, iterations=self.iterations, c=self.c, rng=self._rng
        )
        return move


class HumanAgent:
    """Reads a move from the terminal as a number into the legal-move list.

    Used by the ASCII driver; in the GUI, human input is handled by click-to-move
    in ``gui.py`` rather than through this class.
    """

    def choose(self, state: GameState) -> Move:
        """Print the numbered legal moves and return the one the user selects."""
        moves = state.legal_moves()
        print("Legal moves:")
        for i, m in enumerate(moves):
            print(f"  [{i}] {state.move_to_str(m)}")
        while True:
            raw = input(f"{state.to_move} move # > ").strip()
            try:
                idx = int(raw)
                if 0 <= idx < len(moves):
                    return moves[idx]
            except ValueError:
                pass
            print(f"  enter a number 0..{len(moves) - 1}")
