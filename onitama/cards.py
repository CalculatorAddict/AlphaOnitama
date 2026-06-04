"""Onitama movement cards.

Movement in Onitama comes from cards, not piece type. Each card is a set of
``(dr, dc)`` offsets expressed from **Red's** point of view, where:

  - ``(-1, 0)`` is one square "forward" (toward Blue), i.e. toward higher rank.
  - ``(+1, 0)`` is one square "backward" (toward Red's own back rank).
  - ``(0, -1)`` is one square to Red's left (toward file ``a``).
  - ``(0, +1)`` is one square to Red's right (toward file ``e``).

The board is stored row-major with row 0 at the top (rank 5, Blue's back rank)
and row 4 at the bottom (rank 1, Red's back rank). With that convention an
offset ``(dr, dc)`` adds directly to a Red piece's ``(row, col)``; Blue is the
engine mirror, obtained by negating both components (a 180° rotation of the
board). See :func:`offsets_for`.

This module is intentionally tiny and dependency-free: it is the single source
of truth for card geometry, read directly by the game engine and the GUI.
"""

from __future__ import annotations

# The 16-card Onitama base deck. Offsets are (dr, dc) from Red's perspective.
# These are fixed game data; do not edit without checking against the rulebook.
CARDS: dict[str, list[tuple[int, int]]] = {
    "Tiger":    [(-2, 0), (1, 0)],
    "Crab":     [(-1, 0), (0, -2), (0, 2)],
    "Monkey":   [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "Crane":    [(-1, 0), (1, -1), (1, 1)],
    "Dragon":   [(-1, -2), (-1, 2), (1, -1), (1, 1)],
    "Elephant": [(0, -1), (0, 1), (-1, -1), (-1, 1)],
    "Boar":     [(-1, 0), (0, -1), (0, 1)],
    "Mantis":   [(-1, -1), (-1, 1), (1, 0)],
    "Rooster":  [(0, -1), (-1, 1), (1, -1), (0, 1)],
    "Ox":       [(-1, 0), (0, 1), (1, 0)],
    "Horse":    [(-1, 0), (0, -1), (1, 0)],
    "Frog":     [(-1, -1), (0, -2), (1, 1)],
    "Rabbit":   [(-1, 1), (0, 2), (1, -1)],
    "Goose":    [(0, -1), (-1, -1), (1, 1), (0, 1)],
    "Cobra":    [(0, -1), (-1, 1), (1, 1)],
    "Eel":      [(-1, -1), (0, 1), (1, -1)],
}

# Sorted card names, handy for deterministic dealing/iteration in tests.
CARD_NAMES: list[str] = sorted(CARDS)


def mirror(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return ``offsets`` rotated 180° (negate dr and dc) for Blue's POV."""
    return [(-dr, -dc) for dr, dc in offsets]


def offsets_for(card_name: str, color: str) -> list[tuple[int, int]]:
    """Return the ``(dr, dc)`` offsets a piece of ``color`` may use with a card.

    Args:
        card_name: a key of :data:`CARDS`.
        color: ``"RED"`` or ``"BLUE"``. Red uses the card's stored offsets;
            Blue uses the engine mirror (both components negated).

    Returns:
        A list of ``(dr, dc)`` offsets directly addable to a piece's
        ``(row, col)`` on the engine's row-0-at-top board.
    """
    base = CARDS[card_name]
    return base if color == "RED" else mirror(base)
