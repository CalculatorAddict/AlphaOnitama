"""Onitama game engine: state, move generation, transitions, and notation.

State-space search formalisation
--------------------------------
This module is a textbook state-space problem. Read it through that lens:

  - **State**: the complete tuple ``(board, hands, neutral, to_move)`` carried by
    :class:`GameState`. Nothing else is needed to know the position.
  - **Actions**: :meth:`GameState.legal_moves` — the applicable-action set for the
    side to move.
  - **Transition / successor function**: :meth:`GameState.apply` — returns the
    next state; never mutates the current one (copy-on-apply).
  - **Goal test**: :meth:`GameState.winner` / :meth:`GameState.is_terminal`.

Win-check correctness is the single most common silent-MCTS-bug source, so it is
deliberately simple and is the first thing the tests pin down.

Coordinates and board layout
----------------------------
Files ``a``-``e`` run left-to-right from Red's perspective; ranks ``1``-``5`` run
from Red's back rank (rank 1) to Blue's back rank (rank 5). Internally the board
is a 5x5 row-major grid with **row 0 at the top (rank 5, Blue)** and **row 4 at
the bottom (rank 1, Red)**:

    row 0  <-> rank 5  (Blue back rank, Blue Temple Arch = c5)
    row 4  <-> rank 1  (Red back rank,  Red Temple Arch  = c1)

This layout makes a card offset ``(dr, dc)`` add directly to a Red piece's
``(row, col)``; Blue's offsets are the 180° mirror (see :mod:`onitama.cards`).
"""

from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple, Optional

from .cards import CARDS, offsets_for

# --- Colours -----------------------------------------------------------------
RED = "RED"
BLUE = "BLUE"

# --- Board cell contents -----------------------------------------------------
EMPTY = 0
RED_STUDENT = 1
RED_MASTER = 2
BLUE_STUDENT = 3
BLUE_MASTER = 4

_RED_PIECES = (RED_STUDENT, RED_MASTER)
_BLUE_PIECES = (BLUE_STUDENT, BLUE_MASTER)
_MASTERS = {RED: RED_MASTER, BLUE: BLUE_MASTER}

# Temple Arch a Master must reach to win by Way of the Stream, as (row, col).
# Red wins on Blue's arch (c5 = row 0), Blue wins on Red's arch (c1 = row 4).
_ENEMY_ARCH = {RED: (0, 2), BLUE: (4, 2)}

BOARD_SIZE = 5
MAX_PLIES = 200  # hard cap on game / rollout length to guarantee termination


class Move(NamedTuple):
    """A single Onitama move.

    Fields:
        card: the hand card played (always swapped with the neutral card).
        src: source square in ``file+rank`` notation, or ``None`` for a forced pass.
        dst: destination square, or ``None`` for a forced pass.

    A "forced pass" (``src is dst is None``) only arises when the side to move has
    no legal piece move; the player must still swap the chosen card with neutral.
    """

    card: str
    src: Optional[str]
    dst: Optional[str]

    @property
    def is_pass(self) -> bool:
        """True if this move moves no piece (a forced card-only swap)."""
        return self.src is None


# --- Coordinate <-> notation helpers -----------------------------------------

def sq_to_rc(square: str) -> tuple[int, int]:
    """Convert ``file+rank`` notation (e.g. ``"c1"``) to ``(row, col)``."""
    file_ch, rank_ch = square[0], square[1]
    col = ord(file_ch) - ord("a")
    row = BOARD_SIZE - int(rank_ch)  # rank 1 -> row 4, rank 5 -> row 0
    return row, col


def rc_to_sq(row: int, col: int) -> str:
    """Convert ``(row, col)`` to ``file+rank`` notation (inverse of :func:`sq_to_rc`)."""
    file_ch = chr(ord("a") + col)
    rank = BOARD_SIZE - row
    return f"{file_ch}{rank}"


def _on_board(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _color_of(cell: int) -> Optional[str]:
    if cell in _RED_PIECES:
        return RED
    if cell in _BLUE_PIECES:
        return BLUE
    return None


def _swap_color(cell: int) -> int:
    """Return the same piece kind in the opposite colour (EMPTY stays EMPTY)."""
    return {
        EMPTY: EMPTY,
        RED_STUDENT: BLUE_STUDENT,
        RED_MASTER: BLUE_MASTER,
        BLUE_STUDENT: RED_STUDENT,
        BLUE_MASTER: RED_MASTER,
    }[cell]


def other(color: str) -> str:
    """Return the opposing colour."""
    return BLUE if color == RED else RED


class GameState:
    """An immutable-ish Onitama position (copy-on-apply).

    The complete state is ``(board, hands, neutral, to_move)``:

        board:   5x5 list-of-lists of cell constants (row 0 = rank 5 / Blue side).
        hands:   ``{RED: [card, card], BLUE: [card, card]}``.
        neutral: the single side card's name.
        to_move: ``RED`` or ``BLUE``.

    Instances are treated as values: :meth:`apply` returns a fresh state and the
    receiver is never mutated.
    """

    __slots__ = ("board", "hands", "neutral", "to_move")

    def __init__(
        self,
        board: list[list[int]],
        hands: dict[str, list[str]],
        neutral: str,
        to_move: str,
    ) -> None:
        self.board = board
        self.hands = hands
        self.neutral = neutral
        self.to_move = to_move

    # --- Construction --------------------------------------------------------

    @staticmethod
    def initial(cards: tuple[str, str, str, str, str]) -> "GameState":
        """Build the start position from five card names.

        Args:
            cards: ``(red1, red2, blue1, blue2, neutral)`` — two cards for each
                player's hand plus the neutral side card. All must be in
                :data:`onitama.cards.CARDS`.

        Returns:
            The standard opening: each player's five pawns on their back rank,
            Master on the centre file (c), Red to move.
        """
        red1, red2, blue1, blue2, neutral = cards
        board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        # Blue back rank = row 0 (rank 5), Red back rank = row 4 (rank 1).
        board[0] = [BLUE_STUDENT, BLUE_STUDENT, BLUE_MASTER, BLUE_STUDENT, BLUE_STUDENT]
        board[4] = [RED_STUDENT, RED_STUDENT, RED_MASTER, RED_STUDENT, RED_STUDENT]
        hands = {RED: [red1, red2], BLUE: [blue1, blue2]}
        return GameState(board, hands, neutral, RED)

    def copy(self) -> "GameState":
        """Return a deep copy safe to mutate independently of ``self``."""
        return GameState(
            [row[:] for row in self.board],
            {RED: list(self.hands[RED]), BLUE: list(self.hands[BLUE])},
            self.neutral,
            self.to_move,
        )

    # --- Applicable-action set (move generation) -----------------------------

    def legal_moves(self) -> list[Move]:
        """Return the applicable actions for the side to move.

        For each friendly piece, each hand card, and each of that card's offsets
        (mirrored for Blue), the target square is kept if it is on the board and
        not occupied by a friendly piece (enemy pieces are capturable).

        If no piece move exists, returns the forced-pass set: one ``Card:--`` move
        per hand card (the player must still swap a card with neutral). This is
        rare but real, and returning an empty list here would break search.
        """
        moves: list[Move] = []
        me = self.to_move
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if _color_of(self.board[r][c]) != me:
                    continue
                src = rc_to_sq(r, c)
                for card in self.hands[me]:
                    for dr, dc in offsets_for(card, me):
                        nr, nc = r + dr, c + dc
                        if not _on_board(nr, nc):
                            continue
                        if _color_of(self.board[nr][nc]) == me:
                            continue  # cannot land on a friendly piece
                        moves.append(Move(card, src, rc_to_sq(nr, nc)))
        if not moves:
            # No piece can move: forced card-swap pass, one option per hand card.
            return [Move(card, None, None) for card in self.hands[me]]
        return moves

    # --- Successor / transition function -------------------------------------

    def apply(self, move: Move) -> "GameState":
        """Return the state reached by playing ``move`` (does not mutate ``self``).

        Resolves any capture (landing on an enemy piece removes it), performs the
        forced played-card <-> neutral swap, and flips the side to move. Assumes
        ``move`` is legal.
        """
        nxt = self.copy()
        me = nxt.to_move
        if not move.is_pass:
            sr, sc = sq_to_rc(move.src)
            dr, dc = sq_to_rc(move.dst)
            piece = nxt.board[sr][sc]
            nxt.board[sr][sc] = EMPTY
            nxt.board[dr][dc] = piece  # overwrite captures an enemy piece
        # Forced card swap: played card leaves the hand, neutral enters it.
        hand = nxt.hands[me]
        idx = hand.index(move.card)
        hand[idx], nxt.neutral = nxt.neutral, move.card
        nxt.to_move = other(me)
        return nxt

    # --- Goal test -----------------------------------------------------------

    def winner(self) -> Optional[str]:
        """Return ``RED``/``BLUE`` if the game is won, else ``None``.

        Two immediate win conditions:
          - Way of the Stone: an enemy Master has been captured (is absent).
          - Way of the Stream: a Master stands on the enemy Temple Arch
            (Red on c5, Blue on c1).
        """
        red_master = blue_master = False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.board[r][c]
                if cell == RED_MASTER:
                    red_master = True
                    if (r, c) == _ENEMY_ARCH[RED]:
                        return RED  # Red Master reached c5
                elif cell == BLUE_MASTER:
                    blue_master = True
                    if (r, c) == _ENEMY_ARCH[BLUE]:
                        return BLUE  # Blue Master reached c1
        if not blue_master:
            return RED  # Blue Master captured
        if not red_master:
            return BLUE  # Red Master captured
        return None

    def is_terminal(self) -> bool:
        """True if the position is won by either side."""
        return self.winner() is not None

    # --- Canonicalisation (side-to-move always "Red from the bottom") --------

    def canonical(self) -> "GameState":
        """Return this state from the side-to-move's perspective.

        If Red is to move the state is already canonical and a copy is returned.
        If Blue is to move, the board is rotated 180° and colours are swapped so
        that the side to move always appears as Red moving up from the bottom,
        and the hands are swapped accordingly. The result always has
        ``to_move == RED``.

        This is **critical for Phase 2**: the network must always see "me to move
        from the bottom" so that value signs and policy indices are relative to
        the side to move rather than to a fixed colour. Building and testing it in
        Phase 1 guards against the #1 silent AlphaZero failure mode.

        Note that a 180° board rotation is exactly the geometry that turns Blue's
        mirrored offsets back into the cards' stored (Red-POV) offsets, so card
        names are carried over unchanged.
        """
        if self.to_move == RED:
            return self.copy()
        # Blue to move: rotate 180° and swap colours.
        n = BOARD_SIZE
        new_board = [[EMPTY] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                new_board[n - 1 - r][n - 1 - c] = _swap_color(self.board[r][c])
        new_hands = {RED: list(self.hands[BLUE]), BLUE: list(self.hands[RED])}
        return GameState(new_board, new_hands, self.neutral, RED)

    # --- Notation ------------------------------------------------------------

    def move_to_str(self, move: Move) -> str:
        """Format ``move`` as a ``Card:from-to`` string (``Card:--`` for a pass).

        Capture and the card swap are implicit (see the NOTATION section of the
        spec / README), so they are not encoded.
        """
        if move.is_pass:
            return f"{move.card}:--"
        return f"{move.card}:{move.src}-{move.dst}"

    def parse_move(self, text: str) -> Move:
        """Parse a ``Card:from-to`` (or ``Card:--``) string into a :class:`Move`.

        The inverse of :meth:`move_to_str`. Does not check legality.
        """
        card, _, squares = text.partition(":")
        card = card.strip()
        squares = squares.strip()
        if squares == "--":
            return Move(card, None, None)
        src, _, dst = squares.partition("-")
        return Move(card, src.strip(), dst.strip())

    # --- Equality / hashing / display ----------------------------------------

    def _key(self) -> tuple:
        return (
            tuple(tuple(row) for row in self.board),
            tuple(self.hands[RED]),
            tuple(self.hands[BLUE]),
            self.neutral,
            self.to_move,
        )

    def __eq__(self, other_state: object) -> bool:
        if not isinstance(other_state, GameState):
            return NotImplemented
        return self._key() == other_state._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def render_ascii(self) -> str:
        """Return a human-readable text board (ranks 5..1 top-to-bottom).

        Glyphs: ``r``/``R`` = Red Student/Master, ``b``/``B`` = Blue
        Student/Master, ``.`` = empty. Used by the headless / ``--ascii`` driver.
        """
        glyph = {
            EMPTY: ".",
            RED_STUDENT: "r",
            RED_MASTER: "R",
            BLUE_STUDENT: "b",
            BLUE_MASTER: "B",
        }
        lines = []
        for r in range(BOARD_SIZE):
            rank = BOARD_SIZE - r
            cells = " ".join(glyph[self.board[r][c]] for c in range(BOARD_SIZE))
            lines.append(f"{rank} | {cells}")
        lines.append("  +" + "-" * (2 * BOARD_SIZE))
        lines.append("    " + " ".join("abcde"))
        lines.append(f"  neutral: {self.neutral}")
        lines.append(f"  {RED} hand: {self.hands[RED]}   {BLUE} hand: {self.hands[BLUE]}")
        lines.append(f"  to move: {self.to_move}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GameState(to_move={self.to_move}, neutral={self.neutral!r})"
