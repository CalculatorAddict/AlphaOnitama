"""Unit tests for the Onitama engine (P1.8).

Covers the correctness-critical surface before any search is trusted: win
detection (both win conditions, both colours), start-position move counts, the
forced card swap, notation round-tripping, and canonical-state symmetry.

Runs under pytest *or* standalone (``python -m onitama.test_game``) so the suite
works without the dev extra installed.
"""

from __future__ import annotations

from .cards import CARDS
from .game import (
    BLUE,
    BLUE_MASTER,
    BLUE_STUDENT,
    EMPTY,
    RED,
    RED_MASTER,
    RED_STUDENT,
    GameState,
    Move,
    rc_to_sq,
    sq_to_rc,
)


# --- Helpers -----------------------------------------------------------------

def _empty_board() -> list[list[int]]:
    return [[EMPTY] * 5 for _ in range(5)]


def _state(board, to_move, red_hand=("Tiger", "Boar"),
           blue_hand=("Crab", "Crane"), neutral="Monkey") -> GameState:
    return GameState(board, {RED: list(red_hand), BLUE: list(blue_hand)},
                     neutral, to_move)


def _start(red_hand=("Tiger", "Crab"), blue_hand=("Crane", "Boar"),
           neutral="Monkey") -> GameState:
    r1, r2 = red_hand
    b1, b2 = blue_hand
    return GameState.initial((r1, r2, b1, b2, neutral))


# --- Coordinate sanity -------------------------------------------------------

def test_coordinate_roundtrip():
    for sq in ("a1", "c1", "e1", "a5", "c5", "e5", "c3", "b4"):
        assert rc_to_sq(*sq_to_rc(sq)) == sq
    # The arches are where the spec says they are.
    assert sq_to_rc("c1") == (4, 2)  # Red Temple Arch (bottom-centre)
    assert sq_to_rc("c5") == (0, 2)  # Blue Temple Arch (top-centre)


# --- Win conditions ----------------------------------------------------------

def test_capture_master_red_wins():
    """Way of the Stone: Red student captures the Blue Master (not on an arch)."""
    board = _empty_board()
    board[4][2] = RED_MASTER     # safe, present so Red isn't already lost
    board[2][1] = RED_STUDENT    # b3
    board[1][1] = BLUE_MASTER    # b4, not an arch square
    board[0][0] = BLUE_STUDENT
    s = _state(board, RED, red_hand=("Boar", "Tiger"))
    assert s.winner() is None
    move = Move("Boar", "b3", "b4")   # (-1,0): b3 -> b4 captures the Blue Master
    assert move in s.legal_moves()
    s2 = s.apply(move)
    assert s2.winner() == RED


def test_capture_master_blue_wins():
    """Mirror: Blue captures the Red Master, exercising the Blue offset mirror."""
    board = _empty_board()
    board[0][2] = BLUE_MASTER
    board[2][3] = BLUE_STUDENT    # d3
    board[3][3] = RED_MASTER      # d2, will be captured
    board[4][0] = RED_STUDENT
    s = _state(board, BLUE, blue_hand=("Boar", "Tiger"))
    assert s.winner() is None
    # Boar for Blue mirrors (-1,0) -> (1,0): d3 (row2) -> d2 (row3) captures.
    move = Move("Boar", "d3", "d2")
    assert move in s.legal_moves()
    assert s.apply(move).winner() == BLUE


def test_stream_red_wins():
    """Way of the Stream: Red Master reaches c5 with the Blue Master still on board."""
    board = _empty_board()
    board[1][2] = RED_MASTER      # c4
    board[4][4] = BLUE_MASTER     # safely tucked away, still alive
    s = _state(board, RED, red_hand=("Boar", "Tiger"))
    assert s.winner() is None
    move = Move("Boar", "c4", "c5")   # (-1,0) onto the enemy arch
    assert move in s.legal_moves()
    s2 = s.apply(move)
    assert s2.winner() == RED
    # It's a Stream win, not a Stone win: the Blue Master is still present.
    assert any(BLUE_MASTER in row for row in s2.board)


def test_stream_blue_wins():
    board = _empty_board()
    board[3][2] = BLUE_MASTER     # c2
    board[0][0] = RED_MASTER
    s = _state(board, BLUE, blue_hand=("Boar", "Tiger"))
    assert s.winner() is None
    move = Move("Boar", "c2", "c1")   # Blue Boar mirror (1,0) onto c1
    assert move in s.legal_moves()
    assert s.apply(move).winner() == BLUE


# --- Move generation count ---------------------------------------------------

def test_start_move_count_tiger_crab():
    """Hand-counted: Tiger gives 5 moves, Crab gives 5, from the start = 10."""
    s = _start(red_hand=("Tiger", "Crab"))
    moves = s.legal_moves()
    assert len(moves) == 10
    # All distinct, all Red's, none a forced pass.
    assert len(set(moves)) == 10
    assert all(not m.is_pass for m in moves)


# --- Card swap ---------------------------------------------------------------

def test_card_swap():
    """The played card goes to neutral; the old neutral enters the hand."""
    s = _start(red_hand=("Tiger", "Crab"), neutral="Monkey")
    move = s.legal_moves()[0]
    played = move.card
    s2 = s.apply(move)
    assert s2.neutral == played                 # played card -> neutral
    assert "Monkey" in s2.hands[RED]            # old neutral -> hand
    assert played not in s2.hands[RED]          # played card left the hand
    assert s2.to_move == BLUE                    # turn flipped


# --- Notation round-trip -----------------------------------------------------

def test_notation_roundtrip_over_legal_moves():
    states = [
        _start(red_hand=("Tiger", "Crab")),
        _start(red_hand=("Dragon", "Elephant"), blue_hand=("Rooster", "Ox")),
        _start(red_hand=("Monkey", "Mantis")).apply(
            _start(red_hand=("Monkey", "Mantis")).legal_moves()[0]),
    ]
    for s in states:
        for m in s.legal_moves():
            assert s.parse_move(s.move_to_str(m)) == m


def test_notation_pass_roundtrip():
    s = _start()
    passed = Move("Tiger", None, None)
    assert s.move_to_str(passed) == "Tiger:--"
    assert s.parse_move("Tiger:--") == passed


# --- Canonicalisation --------------------------------------------------------

def test_canonical_identity_for_red():
    """Red to move is already canonical."""
    s = _start()
    assert s.canonical() == s


def test_canonical_idempotent():
    """canonical() always yields a Red-to-move state, so a second call is a no-op."""
    # Reach a Blue-to-move state by playing one Red move.
    s = _start().apply(_start().legal_moves()[0])
    assert s.to_move == BLUE
    c = s.canonical()
    assert c.to_move == RED
    assert c.canonical() == c


def test_canonical_preserves_move_count():
    """Legal moves correspond one-for-one under canonicalisation."""
    s = _start().apply(_start().legal_moves()[0])      # Blue to move
    assert len(s.legal_moves()) == len(s.canonical().legal_moves())


def test_canonical_double_rotation_is_geometric():
    """Rotating a Blue-to-move board into canonical form maps pieces correctly."""
    board = _empty_board()
    board[0][1] = BLUE_MASTER       # should map to (4,3) as RED_MASTER
    board[3][4] = RED_STUDENT       # should map to (1,0) as BLUE_STUDENT
    s = _state(board, BLUE)
    c = s.canonical()
    assert c.board[4][3] == RED_MASTER
    assert c.board[1][0] == BLUE_STUDENT
    # Hands swapped: Blue's hand is now Red's.
    assert c.hands[RED] == s.hands[BLUE]
    assert c.hands[BLUE] == s.hands[RED]


# --- Standalone runner -------------------------------------------------------

def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e!r}")
        except Exception as e:  # noqa: BLE001 - surface any error in the runner
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
