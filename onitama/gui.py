"""Pygame GUI: render the board and cards, and drive click-to-move human play.

Rendering only — all rules live in :mod:`onitama.game`. This module imports the
engine; the engine never imports this. Red pieces sit at the bottom, Blue at the
top (matching the engine's row-0-at-top / rank-5 layout). Masters are drawn with
a crown ring to distinguish them from Students. Each card is a small 5x5 grid
read straight from :data:`onitama.cards.CARDS`, centre marked, offsets filled.

Click-to-move: click a friendly piece to highlight its legal destinations; click
a destination to play. If two hand cards both reach the chosen square, you are
prompted to click the card to disambiguate. Modes are wired through
:func:`run` (called by ``play.py``): human-vs-MCTS, watch MCTS-vs-MCTS, and the
Phase 2 human-vs-neural.
"""

from __future__ import annotations

from typing import Optional

from .agents import MCTSAgent, RandomAgent
from .cards import CARDS
from .game import (
    BLUE,
    RED,
    BLUE_MASTER,
    BLUE_STUDENT,
    EMPTY,
    RED_MASTER,
    RED_STUDENT,
    GameState,
    Move,
    rc_to_sq,
    sq_to_rc,
)
from .play import deal

try:
    import pygame
except ImportError:  # pragma: no cover - GUI optional
    pygame = None  # type: ignore[assignment]

# --- Layout constants (named so there are no magic numbers in the draw code) --
SQ = 100                      # board square size in px
BOARD_PX = SQ * 5             # 5x5 board edge
MARGIN = 20
TOPBAR = 70                   # space above the board for status text
BOTTOMBAR = 56                # space below the board for the banner
PANEL_W = 200                 # right-hand card panel width
WIN_W = MARGIN * 2 + BOARD_PX + PANEL_W
WIN_H = TOPBAR + BOARD_PX + BOTTOMBAR
BOARD_X, BOARD_Y = MARGIN, TOPBAR
PANEL_X = MARGIN + BOARD_PX + MARGIN

CARD_CELL = 14                # mini-grid cell for a card's 5x5 pattern
CARD_PX = CARD_CELL * 5

# --- Palette -----------------------------------------------------------------
C_BG = (28, 30, 36)
C_LIGHT = (222, 210, 180)
C_DARK = (150, 130, 96)
C_RED = (200, 70, 60)
C_BLUE = (70, 110, 200)
C_CROWN = (245, 220, 120)
C_HILITE = (90, 200, 120)
C_SELECT = (240, 230, 120)
C_TEXT = (235, 235, 235)
C_CARD_BG = (52, 55, 64)
C_CARD_ON = (210, 200, 120)
C_CARD_CENTRE = (235, 235, 235)
C_CARD_ACTIVE = (90, 200, 120)


def _agent_for(role: str, iterations: int, checkpoint: Optional[str]):
    """Build the non-human agent for a side from a role string."""
    if role == "mcts":
        return MCTSAgent(iterations)
    if role == "random":
        return RandomAgent()
    if role == "neural":
        # Phase 2: load a NeuralMCTSAgent from the checkpoint when available.
        raise NotImplementedError(
            "human-vs-neural needs Phase 2 (net.py + a --checkpoint); not built yet."
        )
    raise ValueError(f"unknown agent role {role!r}")


def run(mode: str = "human-vs-mcts", iterations: int = 1000,
        checkpoint: Optional[str] = None) -> None:
    """Open the GUI for a game and block until the window is closed.

    Args:
        mode: ``human-vs-mcts`` (you are Red), ``mcts-vs-mcts`` (watch), or
            ``human-vs-neural`` (Phase 2).
        iterations: MCTS iterations per move for any MCTS side.
        checkpoint: Phase 2 net checkpoint path (human-vs-neural only).
    """
    if pygame is None:
        raise RuntimeError(
            "pygame is not installed. Install Phase 1 deps (./setup.sh) or run a "
            "headless mode with --ascii."
        )

    # Map mode -> per-colour controller. "human" means click-to-move.
    roles = {
        "human-vs-mcts": {RED: "human", BLUE: "mcts"},
        "human-vs-neural": {RED: "human", BLUE: "neural"},
        "mcts-vs-mcts": {RED: "mcts", BLUE: "mcts"},
    }.get(mode)
    if roles is None:
        raise ValueError(f"mode {mode!r} is not a GUI mode")
    agents = {
        color: (None if role == "human" else _agent_for(role, iterations, checkpoint))
        for color, role in roles.items()
    }

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"AlphaOnitama — {mode}")
    font = pygame.font.SysFont("menlo,monospace", 18)
    small = pygame.font.SysFont("menlo,monospace", 12)
    big = pygame.font.SysFont("menlo,monospace", 30, bold=True)
    clock = pygame.time.Clock()

    state = deal()
    ui = _Interaction()          # transient human-click state
    last_move_str: Optional[str] = None
    running = True

    while running:
        human_turn = not state.is_terminal() and agents[state.to_move] is None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                  and human_turn):
                move = ui.handle_click(state, event.pos)
                if move is not None:
                    last_move_str = state.move_to_str(move)
                    state = state.apply(move)
                    ui.reset()

        # Agent move: render a "thinking" frame first so the window stays live.
        if not state.is_terminal() and agents[state.to_move] is not None:
            _draw(screen, state, ui, font, small, big, last_move_str, thinking=True)
            pygame.display.flip()
            pygame.event.pump()
            move = agents[state.to_move].choose(state)
            last_move_str = state.move_to_str(move)
            state = state.apply(move)
            pygame.time.wait(450 if all(a is not None for a in agents.values()) else 150)

        _draw(screen, state, ui, font, small, big, last_move_str, thinking=False)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


class _Interaction:
    """Mutable click-to-move state for a human turn (selection / disambiguation)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.selected_sq: Optional[str] = None
        # dst square -> list of cards that reach it from the selected piece
        self.dests: dict[str, list[str]] = {}
        # pending (src, dst) awaiting a card click because two cards both reach dst
        self.pending: Optional[tuple[str, str]] = None
        self.pending_cards: list[str] = []

    def _dests_from(self, state: GameState, src: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for m in state.legal_moves():
            if m.src == src:
                out.setdefault(m.dst, []).append(m.card)
        return out

    def handle_click(self, state: GameState, pos: tuple[int, int]) -> Optional[Move]:
        """Translate a click into a completed :class:`Move`, or ``None`` if more input
        is needed. Updates selection/disambiguation state in place.
        """
        legal = state.legal_moves()

        # Forced-pass position: every legal move is a card-only swap. Click a hand
        # card (in the panel) to pass with it.
        if legal and all(m.is_pass for m in legal):
            card = _card_at(pos, state)
            if card is not None and any(m.card == card for m in legal):
                return Move(card, None, None)
            return None

        # Awaiting card disambiguation: a destination needed two cards.
        if self.pending is not None:
            card = _card_at(pos, state)
            if card in self.pending_cards:
                src, dst = self.pending
                return Move(card, src, dst)
            # clicking elsewhere cancels the pending choice
            self.pending, self.pending_cards = None, []
            return None

        rc = _square_at(pos)
        if rc is None:
            return None
        sq = rc_to_sq(*rc)

        # Click a highlighted destination -> commit (or ask which card).
        if self.selected_sq is not None and sq in self.dests:
            cards = self.dests[sq]
            if len(cards) == 1:
                return Move(cards[0], self.selected_sq, sq)
            self.pending = (self.selected_sq, sq)
            self.pending_cards = cards
            return None

        # Otherwise (re)select a friendly piece, or clear.
        cell = state.board[rc[0]][rc[1]]
        if _cell_color(cell) == state.to_move:
            self.selected_sq = sq
            self.dests = self._dests_from(state, sq)
        else:
            self.reset()
        return None


# --- Pixel <-> board / card hit-testing --------------------------------------

def _square_at(pos: tuple[int, int]) -> Optional[tuple[int, int]]:
    x, y = pos
    if BOARD_X <= x < BOARD_X + BOARD_PX and BOARD_Y <= y < BOARD_Y + BOARD_PX:
        return (y - BOARD_Y) // SQ, (x - BOARD_X) // SQ
    return None


def _card_layout(state: GameState) -> list[tuple[str, str, int, int]]:
    """Return ``(label, card_name, x, y)`` for the five panel cards, top to bottom."""
    entries = [
        (f"{BLUE} hand", state.hands[BLUE][0]),
        (f"{BLUE} hand", state.hands[BLUE][1]),
        ("neutral", state.neutral),
        (f"{RED} hand", state.hands[RED][0]),
        (f"{RED} hand", state.hands[RED][1]),
    ]
    gap = (BOARD_PX - 5 * (CARD_PX + 16)) // 4 if BOARD_PX > 5 * (CARD_PX + 16) else 8
    out = []
    y = BOARD_Y
    for label, name in entries:
        out.append((label, name, PANEL_X, y + 16))
        y += CARD_PX + 16 + gap
    return out


def _card_at(pos: tuple[int, int], state: GameState) -> Optional[str]:
    x, y = pos
    for _label, name, cx, cy in _card_layout(state):
        if cx <= x < cx + CARD_PX and cy <= y < cy + CARD_PX:
            return name
    return None


def _cell_color(cell: int) -> Optional[str]:
    if cell in (RED_STUDENT, RED_MASTER):
        return RED
    if cell in (BLUE_STUDENT, BLUE_MASTER):
        return BLUE
    return None


# --- Rendering ---------------------------------------------------------------

def _draw(screen, state: GameState, ui: "_Interaction", font, small, big,
          last_move_str: Optional[str], thinking: bool) -> None:
    screen.fill(C_BG)
    _draw_board(screen, state, ui)
    _draw_cards(screen, state, ui, small)
    _draw_status(screen, state, font, big, last_move_str, thinking)


def _draw_board(screen, state: GameState, ui: "_Interaction") -> None:
    for r in range(5):
        for c in range(5):
            x, y = BOARD_X + c * SQ, BOARD_Y + r * SQ
            base = C_LIGHT if (r + c) % 2 == 0 else C_DARK
            pygame.draw.rect(screen, base, (x, y, SQ, SQ))
            # Mark the two Temple Arches (c1 bottom for Red, c5 top for Blue).
            if (r, c) in ((0, 2), (4, 2)):
                pygame.draw.rect(screen, (180, 160, 120), (x, y, SQ, SQ), 4)

    # Highlight current selection and its legal destinations.
    if ui.selected_sq is not None:
        sr, sc = sq_to_rc(ui.selected_sq)
        pygame.draw.rect(screen, C_SELECT,
                         (BOARD_X + sc * SQ, BOARD_Y + sr * SQ, SQ, SQ), 5)
        for dst in ui.dests:
            dr, dc = sq_to_rc(dst)
            cx, cy = BOARD_X + dc * SQ + SQ // 2, BOARD_Y + dr * SQ + SQ // 2
            pygame.draw.circle(screen, C_HILITE, (cx, cy), 14)

    # Pieces.
    for r in range(5):
        for c in range(5):
            cell = state.board[r][c]
            if cell == EMPTY:
                continue
            cx, cy = BOARD_X + c * SQ + SQ // 2, BOARD_Y + r * SQ + SQ // 2
            color = C_RED if _cell_color(cell) == RED else C_BLUE
            pygame.draw.circle(screen, color, (cx, cy), 34)
            pygame.draw.circle(screen, (20, 20, 24), (cx, cy), 34, 3)
            if cell in (RED_MASTER, BLUE_MASTER):
                pygame.draw.circle(screen, C_CROWN, (cx, cy), 16, 4)  # crown ring


def _draw_cards(screen, state: GameState, ui: "_Interaction", small) -> None:
    # Which cards are "active" to highlight: the side-to-move's hand, or the
    # pending-disambiguation pair.
    active = set(ui.pending_cards) if ui.pending_cards else set(state.hands[state.to_move])
    for label, name, x, y in _card_layout(state):
        lbl = small.render(f"{label}: {name}", True, C_TEXT)
        screen.blit(lbl, (x, y - 14))
        _draw_card_grid(screen, name, x, y, highlight=name in active)


def _draw_card_grid(screen, name: str, x: int, y: int, highlight: bool) -> None:
    pygame.draw.rect(screen, C_CARD_BG, (x - 2, y - 2, CARD_PX + 4, CARD_PX + 4))
    if highlight:
        pygame.draw.rect(screen, C_CARD_ACTIVE, (x - 2, y - 2, CARD_PX + 4, CARD_PX + 4), 2)
    offsets = set(CARDS[name])
    for gr in range(5):
        for gc in range(5):
            cx, cy = x + gc * CARD_CELL, y + gr * CARD_CELL
            dr, dc = gr - 2, gc - 2  # offset relative to centre (2,2)
            if (dr, dc) == (0, 0):
                col = C_CARD_CENTRE
            elif (dr, dc) in offsets:
                col = C_CARD_ON
            else:
                col = (40, 42, 50)
            pygame.draw.rect(screen, col, (cx + 1, cy + 1, CARD_CELL - 2, CARD_CELL - 2))


def _draw_status(screen, state: GameState, font, big,
                 last_move_str: Optional[str], thinking: bool) -> None:
    winner = state.winner()
    if winner is not None:
        turn = f"{winner} wins!"
    elif thinking:
        turn = f"{state.to_move} thinking..."
    else:
        turn = f"{state.to_move} to move"
    screen.blit(font.render(turn, True, C_TEXT), (MARGIN, 14))
    if last_move_str:
        lm = font.render(f"last: {last_move_str}", True, C_TEXT)
        screen.blit(lm, (MARGIN, 40))

    if winner is not None:
        banner = big.render(f"{winner} wins", True, C_CROWN)
        rect = banner.get_rect(center=(BOARD_X + BOARD_PX // 2,
                                       BOARD_Y + BOARD_PX + BOTTOMBAR // 2))
        screen.blit(banner, rect)
