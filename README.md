# AlphaOnitama

An MCTS [Onitama](https://en.wikipedia.org/wiki/Onitama) engine with an
AlphaZero-style self-play training loop, built in two phases. **Phase 1** (this
release) is a complete, playable agent: a stdlib-only game engine, pure Monte
Carlo Tree Search with random rollouts, and a pygame GUI. **Phase 2** (optional,
behind the `[ml]` extra) replaces random rollouts with a small policy/value
network trained by self-play — AlphaZero-lite — and is gated behind Phase 1
passing its correctness oracle.

Onitama is a good target for this: 5×5, perfect information, deterministic,
two-player zero-sum, ~10¹⁵ states, branching factor ~10–40. A small network
trains on a MacBook (CPU/MPS) with no GPU cluster.

## Rules of Onitama (brief)

- 5×5 board. Two players, **Red** and **Blue**; Red moves first.
- Each player starts with 5 pawns on their back rank — the centre pawn is the
  **Master**, the other four are **Students**. Pieces have no inherent moves.
- **Cards** provide movement. The 16-card base deck deals 2 cards to each player
  and 1 to the side (the **neutral** card). Each card is a set of board offsets.
- **A turn**: pick one of your two hand cards, move one piece by one of that
  card's offsets (Blue's offsets are the mirror — a 180° rotation), then swap the
  played card with the neutral card (forced). Landing on an enemy piece captures
  it; landing on a friendly piece or off the board is illegal.
- **Win** immediately by either:
  - **Way of the Stone** — capture the enemy Master.
  - **Way of the Stream** — move your Master onto the enemy's Temple Arch
    (`c5` for Red, `c1` for Blue).
- If you have **no legal piece move**, you must still swap a card with neutral
  and pass.

Full rulebook: <https://www.arcanewonders.com/wp-content/uploads/2020/01/Onitama-Rulebook.pdf>.

## Notation

There is no standard Onitama notation, so we define one and use it everywhere
(move I/O, logs, GUI entry, test fixtures).

**Coordinates.** Files `a`–`e` left-to-right from Red's perspective. Ranks `1`–`5`,
with rank `1` = Red's back rank (Red's Temple Arch is `c1`) and rank `5` = Blue's
back rank (Blue's Temple Arch is `c5`). A square is `file+rank`, e.g. `c1`, `a5`,
`e3`.

**Move string.** `Card:from-to`
- `Tiger:c1-c3` — play the Tiger card, move the piece on `c1` to `c3`.
- Capture is implicit: if `to` holds an enemy piece, it is captured (no mark).
- The card swap is implicit and forced (played card → neutral, old neutral → hand).
- Master vs Student is not encoded; it is whatever stands on `from`.
- **Forced pass** (zero legal piece moves): `Card:--` means "swap this card with
  neutral, move nothing." Only legal when no piece move is available.

**Game log.** One move per line, prefixed with the move number and side:

```
1. R Tiger:c1-c3
1. B Crab:c5-c4
2. R Boar:c3-c4   (captures)
```

`game.py` provides `GameState.parse_move(str)` and `GameState.move_to_str(Move)`,
which round-trip (tested).

## Install

```bash
./setup.sh                              # installs uv if needed, creates .venv, installs Phase 1 + dev
uv run onitama --mode human-vs-mcts     # play against MCTS in the GUI
```

Phase 1 needs only `pygame` and `pyperplan`. PyTorch is **not** installed by
default; opt in with `uv pip install -e '.[ml]'` for Phase 2.

## Usage

All modes go through `play.py` (the `onitama` console script):

| Invocation | What it does |
|---|---|
| `uv run onitama --mode human-vs-mcts --iterations 2000` | GUI: you (Red) vs MCTS (Blue), 2000 sims/move. |
| `uv run onitama --mode mcts-vs-mcts` | GUI: watch two MCTS agents play. |
| `uv run onitama --mode mcts-vs-random --games 50` | Headless batch; prints MCTS's win rate vs random (the debugging oracle). |
| `uv run onitama --mode random-vs-random --games 200` | Headless sanity baseline — should be ~50/50. |
| `uv run onitama --mode human-vs-neural --checkpoint path.pt` | Phase 2: play the trained net. |
| `… --ascii` | Force text rendering / headless play (works without a display). |
| `… --seed N` | Seed the RNG for reproducibility. |

The headless batch is the **Phase 1 correctness oracle**: two `RandomAgent`s
should be ~50/50, and MCTS at ≥1000 iterations should win ~95%+ vs random and
climb monotonically with the iteration count. A skewed random-vs-random result,
or MCTS failing to dominate, points at a win-check or backprop-sign bug.

Run the tests with `uv run pytest onitama/test_game.py`, or without pytest via
`python -m onitama.test_game`.

## Project layout

```
onitama/
  cards.py        # CARDS (16-card deck) + Red→Blue offset mirroring. No deps.
  game.py         # GameState: move gen, apply, win check, canonical(), notation. Depends on cards.
  mcts.py         # Node + four-phase MCTS (UCB1 + random rollout), pluggable for Phase 2. Depends on game.
  agents.py       # RandomAgent, MCTSAgent, HumanAgent (+ Phase 2 NeuralMCTSAgent). Depends on game, mcts.
  play.py         # CLI driver: deal, game loop, headless batch oracle, ASCII fallback, argparse. Depends on agents, game.
  gui.py          # pygame board: render + click-to-move. Depends on game, agents, play; engine never imports it.
  test_game.py    # unit tests: win conditions, move counts, card swap, notation, canonical symmetry.
  pddl_runner.py  # wires pyperplan to the hand-written PDDL (stub; gitignored).
  onitama_one_turn.pddl  # hand-filled PDDL exercise (stub header; gitignored).
  net.py / selfplay.py / train.py  # Phase 2 (not in this release).
```

## How MCTS works here

See `mcts.py` — it is documented to be readable on its own. Each search iteration
runs the four MCTS stages:

1. **Select** (selection) — descend from the root by a *tree policy* until
   reaching a node that is not fully expanded or is terminal.
2. **Expand** (expansion) — add one child for an untried move.
3. **Simulate** (simulation / evaluation) — estimate the leaf's value. Phase 1
   plays a uniform-random rollout to a terminal (or a 200-ply cap); Phase 2
   instead reads the network's value head (the `evaluate_leaf` hook).
4. **Backpropagate** (backpropagation) — carry the value to the root, **flipping
   its sign at each level** (zero-sum, alternating players).

The chosen move is the **most-visited** root child (the robust statistic), and the
full visit-count distribution is returned for Phase 2's policy target. The tree
policy is pluggable: Phase 1 uses **UCB1**,
`W/N + c·sqrt(ln(parent.N)/N)` (mean value + exploration bonus); Phase 2 swaps in
**PUCT**, `Q + c_puct·P·sqrt(parent.N)/(1+N)`, which folds in the network's prior
`P` instead of relying on raw visit statistics.

## Phase 2 — AlphaZero-lite (not in this release)

Gated behind Phase 1 passing its oracle. Install the extra and (once built) run
the loop:

```bash
uv pip install -e '.[ml]'        # torch + numpy
# python -m onitama.selfplay      # generate self-play games
# python -m onitama.train         # train the net, checkpoint, gated eval
uv run onitama --mode human-vs-neural --checkpoint runs/best.pt
```

Network: a canonicalised board tensor (own/enemy Master + Student planes, plus
card-identity planes) through a small ResNet trunk into two heads — a masked
policy over the fixed move encoding and a `tanh` value in `[-1, 1]` from the
side-to-move's perspective. MPS is used on Apple Silicon when available.

**Phase 2 validation oracle:** even a small trained net should beat the Phase 1
rollout-MCTS at equal sim count, and beat `RandomAgent` ~100%. If training
"succeeds" (loss drops) but the agent is weak, suspect a perspective/sign/target
bug and audit canonicalisation and the `z` assignment first.

## Design notes / gotchas

These are the silent failure modes of AlphaZero builds — no exception, just a
weak agent. They were guarded against from the start:

1. **Perspective bugs.** Value and policy must always be from the side-to-move's
   perspective. `GameState.canonical()` flips the board and swaps colours so the
   net always sees "me to move from the bottom." This is the #1 silent killer.
2. **Backprop sign flip.** `mcts.py` flips the value sign at every level on the
   way to the root, because the players alternate in a zero-sum game. A node's
   `W` is stored from the perspective of the player who *moved into* it.
3. **Win/terminal detection.** `GameState.winner()` is deliberately simple and is
   the first thing the tests pin down — a wrong win-check makes the agent learn
   nonsense.
4. **Stale/leaky training targets (Phase 2).** The MCTS visit-count policy target
   and the game-outcome value target are detached constants when training.
5. **Data/replay mismatch (Phase 2).** Self-play stores
   `(canonical_state, π_visit, z_outcome)` triples; `z` is filled in only at game
   end and assigned to every stored position from *that position's* side-to-move
   perspective.

There is also a small **PDDL artefact** (`onitama_one_turn.pddl` +
`pddl_runner.py`, both gitignored) — a hand-fill exercise encoding a single
passive-opponent turn in classical planning, to contrast planning with the
adversarial game search the engine actually uses.
