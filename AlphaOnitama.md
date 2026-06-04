# AlphaOnitama — Build Spec

An MCTS Onitama engine with an AlphaZero-style self-play training loop. Two phases,
built and validated in order:

- **Phase 1**: game engine + pure MCTS (random rollouts) + GUI. A complete, playable
  agent. Must be correct and validated before Phase 2 begins.
- **Phase 2**: replace random rollouts with a policy/value net; self-play training loop;
  the trained net guides MCTS. AlphaZero-lite.

Onitama is a good target: 5x5, perfect information, deterministic, two-player zero-sum,
~10^15 states, branching ~10-40. A small net trains on a MacBook (CPU/MPS), no GPU
cluster needed.

Language: Python 3. Phase 1 core is stdlib-only. Phase 2 adds PyTorch. GUI via pygame.

> READ THIS FIRST — failure modes that kill AlphaZero builds silently. Engineer
> against these from the start; they produce no exception, just a weak agent:
> 1. **Perspective bugs.** Value and policy must always be from the perspective of
>    the player to move. Net input is canonicalised so the side-to-move is always
>    "us". Every value sign and policy index is relative to side-to-move, never to a
>    fixed colour. This is the #1 silent killer.
> 2. **Backprop sign flip in MCTS** (zero-sum, alternating players) — see Phase 1.
> 3. **Win/terminal detection wrong** — agent "learns" nonsense. Tested before any
>    search is written.
> 4. **Stale/leaky training targets.** The MCTS visit-count policy target and the
>    game-outcome value target must be detached constants when training the net.
> 5. **Data/replay mismatch.** Self-play stores (canonical_state, π_visit, z_outcome)
>    triples; z is filled in only when the game ends and assigned to every position
>    from that position's side-to-move perspective.

---

## DOCUMENTATION REQUIREMENT

The repo must be readable and self-explanatory. Produce:

- **`README.md`** at repo root, covering:
  - one-paragraph overview (what it is, the two phases).
  - the rules of Onitama, briefly, with a link to a full rulebook.
  - the move/coordinate **notation** (reproduce the NOTATION section).
  - install: `./setup.sh`, then `uv run onitama --mode human-vs-mcts`.
  - usage: every `play.py` mode with an example invocation and what it does.
  - project layout: one line per module (what it contains, what it depends on).
  - a short "how MCTS works here" section pointing readers to `mcts.py`, summarising
    the four phases and the UCB1/PUCT distinction in a few sentences.
  - Phase 2 section: how to install `[ml]`, run self-play/training, load a checkpoint,
    and the validation oracle (trained net should beat Phase 1 rollout-MCTS).
  - a "design notes / gotchas" section listing the five silent-failure modes from the
    spec header, so a future reader knows what was deliberately guarded against.
- **Module docstrings**: every module opens with a docstring stating its purpose and
  its place in the system.
- **Public-function docstrings**: every public function/method gets a docstring with
  its contract (args, return, key invariants). `mcts.py` and `game.py` get the
  fullest treatment per their commenting notes below.
- **Type hints** on all public signatures.
- A `docs/` note is optional; the README + docstrings are the requirement.

General readability bar across the codebase: clear names over clever ones, small
focused functions, comments that explain *why* not *what*, and no undocumented magic
constants (the 200-ply cap, `c_puct`, channel counts, etc. are named and explained
where defined).

---

## NOTATION (define once, use everywhere)

There is no standard Onitama notation (chess has PGN, draughts has PDN; Onitama has
none). We define one. Use it for move I/O, game logs, GUI move entry, and test
fixtures.

**Coordinates.** Files `a`-`e` left-to-right from Red's perspective. Ranks `1`-`5`,
rank `1` = Red's back rank (Red's Temple Arch is `c1`), rank `5` = Blue's back rank
(Blue's Temple Arch is `c5`). A square is `file+rank`, e.g. `c1`, `a5`, `e3`.

**Move string.** `Card:from-to`
- `Tiger:c1-c3` — play Tiger card, move piece from c1 to c3.
- Capture is implicit: if `to` holds an enemy piece, it's captured. No special mark.
- The card swap is implicit and forced (played card → neutral, old neutral → hand).
- Master vs Student is not encoded in the move; it's determined by what's on `from`.
- **Forced pass** (zero legal moves): `Card:--` means "swap this card with neutral,
  move nothing." Only legal when the player has no piece move available.

**Game log.** One move per line, prefixed with ply number and side:
```
1. R Tiger:c1-c3
1. B Crab:c5-c4
2. R Boar:c3-c4   (captures)
...
```
Add a `parse_move(str, state) -> Move` and `move_to_str(Move, state) -> str` in
`game.py`. Round-trip them in tests.

---

## Phase 1 — Engine + MCTS + GUI

### P1.1 Game rules — implement exactly

- 5x5 board. Players Red and Blue. Red moves first (engine fixes Red as first; the
  real game decides by the neutral card's stamp, irrelevant for self-play).
- Each player: 5 pawns on back rank, centre pawn is the **Master**, other four are
  **Students**. Movement comes from cards, not piece type.
- **Cards**: 16-card base deck. Each card is a set of `(dr, dc)` offsets from Red's
  POV. At start, deal 2 to each player, 1 to the side (neutral).
- **Turn**:
  1. Choose one of your 2 hand cards.
  2. Move one piece by one of that card's offsets (Blue's offsets are the engine
     mirror: negate both `dr` and `dc`).
  3. Played card ↔ neutral swap (forced).
  4. Landing on an enemy piece captures it. Landing on own piece or off-board is
     illegal.
- **Win** (immediate):
  - **Way of the Stone**: capture enemy Master.
  - **Way of the Stream**: move your Master onto enemy's Temple Arch (`c5` for Red,
    `c1` for Blue).
- **Zero legal piece moves**: player must still swap a card with neutral and pass
  (the `Card:--` move). Comment this in code to verify against rulebook; in practice
  it's rare but must be handled or move-gen returns empty and search breaks.

### P1.2 Module layout
```
onitama/
  __init__.py
  cards.py          # CARDS dict, mirroring logic
  game.py           # GameState; move gen; apply; win check; notation parse/format
  mcts.py           # Node, MCTS (UCB1, pluggable rollout/prior hooks)
  agents.py         # RandomAgent, MCTSAgent, HumanAgent, NeuralMCTSAgent (Phase 2)
  gui.py            # pygame board: render, click-to-move, card display
  play.py           # CLI driver: human/agent/batch modes, ASCII fallback
  pddl_runner.py    # wires pyperplan to the hand-written PDDL file
  net.py            # Phase 2: policy/value network
  selfplay.py       # Phase 2: self-play data generation
  train.py          # Phase 2: training loop
  test_game.py      # unit tests: move gen, win conditions, notation round-trip
  onitama_one_turn.pddl   # blank-ish, hand-filled by user
README.md           # top-level documentation (see DOCUMENTATION REQUIREMENT)
setup.sh
pyproject.toml
```

### P1.3 game.py — GameState

Immutable-ish (copy on apply; small game, perf is fine).

Fields:
- `board`: 5x5, each square in `{EMPTY, RED_STUDENT, RED_MASTER, BLUE_STUDENT, BLUE_MASTER}`.
- `hands`: `{RED: [c1, c2], BLUE: [c3, c4]}`
- `neutral`: single card name
- `to_move`: RED or BLUE

Methods:
- `legal_moves() -> list[Move]`, `Move = (card_name, from_sq, to_sq)`. For each piece
  of `to_move`, each hand card, each offset (mirrored if Blue): compute target; keep
  if on-board and not own piece. If empty, return the forced-pass moves (one per hand
  card, `to_sq = from_sq = None`).
- `apply(move) -> GameState`: move piece, resolve capture, swap played card ↔ neutral,
  flip `to_move`.
- `winner() -> {RED, BLUE, None}`; `is_terminal() -> bool`.
- `canonical() -> GameState`: returns the state from the side-to-move's perspective
  (board flipped + colours swapped if it's Blue to move) so the net always sees "me
  to move from the bottom." **Critical for Phase 2.** Build it in Phase 1 and test it
  (canonical(canonical-symmetry), apply round-trips).
- `parse_move` / `move_to_str` per the NOTATION section.

**Commenting (game.py):** the state representation and `apply` form a standard
state-space search formalisation; document them as such. One comment block at the top
of the class: `legal_moves` = the applicable-action set, `apply` = the
successor/transition function, `winner`/`is_terminal` = the goal test, and the
(board, hands, neutral, to_move) tuple = the complete state. This framing (state,
actions, transition, goal test) is worth stating explicitly for any reader. Don't
pepper every line.

Win check correctness is the most common silent-MCTS-bug source. Test it first.

### P1.4 mcts.py — the conceptual core

**Commenting (mcts.py):** this is the conceptual heart of the codebase, so document
it thoroughly. Each of the four phases gets a docstring naming it + one-line purpose.
The UCB1 line gets an inline comment splitting the exploitation term from the
exploration term. The backprop sign flip gets a comment explaining *why* it flips
(zero-sum, alternating players). Name the concepts where the code embodies them:
UCB1 / bandit exploration-exploitation tradeoff, search as a policy-improvement
operator, robust child selection by visit count vs value. A reader should be able to
learn how MCTS works from this file alone.

```
Node:
  state
  parent
  move_that_led_here
  children: dict[Move, Node]
  N: visit count
  W: total value (perspective: player who MOVED INTO this node)
  P: prior prob of the move into this node   # Phase 2; ignored/uniform in Phase 1
  untried_moves: list[Move]
```

Phases:
- **Selection**: from root, while fully expanded and non-terminal, descend to child
  maximising the tree policy. Phase 1 uses UCB1:
  ```
  UCB1(child) = (W/N) + c * sqrt( ln(parent.N) / child.N )    # c = sqrt(2) default
  ```
  N=0 children → +inf (must-visit). **Phase 2** swaps this for the PUCT rule:
  ```
  PUCT(child) = Q(child) + c_puct * P(child) * sqrt(parent.N) / (1 + child.N)
  ```
  Keep the tree-policy function pluggable so Phase 2 swaps it without touching the
  rest. Expose `c` / `c_puct`.
- **Expansion**: pop one untried move, create child. Phase 2: set child priors `P`
  from the net's policy head at expansion time.
- **Simulation/evaluation**: Phase 1 — uniform-random rollout to terminal or 200-ply
  cap. Phase 2 — **no rollout**; evaluate the leaf with the net's value head and
  return that. Make this a `evaluate_leaf(node)` hook: rollout in P1, net in P2.
- **Backpropagation**: walk to root, `N += 1`, `W += value`, **flip value sign each
  level** (zero-sum). A Red win is +1 for Red-moved nodes, -1 for Blue-moved nodes.
  Second classic bug after win-check; get the sign right.

Entry point:
```
mcts_search(root_state, iterations=1000, evaluate_leaf=rollout, tree_policy=ucb1) -> (Move, visit_counts)
  build root
  repeat iterations: select -> expand -> evaluate_leaf -> backprop
  return argmax-N child move, and the full visit-count distribution over root moves
```
Returning the visit-count distribution is needed for Phase 2 training targets. Phase 1
just uses the argmax.

Reward/value convention:
- terminal, side-to-move-at-leaf wins: +1; loses: -1
- 200-ply cap with no terminal (P1 rollout): 0
- P2: value head output in [-1, 1], from leaf side-to-move perspective
At low iteration counts most P1 rollouts hit the cap → near-zero signal → near-random
play. Signal emerges as rollouts reach terminals. **Win rate vs RandomAgent climbing
monotonically with iterations is the Phase 1 correctness oracle.**

### P1.5 agents.py
- `RandomAgent.choose(state) -> Move`: uniform over legal.
- `MCTSAgent(iterations).choose(state) -> Move`: `mcts_search` with rollout eval.
- `HumanAgent.choose(state) -> Move`: GUI click or CLI numbered-move entry.
- `NeuralMCTSAgent(net, iterations)`: Phase 2, `mcts_search` with net eval + PUCT.

### P1.6 gui.py — pygame board

This is the payoff; make it actually nice to play.
- Window: 5x5 board, ~100px squares. Red pieces bottom, Blue top. Master visually
  distinct from Students (e.g. ring/crown overlay or different shade).
- Render below/beside the board: Red's 2 cards, Blue's 2 cards, the neutral card.
  Each card drawn as a small 5x5 grid showing its offset pattern with the centre
  marked. (Card pattern rendering reads straight from `CARDS`.)
- **Click-to-move**: click a piece → highlight all legal destinations for the
  currently selectable cards → click a destination → if the move is achievable by
  exactly one card, play it; if two cards both reach it, prompt which card (click the
  card). Illegal clicks are ignored.
- Show whose turn, last move (in notation), and a win banner on terminal.
- Modes wired through `play.py`: human-vs-MCTS, human-vs-neural, watch MCTS-vs-MCTS.
- Keep rendering logic separate from game logic; gui.py imports game.py, never the
  reverse.

ASCII fallback in `play.py` for headless runs (batch eval), so the engine is testable
without a display.

### P1.7 play.py
- `uv run onitama --mode human-vs-mcts --iterations 2000` → launches GUI.
- `uv run onitama --mode mcts-vs-random --games 50` → headless, prints win rate
  (the debugging oracle; two RandomAgents should be ~50/50, MCTS@1000+ should be ~95%+
  vs random; if not, win-check or backprop sign is wrong).
- `uv run onitama --mode human-vs-neural --checkpoint path.pt` → Phase 2.
- `--ascii` forces text rendering.

### P1.8 test_game.py (before trusting MCTS)
- capture-the-Master sequence → correct `winner()`.
- Master-reaches-enemy-Temple-Arch sequence → correct `winner()`.
- legal move count on start position for a named card pair = hand-counted value.
- card swap: played → neutral, old neutral → hand.
- notation round-trip: `parse_move(move_to_str(m)) == m` over a batch of legal moves.
- `canonical()` symmetry + that legal moves correspond under canonicalisation.

### P1.9 Build order (do not skip ahead)
1. `cards.py` structure + 2 example cards (user fills the rest).
2. `game.py` + `test_game.py`. **No MCTS until win-check + notation tests pass.**
3. `RandomAgent` + headless `mcts-vs-random` harness (two RandomAgents ≈ 50/50).
4. `mcts.py` (UCB1 + rollout). Validate via batch → expect lopsided win rate.
5. `gui.py` + human play. **Phase 1 done — a complete, playable agent. Validate it
   fully before moving to Phase 2.**

---

## Phase 2 — AlphaZero-lite (do not start until Phase 1 is validated)

Gated behind Phase 1 passing its oracle. If Phase 1's MCTS doesn't crush random, do
not build Phase 2 on top of a broken base.

### P2.1 net.py
- Input: canonical board tensor. Planes: own Master, own Students, enemy Master,
  enemy Students, plus card-identity planes (which 5 of 16 cards are where: own hand
  ×2, enemy hand ×2, neutral). Keep encoding explicit and documented.
- Trunk: a few conv layers (small ResNet, 3-5 blocks, 64 channels is plenty for 5x5).
- **Two heads**:
  - policy: logits over the move space. Define a fixed move encoding (card-index ×
    from-square × offset-index, masked to legal). Document the indexing; mask illegal
    moves before softmax.
  - value: scalar in [-1, 1] via tanh, from side-to-move perspective.
- MPS device on M2 if available, else CPU.

### P2.2 selfplay.py
- Play games with `NeuralMCTSAgent` against itself, ~100-400 MCTS sims/move.
- Add Dirichlet noise to root priors for exploration (AlphaZero standard).
- Temperature: sample moves ∝ visit counts for first ~10 plies (exploration), then
  argmax.
- Store per position: `(canonical_state_tensor, π_visit, side_to_move)`. On game end,
  set `z` = +1/-1/0 from each stored position's side-to-move perspective.
- These targets are constants when training (detach); see failure mode #4.

### P2.3 train.py
- Loss: `(z - v)^2  -  π_visit · log p  +  λ‖θ‖²` (value MSE + policy cross-entropy +
  L2). Comment the three terms.
- Loop: generate self-play games → train on the buffer → replace agent net → repeat.
  Keep a replay buffer of recent games.
- Checkpoint every N iterations. Evaluate new net vs previous best over a fixed match;
  promote only if it wins clearly (gating prevents regression).
- **Validation oracle for Phase 2**: trained net (even small) should beat the Phase 1
  rollout-MCTS at equal sim count, and beat RandomAgent ~100%. If training "succeeds"
  (loss drops) but the agent is weak, it's a perspective/sign/target bug — audit
  canonicalisation and the z-assignment first.

### P2.4 Phase 2 scope discipline
In scope: the loop above, small net, MPS, checkpointing, gated eval.
Out of scope (don't gold-plate): distributed self-play, RAVE, tree reuse between moves,
opening books, hyperparameter sweeps. Get one honest training run beating Phase 1,
then stop.

---

## PDDL artefact (separate from the build)

Add `onitama/onitama_one_turn.pddl`, blank except this header:
```pddl
;; Onitama: one-player turn encoded in PDDL.
;; Domain + problem stub — to be filled in by hand.
;;
;; Encode:
;;   - predicates: piece positions, card ownership, neutral card, whose turn
;;   - action schema for a student move
;;   - action schema for a master move
;;   - card-swap mechanic in the effect (the interesting bit)
;;   - a sample problem: specific position, hands, goal
;;
;; Assumes a passive opponent (no adversarial response). That's the point —
;; see where PDDL's scope ends (planning vs game search).
```
Do NOT generate PDDL content. Left for manual completion.

`pddl_runner.py`: wire pyperplan to this file, print the plan. Stub it; it just needs
to call the planner and print output once the file is filled. FF is NOT used (no
homebrew formula, manual C build). pyperplan: pure Python, installs via uv, M2-native,
same PDDL format.

---

## Setup — Mac (Apple Silicon, M2 Pro)

`setup.sh`:
```bash
#!/usr/bin/env bash
set -e
if ! command -v brew &>/dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
if ! command -v uv &>/dev/null; then
  brew install uv
fi
uv venv .venv
uv pip install -e ".[dev]"      # Phase 1 + tooling
echo "Phase 1 ready. Run: uv run onitama --mode human-vs-mcts"
echo "For Phase 2: uv pip install -e '.[ml]'  then see train.py"
```

`pyproject.toml`:
- `[project]` metadata, `requires-python = ">=3.10"`.
- `dependencies = ["pygame", "pyperplan"]`  (Phase 1 playable + PDDL runner).
- `[project.optional-dependencies]`
  - `dev = ["pytest"]`
  - `ml = ["torch", "numpy"]`  (Phase 2 only; keep heavy dep out of the default install).
- `[project.scripts] onitama = "onitama.play:main"`.

So `uv run onitama` works immediately; PyTorch only installs when you opt into `[ml]`.

---

### Stub for cards.py (fill the 14 remaining from the base set)
# Offsets are (dr, dc) from the moving piece, RED's perspective.
# (-1, 0) = one square "forward" (toward Blue) for Red. (+1,0) = backward.
# (0, -1) = one square to Red's left. Engine mirrors (negate dr and dc) for Blue.
CARDS = {
    "Tiger":   [(-2, 0), (1, 0)],
    "Crab":    [(-1, 0), (0, -2), (0, 2)],
    "Monkey":  [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "Crane":   [(-1, 0), (1, -1), (1, 1)],
    "Dragon":  [(-1, -2), (-1, 2), (1, -1), (1, 1)],
    "Elephant":[(0, -1), (0, 1), (-1, -1), (-1, 1)],
    "Boar":    [(-1, 0), (0, -1), (0, 1)],
    "Mantis":  [(-1, -1), (-1, 1), (1, 0)],
    "Rooster": [(0, -1), (-1, 1), (1, -1), (0, 1)],
    "Ox":      [(-1, 0), (0, 1), (1, 0)],
    "Horse":   [(-1, 0), (0, -1), (1, 0)],
    "Frog":    [(-1, -1), (0, -2), (1, 1)],
    "Rabbit":  [(-1, 1), (0, 2), (1, -1)],
    "Goose":   [(0, -1), (-1, -1), (1, 1), (0, 1)],
    "Cobra":   [(0, -1), (-1, 1), (1, 1)],
    "Eel":     [(-1, -1), (0, 1), (1, -1)],
}
