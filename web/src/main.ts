/**
 * Zic-Zac-Zoe - Main Game Controller
 */

// Design dimensions (portrait orientation)
const DESIGN_WIDTH = 400;
const DESIGN_HEIGHT = 740;

import {
  BoardState,
  Player,
  GameResult,
  BOARD_SIZE,
  createBoard,
  getCurrentPlayer,
  makeMove,
  checkResultFast,
  getLegalMoves,
  GameCheckResult,
} from "./game";
import { loadModel, getAIMove, Difficulty, getModelIteration } from "./ai";
import { getRulesMove, isRulesAI, detectCheckmate, CheckmateResult } from "./rules-ai";

// Timing constants (milliseconds)
const AI_MOVE_DELAY = 500;      // Delay after human move before AI responds
const AI_FIRST_MOVE_DELAY = 500; // Delay when AI goes first

// =============================================================================
// Game State
// =============================================================================

interface GameState {
  board: BoardState;
  humanPlayer: Player;
  twoPlayer: boolean;
  difficulty: Difficulty;
  gameOver: boolean;
  lastMove: number | null;
  result: GameCheckResult | null;
  lastGuardrailWeight: number | null;
  checkmate: CheckmateResult | null;
}

const state: GameState = {
  board: createBoard(),
  humanPlayer: Player.X,
  twoPlayer: false,
  difficulty: "medium",
  gameOver: false,
  lastMove: null,
  result: null,
  lastGuardrailWeight: null,
  checkmate: null,
};

function isTwoPlayerMode(): boolean {
  return state.twoPlayer;
}

// =============================================================================
// Stats Tracking
// =============================================================================

interface Stats {
  won: number;
  lost: number;
}

const STATS_KEY = "ziczaczoe_stats";

function loadStats(): Stats {
  try {
    const saved = localStorage.getItem(STATS_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    // Ignore errors
  }
  return { won: 0, lost: 0 };
}

function saveStats(stats: Stats): void {
  try {
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
  } catch (e) {
    // Ignore errors
  }
}

const stats: Stats = loadStats();

function updateStatsDisplay(): void {
  const wonEl = document.getElementById("stats-won")!;
  const lostEl = document.getElementById("stats-lost")!;
  wonEl.textContent = `WON: ${String(stats.won).padStart(3, "0")}`;
  lostEl.textContent = `LOST: ${String(stats.lost).padStart(3, "0")}`;
}

function clearStatsPulsing(): void {
  document.getElementById("stats-won")?.classList.remove("pulsing");
  document.getElementById("stats-lost")?.classList.remove("pulsing");
}

function updateGuardrailDisplay(): void {
  const guardrailEl = document.getElementById("guardrail-weight");
  if (!guardrailEl) return;

  if (state.lastGuardrailWeight === null) {
    guardrailEl.textContent = "";
  } else {
    guardrailEl.textContent = `GR ${state.lastGuardrailWeight.toFixed(2)}`;
  }
}

function recordGameResult(result: GameResult, humanPlayer: Player): void {
  // Don't track ties - just win/loss
  if (result === GameResult.Draw) {
    return;
  }
  const winner = result === GameResult.XWins ? Player.X : Player.O;
  const humanWon = winner === humanPlayer;
  if (humanWon) {
    stats.won++;
  } else {
    stats.lost++;
  }
  saveStats(stats);
  updateStatsDisplay();

  // Pulse the counter that changed
  const elementId = humanWon ? "stats-won" : "stats-lost";
  document.getElementById(elementId)?.classList.add("pulsing");
}

// =============================================================================
// DOM Elements
// =============================================================================

const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const loadingEl = document.getElementById("loading")!;
const statsEl = document.querySelector(".stats") as HTMLElement;
const btnMode = document.getElementById("btn-mode")!;
const btnPlayer = document.getElementById("btn-player")!;
const btnDifficulty = document.getElementById("btn-difficulty")!;
const btnNewGame = document.getElementById("btn-new-game")!;

// Mode options (1-player vs 2-player)
const MODE_OPTIONS: { value: boolean; label: string }[] = [
  { value: false, label: "1-PLAYER" },
  { value: true, label: "2-PLAYER" },
];

// Player options for cycling
const PLAYER_OPTIONS: { value: Player; label: string }[] = [
  { value: Player.X, label: "X (1ST)" },
  { value: Player.O, label: "O (2ND)" },
];

// Difficulty options for cycling
const DIFFICULTY_OPTIONS: { value: Difficulty; label: string }[] = [
  { value: "easy", label: "EASY" },
  { value: "medium", label: "MEDIUM" },
  { value: "hard", label: "HARD" },
  { value: "expert", label: "EXPERT" },
];

// =============================================================================
// Rendering
// =============================================================================

/**
 * Create the board cells
 */
function createBoardCells(): void {
  boardEl.innerHTML = "";
  for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.index = i.toString();
    cell.addEventListener("click", () => handleCellClick(i));
    boardEl.appendChild(cell);
  }
}

/**
 * Render the current board state
 */
function renderBoard(): void {
  const cells = boardEl.querySelectorAll(".cell");

  // Pre-compute checkmate highlight info (avoid recomputing per cell)
  let checkmateHighlightClass: string | null = null;
  let checkmateIndices: Set<number> | null = null;
  if (state.checkmate?.isCheckmate) {
    // In 2-player mode, use winner's color; in vs AI mode, use win/lose
    if (isTwoPlayerMode()) {
      const winner = state.checkmate.loser === Player.X ? Player.O : Player.X;
      checkmateHighlightClass = winner === Player.X ? "checkmate-win-highlight" : "checkmate-lose-highlight";
    } else {
      const humanWins = state.checkmate.loser !== state.humanPlayer;
      checkmateHighlightClass = humanWins ? "checkmate-win-highlight" : "checkmate-lose-highlight";
    }
    // Flatten all pattern indices into a Set for O(1) lookup
    checkmateIndices = new Set(
      [...state.checkmate.threatPatterns, ...state.checkmate.suicidePatterns].flat()
    );
  }

  cells.forEach((cell, i) => {
    const el = cell as HTMLElement;
    const piece = state.board[i];

    // Clear classes
    el.classList.remove(
      "occupied",
      "last-move",
      "win-highlight",
      "lose-highlight",
      "checkmate-win-highlight",
      "checkmate-lose-highlight",
      "game-over"
    );

    // Set piece content
    if (piece === Player.Empty) {
      el.innerHTML = "";
    } else {
      const symbol = piece === Player.X ? "X" : "O";
      const pieceClass = piece === Player.X ? "x" : "o";
      el.innerHTML = `<span class="piece ${pieceClass}">${symbol}</span>`;
      el.classList.add("occupied");
    }

    // Highlight last move
    if (i === state.lastMove) {
      el.classList.add("last-move");
    }

    // Game over state
    if (state.gameOver) {
      el.classList.add("game-over");
    }

    // Win/lose highlights
    if (state.result) {
      if (state.result.winningIndices.includes(i)) {
        el.classList.add("win-highlight");
      }
      if (state.result.losingIndices.includes(i)) {
        el.classList.add("lose-highlight");
      }
    }

    // Checkmate highlights (O(1) lookup via Set)
    if (checkmateHighlightClass && checkmateIndices?.has(i)) {
      el.classList.add(checkmateHighlightClass);
    }
  });
}

/**
 * Update the status display
 */
function updateStatus(): void {
  statusEl.classList.remove("thinking", "your-turn", "win", "lose", "draw", "x-wins", "o-wins");

  if (state.gameOver && state.result) {
    const { result } = state.result;

    if (result === GameResult.Draw) {
      statusEl.textContent = "TIE!";
      statusEl.classList.add("draw");
    } else {
      // Determine winner
      const winner = result === GameResult.XWins ? Player.X : Player.O;

      if (isTwoPlayerMode()) {
        // 2-player mode: show "X WINS!" or "O WINS!" in winner's color
        const winClass = winner === Player.X ? "x-wins" : "o-wins";
        if (state.checkmate?.isCheckmate) {
          statusEl.textContent = "CHECKMATE!!";
        } else {
          statusEl.textContent = winner === Player.X ? "X WINS!" : "O WINS!";
        }
        statusEl.classList.add(winClass);
      } else {
        // VS AI mode: show win/lose from human's perspective
        const humanWins = winner === state.humanPlayer;
        if (state.checkmate?.isCheckmate) {
          statusEl.textContent = "CHECKMATE!!";
          statusEl.classList.add(humanWins ? "win" : "lose");
        } else if (humanWins) {
          statusEl.textContent = "YOU WIN!!!";
          statusEl.classList.add("win");
        } else {
          statusEl.textContent = "GAME OVER";
          statusEl.classList.add("lose");
        }
      }
    }
  } else {
    const currentPlayer = getCurrentPlayer(state.board);
    if (isTwoPlayerMode()) {
      // 2-player mode: show whose turn it is
      statusEl.textContent = currentPlayer === Player.X ? "X PLAYS" : "O PLAYS";
      statusEl.classList.add("your-turn");
    } else if (currentPlayer === state.humanPlayer) {
      statusEl.textContent = "YOUR TURN";
      statusEl.classList.add("your-turn");
    } else {
      statusEl.textContent = "";
    }
  }
}

/**
 * Update button text to reflect current state
 */
function updateButtons(): void {
  // Mode button
  const modeOption = MODE_OPTIONS.find(o => o.value === state.twoPlayer);
  btnMode.textContent = modeOption?.label ?? "1-PLAYER";

  // Player button - disabled in 2-player mode
  const playerOption = PLAYER_OPTIONS.find(o => o.value === state.humanPlayer);
  btnPlayer.textContent = playerOption?.label ?? "X (1ST)";
  btnPlayer.classList.toggle("disabled", isTwoPlayerMode());

  // Difficulty button - disabled in 2-player mode
  const diffOption = DIFFICULTY_OPTIONS.find(o => o.value === state.difficulty);
  btnDifficulty.textContent = diffOption?.label ?? "MEDIUM";
  btnDifficulty.classList.toggle("disabled", isTwoPlayerMode());

  // Stats visibility - hidden in 2-player mode
  statsEl.style.visibility = isTwoPlayerMode() ? "hidden" : "visible";
}

// =============================================================================
// Game Logic
// =============================================================================

/**
 * Start a new game
 */
function newGame(): void {
  state.board = createBoard();
  state.gameOver = false;
  state.lastMove = null;
  state.result = null;
  state.lastGuardrailWeight = null;
  state.checkmate = null;

  clearStatsPulsing();
  renderBoard();
  updateStatus();
  updateGuardrailDisplay();

  // If AI goes first (and we're not in 2-player mode), make AI move
  if (!isTwoPlayerMode() && state.humanPlayer === Player.O) {
    setTimeout(() => makeAIMove(), AI_FIRST_MOVE_DELAY);
  }
}

/**
 * Handle cell click
 */
function handleCellClick(index: number): void {
  // Ignore if game over
  if (state.gameOver) return;

  // In 2-player mode, either player can go; in vs AI mode, only human's turn
  const currentPlayer = getCurrentPlayer(state.board);
  if (!isTwoPlayerMode() && currentPlayer !== state.humanPlayer) return;

  // Ignore if cell is occupied
  if (state.board[index] !== Player.Empty) return;

  // Make the move
  makeHumanMove(index);
}

/**
 * Make a human move (handles both vs AI and 2-player modes)
 */
function makeHumanMove(index: number): void {
  const currentPlayer = getCurrentPlayer(state.board);
  state.board = makeMove(state.board, index);
  state.lastMove = index;

  // Check for game end
  const result = checkResultFast(state.board, index);
  if (result.result !== GameResult.Ongoing) {
    state.gameOver = true;
    state.result = result;
    if (!isTwoPlayerMode()) {
      recordGameResult(result.result, state.humanPlayer);
    }
    renderBoard();
    updateStatus();
    return;
  }

  // Check if opponent is now checkmated (before they move)
  const opponent = currentPlayer === Player.X ? Player.O : Player.X;
  const checkmate = detectCheckmate(state.board, opponent);
  if (checkmate.isCheckmate) {
    state.gameOver = true;
    state.checkmate = checkmate;
    // Opponent is checkmated, so current player wins
    const winResult = currentPlayer === Player.X ? GameResult.XWins : GameResult.OWins;
    state.result = {
      result: winResult,
      winningIndices: [],
      losingIndices: [],
      losingPlayer: opponent,
    };
    if (!isTwoPlayerMode()) {
      recordGameResult(winResult, state.humanPlayer);
    }
    renderBoard();
    updateStatus();
    return;
  }

  renderBoard();
  updateStatus();

  // In vs AI mode, trigger AI's turn
  if (!isTwoPlayerMode()) {
    setTimeout(() => makeAIMove(), AI_MOVE_DELAY);
  }
}

/**
 * Make an AI move
 */
async function makeAIMove(): Promise<void> {
  if (state.gameOver) return;

  // Sanity check
  const legalMoves = getLegalMoves(state.board);
  if (legalMoves.length === 0) return;

  // Use rules-based AI for v3, neural network otherwise
  let move: number;
  let guardrailWeight = 0;

  try {
    if (isRulesAI()) {
      move = getRulesMove(state.board, getCurrentPlayer(state.board));
      guardrailWeight = 1.0; // Rules AI always uses guardrails
    } else {
      const result = await getAIMove(state.board, state.difficulty);
      move = result.move;
      guardrailWeight = result.guardrailWeight;
    }
  } catch (error) {
    // Fallback to random move - never show error to user
    console.error("AI move calculation failed, using random:", error);
    move = legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }

  state.board = makeMove(state.board, move);
  state.lastMove = move;
  state.lastGuardrailWeight = guardrailWeight;
  updateGuardrailDisplay();

  // Check for game end
  const result = checkResultFast(state.board, move);
  if (result.result !== GameResult.Ongoing) {
    state.gameOver = true;
    state.result = result;
    recordGameResult(result.result, state.humanPlayer);
    renderBoard();
    updateStatus();
    return;
  }

  // Check if human is now checkmated (before they even move)
  const checkmate = detectCheckmate(state.board, state.humanPlayer);
  if (checkmate.isCheckmate) {
    state.gameOver = true;
    state.checkmate = checkmate;
    // Human is checkmated, so AI wins
    const aiWinResult = state.humanPlayer === Player.X ? GameResult.OWins : GameResult.XWins;
    state.result = {
      result: aiWinResult,
      winningIndices: [],
      losingIndices: [],
      losingPlayer: state.humanPlayer,
    };
    recordGameResult(aiWinResult, state.humanPlayer);
  }

  renderBoard();
  updateStatus();
}

// =============================================================================
// Event Handlers
// =============================================================================

function setupEventListeners(): void {
  // Mode selection - toggles between 1-player and 2-player and starts new game
  btnMode.addEventListener("click", () => {
    const currentIndex = MODE_OPTIONS.findIndex(o => o.value === state.twoPlayer);
    const nextIndex = (currentIndex + 1) % MODE_OPTIONS.length;
    state.twoPlayer = MODE_OPTIONS[nextIndex].value;
    updateButtons();
    newGame();
  });

  // Player selection - cycles through options and starts new game
  // (disabled in 2-player mode via CSS pointer-events)
  btnPlayer.addEventListener("click", () => {
    if (isTwoPlayerMode()) return; // Extra safety check
    const currentIndex = PLAYER_OPTIONS.findIndex(o => o.value === state.humanPlayer);
    const nextIndex = (currentIndex + 1) % PLAYER_OPTIONS.length;
    state.humanPlayer = PLAYER_OPTIONS[nextIndex].value;
    updateButtons();
    newGame();
  });

  // Difficulty selection - cycles through options and starts new game
  // (disabled in 2-player mode via CSS pointer-events)
  btnDifficulty.addEventListener("click", () => {
    if (isTwoPlayerMode()) return; // Extra safety check
    const currentIndex = DIFFICULTY_OPTIONS.findIndex(o => o.value === state.difficulty);
    const nextIndex = (currentIndex + 1) % DIFFICULTY_OPTIONS.length;
    state.difficulty = DIFFICULTY_OPTIONS[nextIndex].value;
    updateButtons();
    newGame();
  });

  // New game
  btnNewGame.addEventListener("click", () => {
    newGame();
  });
}

// =============================================================================
// Scaling
// =============================================================================

/**
 * Calculate and apply scale to fit app in viewport.
 * Uses portrait dimensions as base, centers in landscape.
 */
function updateScale(): void {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Always scale based on portrait dimensions
  const scale = Math.min(vw / DESIGN_WIDTH, vh / DESIGN_HEIGHT);

  // Apply scale transform
  const appEl = document.getElementById("app")!;
  appEl.style.transform = `scale(${scale})`;
  appEl.style.transformOrigin = "top center";

  // Set explicit dimensions so transform works correctly
  appEl.style.width = `${DESIGN_WIDTH}px`;
  appEl.style.height = `${DESIGN_HEIGHT}px`;
}

// =============================================================================
// Initialization
// =============================================================================

async function init(): Promise<void> {
  // Set up scaling
  updateScale();
  window.addEventListener("resize", updateScale);
  window.addEventListener("orientationchange", () => {
    // Small delay to let orientation change complete
    setTimeout(updateScale, 100);
  });

  // Create board cells
  createBoardCells();

  // Set up event listeners
  setupEventListeners();

  // Update initial button states
  updateButtons();

  // Show stats
  updateStatsDisplay();

  const iterationEl = document.getElementById("model-iteration");

  if (isRulesAI()) {
    // Rules-based AI (no model needed) - enabled via ?rules=1
    loadingEl.classList.add("hidden");
    if (iterationEl) {
      iterationEl.textContent = "RULES";
    }
    newGame();
  } else {
    // Load neural network model
    try {
      const weightsPath = import.meta.env.DEV ? "/weights.json" : "./weights.json";
      await loadModel(weightsPath);
      loadingEl.classList.add("hidden");

      // Display model iteration
      const iteration = getModelIteration();
      if (iterationEl && iteration !== undefined) {
        iterationEl.textContent = `ITER ${iteration}`;
      }

      // Start the game
      newGame();
    } catch (error) {
      console.error("Failed to load model:", error);
      const loadingText = loadingEl.querySelector(".loading-text")!;
      loadingText.textContent = "FAILED TO LOAD AI";
      if (error instanceof Error) {
        console.error("Error details:", error.message);
      }
    }
  }
}

// Start the app
init();
