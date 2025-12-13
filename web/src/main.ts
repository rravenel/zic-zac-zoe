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
import { loadModel, getAIMove, Difficulty } from "./ai";
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

interface TwoPlayerStats {
  x: number;
  o: number;
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

// 2-player stats (not persisted - resets on mode switch)
let twoPlayerStats: TwoPlayerStats = { x: 0, o: 0 };

function updateStatsDisplay(): void {
  const wonEl = document.getElementById("stats-won")!;
  const lostEl = document.getElementById("stats-lost")!;

  if (isTwoPlayerMode()) {
    // 2-player mode: show X/O scores with token colors
    wonEl.textContent = `X: ${String(twoPlayerStats.x).padStart(3, "0")}`;
    lostEl.textContent = `O: ${String(twoPlayerStats.o).padStart(3, "0")}`;
    wonEl.classList.add("x-score");
    wonEl.classList.remove("blinking");
    lostEl.classList.add("o-score");
    lostEl.classList.remove("blinking");
  } else {
    // 1-player mode: show WON/LOST
    wonEl.textContent = `WON: ${String(stats.won).padStart(3, "0")}`;
    lostEl.textContent = `LOST: ${String(stats.lost).padStart(3, "0")}`;
    wonEl.classList.remove("x-score");
    lostEl.classList.remove("o-score");
  }
}

function clearStatsBlinking(): void {
  document.getElementById("stats-won")?.classList.remove("blinking");
  document.getElementById("stats-lost")?.classList.remove("blinking");
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

  // Blink the counter that changed
  const elementId = humanWon ? "stats-won" : "stats-lost";
  document.getElementById(elementId)?.classList.add("blinking");
}

function recordTwoPlayerResult(result: GameResult): void {
  // Don't track ties
  if (result === GameResult.Draw) {
    return;
  }
  const winner = result === GameResult.XWins ? Player.X : Player.O;
  if (winner === Player.X) {
    twoPlayerStats.x++;
  } else {
    twoPlayerStats.o++;
  }
  updateStatsDisplay();

  // Blink the winner's counter
  const elementId = winner === Player.X ? "stats-won" : "stats-lost";
  document.getElementById(elementId)?.classList.add("blinking");
}

// =============================================================================
// DOM Elements
// =============================================================================

const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const loadingEl = document.getElementById("loading")!;
const btnMode = document.getElementById("btn-mode")!;
const btnPlayer = document.getElementById("btn-player")!;
const btnDifficulty = document.getElementById("btn-difficulty")!;
const btnNewGame = document.getElementById("btn-new-game")!;

// Difficulty levels for cycling (1-4 stars)
const DIFFICULTY_LEVELS: Difficulty[] = ["easy", "medium", "hard", "expert"];

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

  // Update board's turn class for hover styling in 2P mode
  const currentPlayer = getCurrentPlayer(state.board);
  boardEl.classList.toggle("o-turn", isTwoPlayerMode() && currentPlayer === Player.O);

  // Pre-compute game end highlight info
  let endHighlightClass: string | null = null;
  let endIndices: Set<number> | null = null;
  let cruxIndices: Set<number> | null = null;

  if (state.gameOver && state.result) {
    const { result } = state.result;
    if (result !== GameResult.Draw) {
      const winner = result === GameResult.XWins ? Player.X : Player.O;

      // Determine highlight class based on mode
      if (isTwoPlayerMode()) {
        // 2P: winner's color
        endHighlightClass = winner === Player.X ? "end-x" : "end-o";
      } else {
        // 1P: green if human wins, red if AI wins
        const humanWins = winner === state.humanPlayer;
        endHighlightClass = humanWins ? "end-green" : "end-red";
      }

      // Collect indices that ended the game
      endIndices = new Set([
        ...state.result.winningIndices,
        ...state.result.losingIndices,
      ]);
    }
  }

  // For checkmate, find crux cells (empty cells in the threat/suicide patterns)
  if (state.checkmate?.isCheckmate) {
    const winner = state.checkmate.loser === Player.X ? Player.O : Player.X;

    // Determine highlight class for checkmate patterns
    if (isTwoPlayerMode()) {
      endHighlightClass = winner === Player.X ? "end-x" : "end-o";
    } else {
      const humanWins = state.checkmate.loser !== state.humanPlayer;
      endHighlightClass = humanWins ? "end-green" : "end-red";
    }

    // Collect all cells in checkmate patterns
    endIndices = new Set(
      [...state.checkmate.threatPatterns, ...state.checkmate.suicidePatterns].flat()
    );

    // Find crux cells - empty cells that would complete patterns
    cruxIndices = new Set<number>();
    for (const pattern of [...state.checkmate.threatPatterns, ...state.checkmate.suicidePatterns]) {
      for (const idx of pattern) {
        if (state.board[idx] === Player.Empty) {
          cruxIndices.add(idx);
        }
      }
    }
  }

  cells.forEach((cell, i) => {
    const el = cell as HTMLElement;
    const piece = state.board[i];

    // Clear classes
    el.classList.remove(
      "occupied",
      "last-move-x",
      "last-move-o",
      "end-green",
      "end-red",
      "end-x",
      "end-o",
      "crux-blink",
      "game-over"
    );

    // Set piece content
    if (piece === Player.Empty) {
      el.innerHTML = "";
    } else {
      const symbol = piece === Player.X ? "X" : "O";
      // In 1P mode: user = cyan (class "x"), AI = yellow (class "o")
      // In 2P mode: X = cyan, O = yellow (standard)
      let pieceClass: string;
      if (isTwoPlayerMode()) {
        pieceClass = piece === Player.X ? "x" : "o";
      } else {
        // User's token is always cyan, AI's is always yellow
        const isUserPiece = piece === state.humanPlayer;
        pieceClass = isUserPiece ? "x" : "o";
      }
      el.innerHTML = `<span class="piece ${pieceClass}">${symbol}</span>`;
      el.classList.add("occupied");
    }

    // Highlight last move with player's color (only during game, not at end)
    if (i === state.lastMove && piece !== Player.Empty && !state.gameOver) {
      el.classList.add(piece === Player.X ? "last-move-x" : "last-move-o");
    }

    // Game over state
    if (state.gameOver) {
      el.classList.add("game-over");

      // Apply end highlight to cells that ended the game
      if (endHighlightClass && endIndices?.has(i)) {
        el.classList.add(endHighlightClass);

        // Blink crux cells and the last move cell
        if (cruxIndices?.has(i) || i === state.lastMove) {
          el.classList.add("crux-blink");
        }
      }
    }
  });
}

/**
 * Update the status display
 */
function updateStatus(): void {
  statusEl.classList.remove("your-turn", "x-turn", "o-turn", "win", "lose", "draw", "x-wins", "o-wins", "hidden");

  if (state.gameOver && state.result) {
    const { result } = state.result;

    if (result === GameResult.Draw) {
      statusEl.textContent = "TIE GAME";
      statusEl.classList.add("draw");
    } else {
      // Determine winner
      const winner = result === GameResult.XWins ? Player.X : Player.O;

      if (isTwoPlayerMode()) {
        // 2-player mode: show "X WINS!!" or "O WINS!!" in winner's color
        const winClass = winner === Player.X ? "x-wins" : "o-wins";
        if (state.checkmate?.isCheckmate) {
          statusEl.textContent = "CHECKMATE!!";
        } else {
          statusEl.textContent = winner === Player.X ? "X WINS!!" : "O WINS!!";
        }
        statusEl.classList.add(winClass);
      } else {
        // VS AI mode: show win/lose from human's perspective
        const humanWins = winner === state.humanPlayer;
        if (state.checkmate?.isCheckmate) {
          statusEl.textContent = "CHECKMATE!!";
          statusEl.classList.add(humanWins ? "win" : "lose");
        } else if (humanWins) {
          statusEl.textContent = "YOU WIN!!";
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
      // 2-player mode: show whose turn in player's color
      statusEl.textContent = currentPlayer === Player.X ? "X PLAYS" : "O PLAYS";
      statusEl.classList.add(currentPlayer === Player.X ? "x-turn" : "o-turn");
    } else if (currentPlayer === state.humanPlayer) {
      statusEl.textContent = "YOUR TURN";
      statusEl.classList.add("your-turn");
    } else {
      // Hide status completely when AI is thinking (avoids repaint artifacts)
      statusEl.classList.add("hidden");
    }
  }
}

// Difficulty level display names
const DIFFICULTY_NAMES: Record<Difficulty, string> = {
  easy: "EASY",
  medium: "MED",
  hard: "HARD",
  expert: "XPRT",
};

/**
 * Update button visuals to reflect current state
 */
function updateButtons(): void {
  // Mode button - shows "1P" or "2P"
  btnMode.textContent = state.twoPlayer ? "2P" : "1P";

  // Player button - shows "X" or "O", disabled in 2-player mode
  btnPlayer.textContent = state.humanPlayer === Player.X ? "X" : "O";
  btnPlayer.classList.toggle("disabled", isTwoPlayerMode());

  // Difficulty button - shows difficulty name, disabled in 2-player mode
  btnDifficulty.textContent = DIFFICULTY_NAMES[state.difficulty];
  btnDifficulty.classList.toggle("disabled", isTwoPlayerMode());
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
  state.checkmate = null;

  clearStatsBlinking();
  renderBoard();
  updateStatus();

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
    if (isTwoPlayerMode()) {
      recordTwoPlayerResult(result.result);
    } else {
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
    if (isTwoPlayerMode()) {
      recordTwoPlayerResult(winResult);
    } else {
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

  try {
    if (isRulesAI()) {
      move = getRulesMove(state.board, getCurrentPlayer(state.board));
    } else {
      const result = await getAIMove(state.board, state.difficulty);
      move = result.move;
    }
  } catch (error) {
    // Fallback to random move - never show error to user
    console.error("AI move calculation failed, using random:", error);
    move = legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }

  state.board = makeMove(state.board, move);
  state.lastMove = move;

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
  // Mode selection - toggles between 1-player and 2-player
  btnMode.addEventListener("click", () => {
    state.twoPlayer = !state.twoPlayer;
    // Reset 2-player stats when entering 2-player mode
    if (state.twoPlayer) {
      twoPlayerStats = { x: 0, o: 0 };
    }
    updateButtons();
    updateStatsDisplay();
    newGame();
  });

  // Player selection - toggles X/O (disabled in 2-player mode)
  btnPlayer.addEventListener("click", () => {
    if (isTwoPlayerMode()) return;
    state.humanPlayer = state.humanPlayer === Player.X ? Player.O : Player.X;
    updateButtons();
    newGame();
  });

  // Difficulty selection - cycles through 1-4 stars (disabled in 2-player mode)
  btnDifficulty.addEventListener("click", () => {
    if (isTwoPlayerMode()) return;
    const currentIndex = DIFFICULTY_LEVELS.indexOf(state.difficulty);
    const nextIndex = (currentIndex + 1) % DIFFICULTY_LEVELS.length;
    state.difficulty = DIFFICULTY_LEVELS[nextIndex];
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

  if (isRulesAI()) {
    // Rules-based AI (no model needed) - enabled via ?rules=1
    loadingEl.classList.add("hidden");
    newGame();
  } else {
    // Load neural network model
    try {
      const weightsPath = import.meta.env.DEV ? "/weights.json" : "./weights.json";
      await loadModel(weightsPath);
      loadingEl.classList.add("hidden");
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
