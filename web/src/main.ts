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
import { getRulesMove, isRulesAI } from "./rules-ai";

// Timing constants (milliseconds)
const AI_MOVE_DELAY = 500;      // Delay after human move before AI responds
const AI_FIRST_MOVE_DELAY = 500; // Delay when AI goes first

// =============================================================================
// Game State
// =============================================================================

interface GameState {
  board: BoardState;
  humanPlayer: Player;
  difficulty: Difficulty;
  gameOver: boolean;
  lastMove: number | null;
  result: GameCheckResult | null;
}

const state: GameState = {
  board: createBoard(),
  humanPlayer: Player.X,
  difficulty: "medium",
  gameOver: false,
  lastMove: null,
  result: null,
};

// =============================================================================
// DOM Elements
// =============================================================================

const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const loadingEl = document.getElementById("loading")!;
const btnPlayer = document.getElementById("btn-player")!;
const btnDifficulty = document.getElementById("btn-difficulty")!;
const btnNewGame = document.getElementById("btn-new-game")!;

// Player options for cycling
const PLAYER_OPTIONS: { value: Player; label: string }[] = [
  { value: Player.X, label: "1ST (X)" },
  { value: Player.O, label: "2ND (O)" },
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

  cells.forEach((cell, i) => {
    const el = cell as HTMLElement;
    const piece = state.board[i];

    // Clear classes
    el.classList.remove(
      "occupied",
      "last-move",
      "win-highlight",
      "lose-highlight",
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
  });
}

/**
 * Update the status display
 */
function updateStatus(): void {
  statusEl.classList.remove("thinking", "your-turn", "win", "lose");

  if (state.gameOver && state.result) {
    const { result, losingPlayer } = state.result;

    if (result === GameResult.Draw) {
      statusEl.textContent = "DRAW";
      statusEl.classList.add("your-turn");
    } else {
      // Determine winner
      const winner = result === GameResult.XWins ? Player.X : Player.O;
      const humanWins = winner === state.humanPlayer;

      if (losingPlayer !== null) {
        // Someone lost by making 3 in a row
        if (losingPlayer === state.humanPlayer) {
          statusEl.textContent = "YOU LOSE!";
          statusEl.classList.add("lose");
        } else {
          statusEl.textContent = "I LOSE!!!";
          statusEl.classList.add("win");
        }
      } else {
        // Someone won by making 4 in a row
        if (humanWins) {
          statusEl.textContent = "YOU WIN!!!";
          statusEl.classList.add("win");
        } else {
          statusEl.textContent = "I WIN!";
          statusEl.classList.add("lose");
        }
      }
    }
  } else {
    const currentPlayer = getCurrentPlayer(state.board);
    if (currentPlayer === state.humanPlayer) {
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
  // Player button
  const playerOption = PLAYER_OPTIONS.find(o => o.value === state.humanPlayer);
  btnPlayer.textContent = playerOption?.label ?? "1ST (X)";

  // Difficulty button
  const diffOption = DIFFICULTY_OPTIONS.find(o => o.value === state.difficulty);
  btnDifficulty.textContent = diffOption?.label ?? "MEDIUM";
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

  renderBoard();
  updateStatus();

  // If AI goes first, make AI move
  if (state.humanPlayer === Player.O) {
    setTimeout(() => makeAIMove(), AI_FIRST_MOVE_DELAY);
  }
}

/**
 * Handle cell click
 */
function handleCellClick(index: number): void {
  // Ignore if game over
  if (state.gameOver) return;

  // Ignore if not human's turn
  const currentPlayer = getCurrentPlayer(state.board);
  if (currentPlayer !== state.humanPlayer) return;

  // Ignore if cell is occupied
  if (state.board[index] !== Player.Empty) return;

  // Make the move
  makeHumanMove(index);
}

/**
 * Make a human move
 */
function makeHumanMove(index: number): void {
  state.board = makeMove(state.board, index);
  state.lastMove = index;

  // Check for game end
  const result = checkResultFast(state.board, index);
  if (result.result !== GameResult.Ongoing) {
    state.gameOver = true;
    state.result = result;
    renderBoard();
    updateStatus();
    return;
  }

  renderBoard();
  updateStatus();

  // AI's turn - add small delay for arcade feel
  setTimeout(() => makeAIMove(), AI_MOVE_DELAY);
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
  const currentPlayer = getCurrentPlayer(state.board);
  let move: number;

  try {
    move = isRulesAI()
      ? getRulesMove(state.board, currentPlayer)
      : await getAIMove(state.board, state.difficulty);
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
  }

  renderBoard();
  updateStatus();
}

// =============================================================================
// Event Handlers
// =============================================================================

function setupEventListeners(): void {
  // Player selection - cycles through options and starts new game
  btnPlayer.addEventListener("click", () => {
    const currentIndex = PLAYER_OPTIONS.findIndex(o => o.value === state.humanPlayer);
    const nextIndex = (currentIndex + 1) % PLAYER_OPTIONS.length;
    state.humanPlayer = PLAYER_OPTIONS[nextIndex].value;
    updateButtons();
    newGame();
  });

  // Difficulty selection - cycles through options
  btnDifficulty.addEventListener("click", () => {
    const currentIndex = DIFFICULTY_OPTIONS.findIndex(o => o.value === state.difficulty);
    const nextIndex = (currentIndex + 1) % DIFFICULTY_OPTIONS.length;
    state.difficulty = DIFFICULTY_OPTIONS[nextIndex].value;
    updateButtons();
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
