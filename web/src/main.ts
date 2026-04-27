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
  calculateMoveScore,
  getScoringIndices,
} from "./game";
import { loadModel, getAIMove, Difficulty } from "./ai";
import { getRulesMove, isRulesAI } from "./rules-ai";

// Timing constants (milliseconds)
const AI_MOVE_DELAY = 500;      // Delay after human move before AI responds
const AI_FIRST_MOVE_DELAY = 500; // Delay when AI goes first

// =============================================================================
// Configuration
// =============================================================================

interface ConfigMode {
  id: string;
  active: boolean;
  settings: Record<string, number>;
}

interface GameConfig {
  modes: ConfigMode[];
}

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
  lastMoveX: number | null;
  lastMoveO: number | null;
  result: GameCheckResult | null;
  playerXScore: number;
  playerOScore: number;
  lastMoveScoreX: number | null;
  lastMoveScoreO: number | null;
  scoringHighlightsX: number[];
  scoringHighlightsO: number[];
  // Claim Variant additions
  pendingMove: number | null;
  awaitingDecision: boolean;
  movesRemainingX: number;
  movesRemainingO: number;
  activeModeId: string;
  modeSettings: Record<string, number>;
  configError: string | null;
}

const state: GameState = {
  board: createBoard(),
  humanPlayer: Player.X,
  twoPlayer: false,
  difficulty: "medium",
  gameOver: false,
  lastMove: null,
  lastMoveX: null,
  lastMoveO: null,
  result: null,
  playerXScore: 0,
  playerOScore: 0,
  lastMoveScoreX: null,
  lastMoveScoreO: null,
  scoringHighlightsX: [],
  scoringHighlightsO: [],
  // Claim Variant defaults
  pendingMove: null,
  awaitingDecision: false,
  movesRemainingX: 0,
  movesRemainingO: 0,
  activeModeId: "",
  modeSettings: {},
  configError: null,
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

function recordGameResult(_result: GameResult, humanPlayer: Player): void {
  // Compare scores to determine winner
  if (state.playerXScore === state.playerOScore) {
    return; // Tie - no change to stats
  }

  const humanScore = humanPlayer === Player.X ? state.playerXScore : state.playerOScore;
  const aiScore = humanPlayer === Player.X ? state.playerOScore : state.playerXScore;
  
  const humanWon = humanScore > aiScore;
  
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

function recordTwoPlayerResult(_result: GameResult): void {
  // Compare scores to determine winner
  if (state.playerXScore === state.playerOScore) {
    return; // Tie
  }

  const winner = state.playerXScore > state.playerOScore ? Player.X : Player.O;
  
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
const scoreXEl = document.getElementById("score-x")!;
const scoreOEl = document.getElementById("score-o")!;
const lastPointsXEl = document.getElementById("last-points-x")!;
const lastPointsOEl = document.getElementById("last-points-o")!;

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
      "game-over",
      "scoring-highlight-x",
      "scoring-highlight-o",
      "last-move-blink-x",
      "last-move-blink-o"
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

    // Scoring highlights
    if (state.scoringHighlightsX.includes(i)) {
      el.classList.add("scoring-highlight-x");
    }
    if (state.scoringHighlightsO.includes(i)) {
      el.classList.add("scoring-highlight-o");
    }

    // Last move blinks
    if (state.lastMoveX === i) {
      el.classList.add("last-move-blink-x");
    }
    if (state.lastMoveO === i) {
      el.classList.add("last-move-blink-o");
    }

    // Game over state
    if (state.gameOver) {
      el.classList.add("game-over");
    }
  });
}

/**
 * Update the scoreboard display with current scores
 */
function updateScoreboardDisplay(): void {
  scoreXEl.textContent = state.playerXScore.toString().padStart(5, " ");
  scoreOEl.textContent = state.playerOScore.toString().padStart(5, " ");

  // Update last points indicators
  if (state.lastMoveScoreX !== null) {
    lastPointsXEl.textContent = `+${state.lastMoveScoreX}`;
    lastPointsXEl.classList.add("visible");
  } else {
    lastPointsXEl.classList.remove("visible");
  }

  if (state.lastMoveScoreO !== null) {
    lastPointsOEl.textContent = `+${state.lastMoveScoreO}`;
    lastPointsOEl.classList.add("visible");
  } else {
    lastPointsOEl.classList.remove("visible");
  }
}

/**
 * Update the status display
 */
function updateStatus(): void {
  statusEl.classList.remove("win", "lose", "draw", "x-wins", "o-wins", "game-over-visible");

  if (state.gameOver) {
    statusEl.classList.add("game-over-visible");
    // Determine winner based on scores
    if (state.playerXScore === state.playerOScore) {
      statusEl.textContent = "TIE GAME";
      statusEl.classList.add("draw");
    } else {
      const winner = state.playerXScore > state.playerOScore ? Player.X : Player.O;

      if (isTwoPlayerMode()) {
        // 2-player mode: show "X WINS!!" or "O WINS!!" in winner's color
        const winClass = winner === Player.X ? "x-wins" : "o-wins";
        statusEl.textContent = winner === Player.X ? "X WINS!!" : "O WINS!!";
        statusEl.classList.add(winClass);
      } else {
        // VS AI mode: show win/lose from human's perspective
        const humanWins = winner === state.humanPlayer;
        if (humanWins) {
          statusEl.textContent = "YOU WIN!!";
          statusEl.classList.add("win");
        } else {
          statusEl.textContent = "GAME OVER";
          statusEl.classList.add("lose");
        }
      }
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
  state.playerXScore = 0;
  state.playerOScore = 0;
  state.lastMoveScoreX = null;
  state.lastMoveScoreO = null;
  state.lastMoveX = null;
  state.lastMoveO = null;
  state.scoringHighlightsX = [];
  state.scoringHighlightsO = [];

  // Claim Variant initialization
  state.pendingMove = null;
  state.awaitingDecision = false;
  state.movesRemainingX = state.modeSettings.limit_per_side || 0;
  state.movesRemainingO = state.modeSettings.limit_per_side || 0;

  clearStatsBlinking();
  renderBoard();
  updateStatus();
  updateScoreboardDisplay();

  // If AI goes first (and we're not in 2-player mode), make AI move
  if (!isTwoPlayerMode() && state.humanPlayer === Player.O) {
    setTimeout(() => makeAIMove(), AI_FIRST_MOVE_DELAY);
  }
}

/**
 * Handle cell click
 */
function handleCellClick(index: number): void {
  // Ignore if game over or config error
  if (state.gameOver || state.configError) return;

  // Ignore if awaiting decision
  if (state.awaitingDecision) return;

  // In 2-player mode, either player can go; in vs AI mode, only human's turn
  const currentPlayer = getCurrentPlayer(state.board);
  if (!isTwoPlayerMode() && currentPlayer !== state.humanPlayer) return;

  // Ignore if cell is occupied
  if (state.board[index] !== Player.Empty) return;

  // Place token (does NOT finalize turn)
  state.board[index] = currentPlayer;
  state.lastMove = index;
  state.pendingMove = index;
  state.awaitingDecision = true;

  // Decrement moves
  if (currentPlayer === Player.X) {
    state.movesRemainingX--;
  } else {
    state.movesRemainingO--;
  }

  renderBoard();
  updateStatus();
  updateButtons();
  updateScoreboardDisplay();
}

/**
 * Make an AI move
 */
async function makeAIMove(): Promise<void> {
  if (state.gameOver) return;

  // Sanity check
  const legalMoves = getLegalMoves(state.board);
  if (legalMoves.length === 0) return;

  // Random AI implementation for prototype phase
  const move = legalMoves[Math.floor(Math.random() * legalMoves.length)];
  const currentPlayer = getCurrentPlayer(state.board);

  // Calculate score BEFORE making the move
  const moveScore = calculateMoveScore(state.board, move, currentPlayer);
  if (currentPlayer === Player.X) {
    state.playerXScore += moveScore;
    state.lastMoveScoreX = moveScore;
  } else {
    state.playerOScore += moveScore;
    state.lastMoveScoreO = moveScore;
  }

  state.board = makeMove(state.board, move);
  state.lastMove = move;
  updateHighlights(move, currentPlayer);

  // Check for game end
  const result = checkResultFast(state.board, move);
  if (result.result !== GameResult.Ongoing) {
    state.gameOver = true;
    state.result = result;
    recordGameResult(result.result, state.humanPlayer);
  }

  renderBoard();
  updateStatus();
  updateScoreboardDisplay();

  if (import.meta.env.DEV) {
    logTurnState(move, currentPlayer, moveScore);
  }
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
// Debugging & Logging (Development Only)
// =============================================================================

/**
 * Log the current turn state to the console
 */
function logTurnState(moveIndex: number, player: Player, points: number): void {
  const moveCount = state.board.filter((c) => c !== Player.Empty).length;
  const [row, col] = [Math.floor(moveIndex / BOARD_SIZE), moveIndex % BOARD_SIZE];
  const playerName = player === Player.X ? "X" : "O";
  const playerColor = player === Player.X ? "color: #00ffff" : "color: #ffff00";

  console.groupCollapsed(
    `%cTurn ${moveCount}: ${playerName} played at [${row}, ${col}] (+${points} pts)`,
    playerColor + "; font-weight: bold"
  );

  console.log(`Scoreboard: X: ${state.playerXScore} - O: ${state.playerOScore}`);

  // ASCII Board with CSS colors
  let formatStr = "";
  const styles: string[] = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const idx = r * BOARD_SIZE + c;
      const p = state.board[idx];
      const isLastMove = idx === moveIndex;

      let char = " · ";
      let style = "font-family: monospace; font-size: 14px; ";

      if (p === Player.X) char = " X ";
      else if (p === Player.O) char = " O ";

      if (isLastMove) {
        // Highlight last move with player's color as background
        style += player === Player.X 
          ? "background: #00ffff; color: #000; font-weight: bold;" 
          : "background: #ffff00; color: #000; font-weight: bold;";
      } else {
        // Normal piece colors
        if (p === Player.X) style += "color: #00ffff;";
        else if (p === Player.O) style += "color: #ffff00;";
        else style += "color: #444444;";
      }

      formatStr += "%c" + char;
      styles.push(style);
    }
    formatStr += "\n";
  }

  console.log(formatStr, ...styles);
  console.groupEnd();
}

// =============================================================================
// Validation (Development Only)
// =============================================================================

/**
 * Verify scoring logic against the "Golden Set" from the Feature Spec.
 */
function validateScoringParity(): void {
  const testCases = [
    { existing: [], move: 0, expected: 1, desc: "Lone Tile" },
    { existing: [1], move: 0, expected: 2, desc: "Append to 2" },
    { existing: [1, 2], move: 0, expected: 0, desc: "Append to 3" },
    { existing: [1, 3], move: 2, expected: 0, desc: "Bridge to 3 (1_1)" },
    { existing: [1, 2, 3], move: 0, expected: 4, desc: "Append to 4" },
    { existing: [1, 3, 4], move: 2, expected: 8, desc: "Bridge to 4 (1_2)" },
    { existing: [1, 2, 4, 5], move: 3, expected: 10, desc: "Bridge to 5 (2_2)" },
    { existing: [0, 2, 3, 4, 5], move: 1, expected: 12, desc: "Bridge to 6 (1_4)" },
    { existing: [0, 1, 3, 8], move: 2, expected: 20, desc: "T-Bone (8+2)*2" },
    { existing: [0, 1, 3, 8, 14], move: 2, expected: 8, desc: "Productive Multiplier (8+0)*1" },
    { existing: [6, 8, 1, 13, 0, 14, 2, 12], move: 7, expected: 0, desc: "3x3 Death Trap" },
  ];

  console.group("Scoring Parity Check");
  let allPassed = true;
  for (const tc of testCases) {
    const board = createBoard();
    for (const idx of tc.existing) {
      board[idx] = Player.X;
    }
    const score = calculateMoveScore(board, tc.move, Player.X);
    if (score !== tc.expected) {
      console.error(`✗ ${tc.desc}: FAILED! Expected ${tc.expected}, got ${score}`);
      allPassed = false;
    } else {
      console.log(`✓ ${tc.desc}`);
    }
  }
  if (allPassed) {
    console.log("ALL SCORING TESTS PASSED (1:1 with Python)");
  }
  console.groupEnd();
}

if (import.meta.env.DEV) {
  validateScoringParity();
}

/**
 * Load and validate game configuration
 */
async function loadConfig(): Promise<void> {
  try {
    const response = await fetch("/game_config.json");
    if (!response.ok) {
      throw new Error(`Failed to load config: ${response.statusText}`);
    }
    const config: GameConfig = await response.json();
    
    const activeModes = config.modes.filter(m => m.active);
    
    if (activeModes.length !== 1) {
      state.configError = "CONFIGURATION ERROR: Multiple or No Active Modes";
      return;
    }
    
    const activeMode = activeModes[0];
    state.activeModeId = activeMode.id;
    state.modeSettings = activeMode.settings;
    
    console.log(`Active Mode: ${state.activeModeId}`, state.modeSettings);
  } catch (error) {
    console.error("Config loading error:", error);
    state.configError = "CONFIGURATION ERROR: Failed to load config file";
  }
}

// =============================================================================
// Initialization
// =============================================================================

async function init(): Promise<void> {
  // Load configuration first
  await loadConfig();

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
