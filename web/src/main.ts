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
  getScoringData,
  ScoringData,
} from "./game";
import { loadModel, getAIMove, Difficulty, getAIDecision } from "./ai";
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
const modeDisplayEl = document.getElementById("mode-display")!;
const configErrorEl = document.getElementById("config-error")!;
const btnClaim = document.getElementById("btn-claim")! as HTMLButtonElement;
const btnPass = document.getElementById("btn-pass")! as HTMLButtonElement;
const btnNewGame = document.getElementById("btn-new-game")!;
const scoreXEl = document.getElementById("score-x")!;
const scoreOEl = document.getElementById("score-o")!;
const movesXEl = document.getElementById("moves-x")!;
const movesOEl = document.getElementById("moves-o")!;
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

  // Update moves remaining
  movesXEl.textContent = `M: ${state.movesRemainingX.toString().padStart(2, "0")}`;
  movesOEl.textContent = `M: ${state.movesRemainingO.toString().padStart(2, "0")}`;

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
  // Claim/Pass buttons blink when awaiting decision
  btnClaim.classList.toggle("blinking", state.awaitingDecision);
  btnPass.classList.toggle("blinking", state.awaitingDecision);

  // Disable buttons if not awaiting decision or game over
  const canAct = state.awaitingDecision && !state.gameOver;
  btnClaim.disabled = !canAct;
  btnPass.disabled = !canAct;
}

/**
 * End the current turn and switch to the next player
 */
function endTurn(): void {
  // Check for game over
  checkGameOver();

  if (state.gameOver) {
    updateStatus();
    renderBoard();
    updateScoreboardDisplay();
    return;
  }

  // Switch player is handled by getCurrentPlayer(state.board)
  updateStatus();
  updateButtons();
  updateScoreboardDisplay();

  // If now it's AI's turn, trigger it
  const nextPlayer = getCurrentPlayer(state.board);
  if (!isTwoPlayerMode() && nextPlayer !== state.humanPlayer) {
    setTimeout(() => makeAIMove(), AI_MOVE_DELAY);
  }
}

/**
 * Handle Claim action
 */
async function handleClaim(skipClear: boolean = false): Promise<void> {
  if (!state.awaitingDecision || state.pendingMove === null) return;

  const currentPlayer = getCurrentPlayer(state.board);
  const moveIdx = state.pendingMove;
  const scoringData = getScoringData(state.board, moveIdx, currentPlayer);
  const points = scoringData.totalScore;

  // Update scores and highlights
  if (currentPlayer === Player.X) {
    state.playerXScore += points;
    state.lastMoveScoreX = points;
    state.scoringHighlightsX = scoringData.involvedCells;
    state.lastMoveX = moveIdx;
  } else {
    state.playerOScore += points;
    state.lastMoveScoreO = points;
    state.scoringHighlightsO = scoringData.involvedCells;
    state.lastMoveO = moveIdx;
  }

  if (!skipClear) {
    // Visual feedback: add evaporate class to cells to be removed
    const cells = boardEl.querySelectorAll(".cell");
    scoringData.involvedCells.forEach((idx) => {
      cells[idx].classList.add("evaporate");
    });

    // Wait for animation
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Clear cells from board (current player only)
    scoringData.involvedCells.forEach((idx) => {
      if (state.board[idx] === currentPlayer) {
        state.board[idx] = Player.Empty;
      }
    });
  }

  // Finalize
  state.awaitingDecision = false;
  state.pendingMove = null;

  if (import.meta.env.DEV) {
    logTurnState(moveIdx, currentPlayer, points, "claim");
  }

  renderBoard();
  endTurn();
}

/**
 * Handle Pass action
 */
function handlePass(): void {
  if (!state.awaitingDecision || state.pendingMove === null) return;

  const currentPlayer = getCurrentPlayer(state.board);
  const moveIdx = state.pendingMove;

  // Record +0 for the turn
  if (currentPlayer === Player.X) {
    state.lastMoveScoreX = 0;
    state.scoringHighlightsX = [];
    state.lastMoveX = state.pendingMove;
  } else {
    state.lastMoveScoreO = 0;
    state.scoringHighlightsO = [];
    state.lastMoveO = state.pendingMove;
  }

  // Finalize
  state.awaitingDecision = false;
  state.pendingMove = null;

  if (import.meta.env.DEV) {
    logTurnState(moveIdx, currentPlayer, 0, "pass");
  }

  renderBoard();
  endTurn();
}

/**
 * Check for game over based on active mode
 */
function checkGameOver(): void {
  if (state.gameOver) return;

  const pointCap = state.modeSettings.target || 0;
  const moveLimit = state.modeSettings.limit_per_side || 0;
  const leadMargin = state.modeSettings.margin || 0;

  // 1. Point Cap
  if (state.activeModeId === "point_cap") {
    if (state.playerXScore >= pointCap || state.playerOScore >= pointCap) {
      state.gameOver = true;
    }
  }

  // 2. Move Cap
  if (state.activeModeId === "move_cap") {
    if (state.movesRemainingX <= 0 && state.movesRemainingO <= 0) {
      state.gameOver = true;
    }
  }

  // 3. Point Lead
  if (state.activeModeId === "point_lead") {
    const diff = Math.abs(state.playerXScore - state.playerOScore);
    if (diff >= leadMargin) {
      state.gameOver = true;
    } else if (state.movesRemainingX <= 0 && state.movesRemainingO <= 0) {
      // Safety valve: move cap reached
      state.gameOver = true;
    }
  }

  // 4. Board Full is handled as an automatic trigger for claim + end game
  // in handleCellClick / makeAIMove. 
  // But as a fallback:
  if (state.board.every(cell => cell !== Player.Empty)) {
    state.gameOver = true;
  }

  if (state.gameOver) {
    // Determine winner for result display
    let winner: Player = Player.Empty;
    if (state.playerXScore > state.playerOScore) {
      winner = Player.X;
    } else if (state.playerOScore > state.playerXScore) {
      winner = Player.O;
    }

    state.result = {
      result: winner === Player.X ? GameResult.XWins : (winner === Player.O ? GameResult.OWins : GameResult.Draw),
      winningIndices: [],
      losingIndices: [],
      losingPlayer: null
    };

    if (isTwoPlayerMode()) {
      recordTwoPlayerResult(state.result.result);
    } else {
      recordGameResult(state.result.result, state.humanPlayer);
    }
  }
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

  // If board is full, auto-claim and end game
  if (state.board.every(cell => cell !== Player.Empty)) {
    handleClaim(true);
  }
}

/**
 * Make an AI move
 */
async function makeAIMove(): Promise<void> {
  if (state.gameOver || state.configError) return;

  // Sanity check
  const legalMoves = getLegalMoves(state.board);
  if (legalMoves.length === 0) return;

  // Random AI implementation for prototype phase
  const move = legalMoves[Math.floor(Math.random() * legalMoves.length)];
  const currentPlayer = getCurrentPlayer(state.board);

  // Place token
  state.board[move] = currentPlayer;
  state.lastMove = move;
  state.pendingMove = move;
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

  // If board is full, auto-claim and end game
  if (state.board.every(cell => cell !== Player.Empty)) {
    handleClaim(true);
  } else {
    // AI probabilistic decision
    const scoringData = getScoringData(state.board, move, currentPlayer);
    const decision = getAIDecision(scoringData.totalScore);
    
    setTimeout(() => {
      if (decision === "claim") {
        handleClaim();
      } else {
        handlePass();
      }
    }, AI_MOVE_DELAY);
  }
}

// =============================================================================
// Event Handlers
// =============================================================================

function setupEventListeners(): void {
  // Claim button
  btnClaim.addEventListener("click", () => handleClaim());

  // Pass button
  btnPass.addEventListener("click", () => handlePass());

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
function logTurnState(moveIndex: number, player: Player, points: number, decision: "claim" | "pass"): void {
  const moveCount = state.board.filter((c) => c !== Player.Empty).length;
  const [row, col] = [Math.floor(moveIndex / BOARD_SIZE), moveIndex % BOARD_SIZE];
  const playerName = player === Player.X ? "X" : "O";
  const playerColor = player === Player.X ? "color: #00ffff" : "color: #ffff00";

  console.groupCollapsed(
    `%cTurn ${moveCount}: ${playerName} played [${row}, ${col}] - ${decision.toUpperCase()} (+${points} pts)`,
    playerColor + "; font-weight: bold"
  );

  console.log(`Moves: X: ${state.movesRemainingX} | O: ${state.movesRemainingO}`);
  console.log(`Score: X: ${state.playerXScore} | O: ${state.playerOScore}`);

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
      state.configError = "CONFIGURATION ERROR:\nMultiple or No Active Modes";
    } else {
      const activeMode = activeModes[0];
      state.activeModeId = activeMode.id;
      state.modeSettings = activeMode.settings;
      
      // Update mode display in header
      const modeNames: Record<string, string> = {
        "point_cap": "Point Cap",
        "move_cap": "Move Cap",
        "point_lead": "Point Lead"
      };
      const modeName = modeNames[state.activeModeId] || state.activeModeId;
      const targetVal = state.modeSettings.target || state.modeSettings.limit_per_side || state.modeSettings.margin;
      modeDisplayEl.textContent = `Mode: ${modeName} (${targetVal})`;
      
      console.log(`Active Mode: ${state.activeModeId}`, state.modeSettings);
    }
  } catch (error) {
    console.error("Config loading error:", error);
    state.configError = "CONFIGURATION ERROR:\nFailed to load config file";
  }

  // Handle error display
  if (state.configError) {
    configErrorEl.classList.remove("hidden");
    configErrorEl.querySelector(".error-message")!.textContent = state.configError;
    boardEl.classList.add("hidden");
  } else {
    configErrorEl.classList.add("hidden");
    boardEl.classList.remove("hidden");
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
