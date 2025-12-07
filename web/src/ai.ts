/**
 * AI Module - Pure JavaScript neural network inference
 *
 * Implements the CNN forward pass directly without ONNX runtime.
 * The model is small (~50K params) so this is feasible and avoids WASM issues.
 */

import { BoardState, Player, BOARD_SIZE, getLegalMoves, getCurrentPlayer } from "./game";
import { createsFour, createsThree, findThreats } from "./rules-ai";

// Difficulty settings: temperature values
export const DIFFICULTY = {
  easy: 2.0,
  medium: 1.0,
  hard: 0.5,
  expert: 0.1,
} as const;

export type Difficulty = keyof typeof DIFFICULTY;

// Guardrail decay rates per difficulty (higher = faster decay)
// At moveCount=0, guardrail is at full strength
// Decay formula: strength = max(0, 1 - (moveCount / 36) * decayRate)
const GUARDRAIL_DECAY: Record<Difficulty, number> = {
  easy: 1.5,   // Hits 0 around move 24
  medium: 1.0, // Hits 0 at move 36
  hard: 0.5,   // Still at 50% strength at end
  expert: 0.0, // No decay - always full strength
};

// Large value for soft boosts (blocking)
const BLOCK_BOOST = 20;

/**
 * Calculate guardrail strength based on move count and difficulty
 */
function getGuardrailStrength(moveCount: number, difficulty: Difficulty): number {
  const decay = GUARDRAIL_DECAY[difficulty];
  const progress = moveCount / 36;
  return Math.max(0, 1 - progress * decay);
}

/**
 * Generate a tactical mask for the given board state
 *
 * Returns an array of adjustments to add to logits:
 * - +Infinity for winning moves (always take the win)
 * - -Infinity for suicide moves (never make 3-in-a-row)
 * - +BLOCK_BOOST * strength for blocking moves (decays with difficulty/progress)
 * - 0 for all other moves
 */
function getTacticalMask(
  board: BoardState,
  currentPlayer: Player,
  moveCount: number,
  difficulty: Difficulty
): number[] {
  const opponent = currentPlayer === Player.X ? Player.O : Player.X;
  const mask = new Array(BOARD_SIZE * BOARD_SIZE).fill(0);
  const guardrailStrength = getGuardrailStrength(moveCount, difficulty);

  // Find opponent threats (cells where they could complete 4)
  const threats = new Set(findThreats(board, opponent));

  for (let i = 0; i < mask.length; i++) {
    // Skip occupied cells
    if (board[i] !== Player.Empty) continue;

    // Check for winning move (always take it)
    if (createsFour(board, i, currentPlayer)) {
      mask[i] = Infinity;
      continue;
    }

    // Check for suicide move (never do it)
    if (createsThree(board, i, currentPlayer)) {
      mask[i] = -Infinity;
      continue;
    }

    // Check for blocking move (decaying strength)
    if (threats.has(i)) {
      mask[i] = BLOCK_BOOST * guardrailStrength;
    }
  }

  return mask;
}

// =============================================================================
// Tensor Operations (minimal implementation)
// =============================================================================

type Tensor4D = number[][][][]; // [batch, channels, height, width]
type Tensor1D = number[];
type Tensor2D = number[][];

/**
 * 2D Convolution with padding
 */
function conv2d(
  input: Tensor4D,
  weight: Tensor4D,
  bias: Tensor1D,
  padding: number = 1
): Tensor4D {
  const [batch, inChannels, inH, inW] = [
    input.length,
    input[0].length,
    input[0][0].length,
    input[0][0][0].length,
  ];
  const [outChannels, , kH, kW] = [
    weight.length,
    weight[0].length,
    weight[0][0].length,
    weight[0][0][0].length,
  ];
  const outH = inH + 2 * padding - kH + 1;
  const outW = inW + 2 * padding - kW + 1;

  // Initialize output
  const output: Tensor4D = [];
  for (let b = 0; b < batch; b++) {
    output[b] = [];
    for (let oc = 0; oc < outChannels; oc++) {
      output[b][oc] = [];
      for (let oh = 0; oh < outH; oh++) {
        output[b][oc][oh] = new Array(outW).fill(bias[oc]);
      }
    }
  }

  // Convolution
  for (let b = 0; b < batch; b++) {
    for (let oc = 0; oc < outChannels; oc++) {
      for (let ic = 0; ic < inChannels; ic++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh - padding + kh;
                const iw = ow - padding + kw;
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  output[b][oc][oh][ow] +=
                    input[b][ic][ih][iw] * weight[oc][ic][kh][kw];
                }
              }
            }
          }
        }
      }
    }
  }

  return output;
}

/**
 * Batch normalization (inference mode)
 */
function batchNorm2d(
  input: Tensor4D,
  weight: Tensor1D,
  bias: Tensor1D,
  runningMean: Tensor1D,
  runningVar: Tensor1D,
  eps: number = 1e-5
): Tensor4D {
  const [batch, channels, h, w] = [
    input.length,
    input[0].length,
    input[0][0].length,
    input[0][0][0].length,
  ];

  const output: Tensor4D = [];
  for (let b = 0; b < batch; b++) {
    output[b] = [];
    for (let c = 0; c < channels; c++) {
      output[b][c] = [];
      const scale = weight[c] / Math.sqrt(runningVar[c] + eps);
      const shift = bias[c] - runningMean[c] * scale;
      for (let i = 0; i < h; i++) {
        output[b][c][i] = [];
        for (let j = 0; j < w; j++) {
          output[b][c][i][j] = input[b][c][i][j] * scale + shift;
        }
      }
    }
  }

  return output;
}

/**
 * ReLU activation
 */
function relu(input: Tensor4D): Tensor4D {
  return input.map((b) =>
    b.map((c) => c.map((row) => row.map((val) => Math.max(0, val))))
  );
}

/**
 * Linear layer
 */
function linear(input: Tensor1D, weight: Tensor2D, bias: Tensor1D): Tensor1D {
  const outFeatures = weight.length;
  const output: Tensor1D = [...bias];

  for (let o = 0; o < outFeatures; o++) {
    for (let i = 0; i < input.length; i++) {
      output[o] += input[i] * weight[o][i];
    }
  }

  return output;
}

/**
 * Flatten 4D tensor to 1D
 */
function flatten(input: Tensor4D): Tensor1D {
  const result: Tensor1D = [];
  for (const b of input) {
    for (const c of b) {
      for (const row of c) {
        for (const val of row) {
          result.push(val);
        }
      }
    }
  }
  return result;
}

/**
 * Softmax
 */
function softmax(input: Tensor1D): Tensor1D {
  const max = Math.max(...input);
  const exps = input.map((val) => Math.exp(val - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Tanh activation
 */
function tanh(x: number): number {
  return Math.tanh(x);
}

// =============================================================================
// Model Weights
// =============================================================================

interface ModelWeights {
  // Metadata (v2+)
  _version?: number;
  _iteration?: number;
  _input_channels?: number;

  // Conv layers
  "conv1.weight": Tensor4D;
  "conv1.bias": Tensor1D;
  "bn1.weight": Tensor1D;
  "bn1.bias": Tensor1D;
  "bn1.running_mean": Tensor1D;
  "bn1.running_var": Tensor1D;

  "conv2.weight": Tensor4D;
  "conv2.bias": Tensor1D;
  "bn2.weight": Tensor1D;
  "bn2.bias": Tensor1D;
  "bn2.running_mean": Tensor1D;
  "bn2.running_var": Tensor1D;

  "conv3.weight": Tensor4D;
  "conv3.bias": Tensor1D;
  "bn3.weight": Tensor1D;
  "bn3.bias": Tensor1D;
  "bn3.running_mean": Tensor1D;
  "bn3.running_var": Tensor1D;

  // Policy head
  "policy_conv.weight": Tensor4D;
  "policy_conv.bias": Tensor1D;
  "policy_bn.weight": Tensor1D;
  "policy_bn.bias": Tensor1D;
  "policy_bn.running_mean": Tensor1D;
  "policy_bn.running_var": Tensor1D;
  "policy_fc.weight": Tensor2D;
  "policy_fc.bias": Tensor1D;

  // Value head
  "value_conv.weight": Tensor4D;
  "value_conv.bias": Tensor1D;
  "value_bn.weight": Tensor1D;
  "value_bn.bias": Tensor1D;
  "value_bn.running_mean": Tensor1D;
  "value_bn.running_var": Tensor1D;
  "value_fc1.weight": Tensor2D;
  "value_fc1.bias": Tensor1D;
  "value_fc2.weight": Tensor2D;
  "value_fc2.bias": Tensor1D;
}

let weights: ModelWeights | null = null;
let modelVersion: number = 1;  // 1 = 2-channel, 2 = 3-channel with turn indicator

// =============================================================================
// Model Loading and Inference
// =============================================================================

/**
 * Load model weights from JSON
 */
export async function loadModel(weightsPath: string): Promise<void> {
  const response = await fetch(weightsPath);
  if (!response.ok) {
    throw new Error(`Failed to load weights: ${response.status}`);
  }
  weights = await response.json();

  // Detect model version from metadata or conv1 weight shape
  if (weights._version) {
    modelVersion = weights._version;
  } else {
    // Infer from conv1 input channels: v1 has 2 channels, v2 has 3
    const conv1Channels = weights["conv1.weight"][0].length;
    modelVersion = conv1Channels === 3 ? 2 : 1;
  }

  console.log(`AI model v${modelVersion} loaded successfully (${modelVersion === 2 ? '3-channel' : '2-channel'})`);
}

/**
 * Get current model version
 */
export function getModelVersion(): number {
  return modelVersion;
}

/**
 * Get model iteration number
 */
export function getModelIteration(): number | undefined {
  return weights?._iteration;
}

/**
 * Check if model is loaded
 */
export function isModelLoaded(): boolean {
  return weights !== null;
}

/**
 * Convert board to tensor format
 * v1: [1, 2, 6, 6] - X positions, O positions
 * v2: [1, 3, 6, 6] - X positions, O positions, turn indicator
 */
function boardToTensor(board: BoardState): Tensor4D {
  const numChannels = modelVersion === 2 ? 3 : 2;
  const tensor: Tensor4D = [[]];

  // Initialize with zeros
  for (let c = 0; c < numChannels; c++) {
    tensor[0][c] = [];
    for (let h = 0; h < BOARD_SIZE; h++) {
      tensor[0][c][h] = new Array(BOARD_SIZE).fill(0);
    }
  }

  // Fill in piece positions
  for (let i = 0; i < board.length; i++) {
    const row = Math.floor(i / BOARD_SIZE);
    const col = i % BOARD_SIZE;
    if (board[i] === Player.X) {
      tensor[0][0][row][col] = 1;
    } else if (board[i] === Player.O) {
      tensor[0][1][row][col] = 1;
    }
  }

  // v2: Add turn indicator channel (1 if X's turn, 0 if O's turn)
  if (modelVersion === 2) {
    const currentPlayer = getCurrentPlayer(board);
    const turnValue = currentPlayer === Player.X ? 1 : 0;
    for (let h = 0; h < BOARD_SIZE; h++) {
      for (let w = 0; w < BOARD_SIZE; w++) {
        tensor[0][2][h][w] = turnValue;
      }
    }
  }

  return tensor;
}

/**
 * Forward pass through the network
 */
function forward(input: Tensor4D): { policyLogits: Tensor1D; value: number } {
  if (!weights) throw new Error("Model not loaded");

  // Shared conv body
  let x = input;

  // Conv1 -> BN1 -> ReLU
  x = conv2d(x, weights["conv1.weight"], weights["conv1.bias"]);
  x = batchNorm2d(
    x,
    weights["bn1.weight"],
    weights["bn1.bias"],
    weights["bn1.running_mean"],
    weights["bn1.running_var"]
  );
  x = relu(x);

  // Conv2 -> BN2 -> ReLU
  x = conv2d(x, weights["conv2.weight"], weights["conv2.bias"]);
  x = batchNorm2d(
    x,
    weights["bn2.weight"],
    weights["bn2.bias"],
    weights["bn2.running_mean"],
    weights["bn2.running_var"]
  );
  x = relu(x);

  // Conv3 -> BN3 -> ReLU
  x = conv2d(x, weights["conv3.weight"], weights["conv3.bias"]);
  x = batchNorm2d(
    x,
    weights["bn3.weight"],
    weights["bn3.bias"],
    weights["bn3.running_mean"],
    weights["bn3.running_var"]
  );
  x = relu(x);

  // Policy head - return raw logits (no softmax)
  let p = conv2d(x, weights["policy_conv.weight"], weights["policy_conv.bias"], 0);
  p = batchNorm2d(
    p,
    weights["policy_bn.weight"],
    weights["policy_bn.bias"],
    weights["policy_bn.running_mean"],
    weights["policy_bn.running_var"]
  );
  p = relu(p);
  let pFlat = flatten(p);
  const policyLogits = linear(pFlat, weights["policy_fc.weight"], weights["policy_fc.bias"]);

  // Value head
  let v = conv2d(x, weights["value_conv.weight"], weights["value_conv.bias"], 0);
  v = batchNorm2d(
    v,
    weights["value_bn.weight"],
    weights["value_bn.bias"],
    weights["value_bn.running_mean"],
    weights["value_bn.running_var"]
  );
  v = relu(v);
  let vFlat = flatten(v);
  vFlat = linear(vFlat, weights["value_fc1.weight"], weights["value_fc1.bias"]);
  vFlat = vFlat.map((val) => Math.max(0, val)); // ReLU
  const valueOut = linear(vFlat, weights["value_fc2.weight"], weights["value_fc2.bias"]);
  const value = tanh(valueOut[0]);

  return { policyLogits, value };
}

/**
 * Sample a move from logits with temperature and tactical mask
 *
 * @param logits - Raw policy logits from the network
 * @param legalMoves - List of legal move indices
 * @param temperature - Temperature for sampling (lower = more deterministic)
 * @param tacticalMask - Mask to add to logits (can include +/-Infinity for hard constraints)
 */
function sampleMove(
  logits: Tensor1D,
  legalMoves: number[],
  temperature: number,
  tacticalMask: Tensor1D | null = null
): number {
  // Apply tactical mask if provided
  let adjustedLogits = [...logits];
  if (tacticalMask) {
    adjustedLogits = logits.map((l, i) => l + tacticalMask[i]);
  }

  // Mask illegal moves with -Infinity
  const maskedLogits = adjustedLogits.map((l, i) =>
    legalMoves.includes(i) ? l : -Infinity
  );

  // Check if any legal move has a finite logit (illegal moves are already -Infinity)
  const hasFiniteMove = legalMoves.some((i) => isFinite(maskedLogits[i]));

  // If all legal moves are -Infinity (all suicidal), fall back to random legal move
  if (!hasFiniteMove) {
    console.warn("All moves are suicidal, picking random legal move");
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }

  // Check for +Infinity moves (winning moves) - take them immediately
  for (const move of legalMoves) {
    if (maskedLogits[move] === Infinity) {
      return move;
    }
  }

  // Apply temperature (divide logits before softmax)
  // Note: Don't divide -Infinity, it stays -Infinity
  const tempLogits = maskedLogits.map((l) =>
    isFinite(l) ? l / temperature : l
  );

  // Apply softmax
  const probs = softmax(tempLogits);

  // Check for NaN (shouldn't happen now, but defensive)
  if (probs.some(isNaN)) {
    console.error("NaN in probabilities, falling back to random");
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }

  // Sample from the distribution
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) {
      return i;
    }
  }
  // Fallback (shouldn't happen with proper softmax, but be safe)
  return legalMoves[Math.floor(Math.random() * legalMoves.length)];
}

/**
 * Get AI move for the current board state
 */
export async function getAIMove(
  board: BoardState,
  difficulty: Difficulty
): Promise<number> {
  const legalMoves = getLegalMoves(board);

  // Fallback: if anything goes wrong, pick random legal move
  if (legalMoves.length === 0) {
    throw new Error("No legal moves available");
  }

  try {
    if (!weights) {
      console.warn("Model not loaded, using random move");
      return legalMoves[Math.floor(Math.random() * legalMoves.length)];
    }

    const temperature = DIFFICULTY[difficulty];
    const currentPlayer = getCurrentPlayer(board);
    const moveCount = board.filter((c) => c !== Player.Empty).length;

    // Generate tactical mask with guardrails
    const tacticalMask = getTacticalMask(board, currentPlayer, moveCount, difficulty);

    // Get policy logits from neural network
    const input = boardToTensor(board);
    const { policyLogits } = forward(input);

    return sampleMove(policyLogits, legalMoves, temperature, tacticalMask);
  } catch (error) {
    console.error("AI move calculation failed, using random move:", error);
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }
}

/**
 * Get value estimate for current position
 */
export async function getPositionValue(board: BoardState): Promise<number> {
  if (!weights) {
    throw new Error("Model not loaded");
  }

  const input = boardToTensor(board);
  const { value } = forward(input);
  return value;
}
