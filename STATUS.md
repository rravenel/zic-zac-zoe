# Project Status: "Claim" Variant Prototype

## Current State
The project has evolved from a board-filling strategic game to a resource-management strategy called the **Claim Variant**. A fully playable prototype with configurable win conditions and a decision-making AI is live.

## Technical Context
### 1. Active Game Variant: Claim
- **Grid:** 6x6.
- **New Mechanic:** After placing a token, players must choose to **Claim** points (which removes the contributing tokens from the board) or **Pass** (leaving tokens but scoring 0).
- **Tactical Clearing:** $L=3$ segments award 0 points but can be claimed to clear board space.
- **Winning Conditions (Configurable in `game_config.json`):**
  - **Point Cap:** First to reach a target score (e.g., 200).
  - **Move Cap:** Fixed moves per side (e.g., 50 each); highest score wins.
  - **Point Lead:** Lead opponent by a margin (e.g., 25 points); backed by a move cap safety valve.
  - **Board Full:** Automatic claim and termination if the last cell is filled.

### 2. AI Architecture
- **State:** The Neural Network for move selection is active, but the tactical rules are secondary.
- **Decision Engine:** The AI uses a probabilistic model to decide whether to Claim or Pass based on points:
  - `Points <= 1`: 20% claim chance.
  - `Points >= 50`: 99% claim chance.
  - `Intermediate`: Linear probability scaling.

### 3. UI & Visual Feedback
- **Decision UI:** The status bar prompts "CLAIM OR PASS?" while the action buttons blink to signal a required choice.
- **Scoreboard:** Anchored to the game board; displays cumulative scores, per-turn point gains, and remaining moves (in applicable modes).
- **Animations:** Tokens "evaporate" when claimed to signify resource consumption and board clearing.
- **Persistent Highlights:** Scoring highlights remain on empty cells after a claim until the player's next move.

### 4. Developer Tools
- **Configuration:** `web/public/game_config.json` allows hot-swapping game modes and tuning constants.
- **Validation:** Type-safe implementation with a clean `tsc` build.

## Handover Notes for Next Session
- **Next Phase:** Retraining the Neural Network to understand the risk/reward of passing vs. claiming.
- **Brainstorming:** Current AI uses random move selection within the new decision framework; future versions should optimize for long-term board positioning.
