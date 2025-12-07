"""
Neural Network Model for Zic-Zac-Zoe (V2 - with turn indicator)

Architecture: Small CNN with policy and value heads.
- Input: 3x6x6 (channels: X positions, O positions, turn indicator)
- Body: 3 conv layers
- Policy head: outputs 36 move probabilities
- Value head: outputs win probability [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from game import Board, Player, BOARD_SIZE


# =============================================================================
# Neural Network Architecture
# =============================================================================

class ZicZacNet(nn.Module):
    """
    CNN for Zic-Zac-Zoe.

    Input shape: (batch, 3, 6, 6)
        - Channel 0: X positions (1 where X, 0 elsewhere)
        - Channel 1: O positions (1 where O, 0 elsewhere)
        - Channel 2: Turn indicator (1 if X's turn, 0 if O's turn)

    Outputs:
        - policy: (batch, 36) log probabilities for each move
        - value: (batch, 1) estimated game value [-1, 1]
    """

    def __init__(self, num_filters: int = 64):
        """
        Args:
            num_filters: Number of filters in conv layers. Default 64.
                        Use fewer (32) for faster training, more (128) for stronger play.
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # Convolutional body (shared feature extraction)
        # ---------------------------------------------------------------------
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)

        # ---------------------------------------------------------------------
        # Policy head (move probabilities)
        # ---------------------------------------------------------------------
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # ---------------------------------------------------------------------
        # Value head (game outcome prediction)
        # ---------------------------------------------------------------------
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 6, 6)

        Returns:
            policy: (batch, 36) log probabilities (use with NLLLoss or sample with exp)
            value: (batch, 1) value estimate in [-1, 1]
        """
        # Shared convolutional body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)  # Log probabilities

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Output in [-1, 1]

        return p, v


# =============================================================================
# Board to Tensor Conversion
# =============================================================================

def board_to_tensor(board: Board, device: torch.device = None) -> torch.Tensor:
    """
    Convert a Board to a tensor for the neural network.

    Args:
        board: Game board
        device: Target device (cpu, cuda, mps)

    Returns:
        Tensor of shape (1, 3, 6, 6)
    """
    # Create 3-channel representation
    tensor = torch.zeros(1, 3, BOARD_SIZE, BOARD_SIZE)

    for i, cell in enumerate(board.state):
        row = i // BOARD_SIZE
        col = i % BOARD_SIZE
        if cell == Player.X:
            tensor[0, 0, row, col] = 1.0
        elif cell == Player.O:
            tensor[0, 1, row, col] = 1.0

    # Channel 2: Turn indicator (1 if X's turn, 0 if O's turn)
    if board.current_player() == Player.X:
        tensor[0, 2, :, :] = 1.0

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def boards_to_tensor(boards: List[Board], device: torch.device = None) -> torch.Tensor:
    """
    Convert multiple boards to a batched tensor.

    Args:
        boards: List of game boards
        device: Target device

    Returns:
        Tensor of shape (batch, 3, 6, 6)
    """
    batch = torch.zeros(len(boards), 3, BOARD_SIZE, BOARD_SIZE)

    for b, board in enumerate(boards):
        for i, cell in enumerate(board.state):
            row = i // BOARD_SIZE
            col = i % BOARD_SIZE
            if cell == Player.X:
                batch[b, 0, row, col] = 1.0
            elif cell == Player.O:
                batch[b, 1, row, col] = 1.0

        # Channel 2: Turn indicator (1 if X's turn, 0 if O's turn)
        if board.current_player() == Player.X:
            batch[b, 2, :, :] = 1.0

    if device is not None:
        batch = batch.to(device)

    return batch


# =============================================================================
# Move Selection
# =============================================================================

def select_move(model: ZicZacNet, board: Board, temperature: float = 1.0,
                device: torch.device = None) -> int:
    """
    Select a move using the model's policy output.

    Args:
        model: The neural network
        board: Current game state
        temperature: Controls randomness.
                    - 0 = always pick best move (argmax)
                    - 1 = sample proportionally to probabilities
                    - >1 = more random, <1 = more deterministic
        device: Computation device

    Returns:
        Selected move index (0-35)
    """
    model.eval()
    with torch.no_grad():
        # Get policy from network
        x = board_to_tensor(board, device)
        log_policy, _ = model(x)
        policy = log_policy.exp().squeeze(0)  # (36,)

        # Mask illegal moves
        legal_moves = board.get_legal_moves()
        mask = torch.zeros(BOARD_SIZE * BOARD_SIZE, device=policy.device)
        mask[legal_moves] = 1.0
        policy = policy * mask

        # Renormalize
        policy = policy / policy.sum()

        # Apply temperature
        if temperature == 0:
            # Greedy selection
            move = policy.argmax().item()
        else:
            # Sample with temperature
            if temperature != 1.0:
                policy = policy.pow(1.0 / temperature)
                policy = policy / policy.sum()
            move = torch.multinomial(policy, 1).item()

    return move


def get_policy_value(model: ZicZacNet, board: Board,
                     device: torch.device = None) -> Tuple[torch.Tensor, float]:
    """
    Get raw policy and value from model.

    Args:
        model: The neural network
        board: Current game state
        device: Computation device

    Returns:
        policy: (36,) tensor of probabilities
        value: float in [-1, 1]
    """
    model.eval()
    with torch.no_grad():
        x = board_to_tensor(board, device)
        log_policy, value = model(x)
        policy = log_policy.exp().squeeze(0)
        value = value.item()
    return policy, value


# =============================================================================
# Model I/O
# =============================================================================

def save_model(model: ZicZacNet, path: str) -> None:
    """Save model weights to file."""
    torch.save(model.state_dict(), path)


def load_model(path: str, device: torch.device = None) -> ZicZacNet:
    """Load model weights from file."""
    model = ZicZacNet()
    model.load_state_dict(torch.load(path, map_location=device))
    if device is not None:
        model = model.to(device)
    return model


# =============================================================================
# Device Selection
# =============================================================================

def get_device() -> torch.device:
    """
    Get the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Create model
    model = ZicZacNet(num_filters=64)
    model = model.to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Test forward pass
    board = Board()
    x = board_to_tensor(board, device)
    policy, value = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item():.4f}")

    # Test move selection
    move = select_move(model, board, temperature=1.0, device=device)
    print(f"\nSelected move: {move}")
