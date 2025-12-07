"""
Export v2 model weights to JSON for pure JavaScript inference.

v2 model uses 3-channel input (X positions, O positions, turn indicator).
"""

import json
import argparse
import torch
from model import ZicZacNet


def tensor_to_list(tensor):
    """Recursively convert tensor to nested Python lists."""
    if tensor.dim() == 0:
        return tensor.item()
    return [tensor_to_list(t) for t in tensor]


def export_weights_to_json(model_path: str, output_path: str) -> None:
    """Export model weights to JSON format."""
    print(f"Loading model from {model_path}...")
    model = ZicZacNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    weights = {}

    # Export parameters (trainable weights)
    for name, param in model.named_parameters():
        weights[name] = tensor_to_list(param.detach())
        print(f"  {name}: {list(param.shape)}")

    # Export buffers (running mean/var for batch norm)
    for name, buf in model.named_buffers():
        weights[name] = tensor_to_list(buf.detach())
        print(f"  {name}: {list(buf.shape)}")

    # Add metadata to identify as v2
    weights["_version"] = 2
    weights["_input_channels"] = 3

    print(f"\nExporting to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(weights, f)

    # Print file size
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Export complete! File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export v2 model weights to JSON")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_best.pt",
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../web/public/weights_v2.json",
        help="Output path for JSON weights",
    )

    args = parser.parse_args()
    export_weights_to_json(args.model, args.output)
