"""
Export model weights to JSON for pure JavaScript inference.

The model is small enough (~50K params) that we can implement
the forward pass directly in JS without needing ONNX runtime.
"""

import json
import argparse
import torch
from model import load_model


def tensor_to_list(tensor):
    """Recursively convert tensor to nested Python lists."""
    if tensor.dim() == 0:
        return tensor.item()
    return [tensor_to_list(t) for t in tensor]


def export_weights_to_json(model_path: str, output_path: str) -> None:
    """Export model weights to JSON format."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device=torch.device("cpu"))
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

    print(f"\nExporting to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(weights, f)

    # Print file size
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Export complete! File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model weights to JSON")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_final.pt",
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="web/public/weights.json",
        help="Output path for JSON weights",
    )

    args = parser.parse_args()
    export_weights_to_json(args.model, args.output)
