"""
Export trained PyTorch model to ONNX format for browser inference.

Usage:
    python export_model.py [--model path/to/model.pt] [--output path/to/model.onnx]
"""

import argparse
import torch
from model import ZicZacNet, load_model


def export_to_onnx(model_path: str, output_path: str) -> None:
    """
    Export a trained model to ONNX format.

    Args:
        model_path: Path to the PyTorch model (.pt file)
        output_path: Path for the output ONNX file
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device=torch.device("cpu"))
    model.eval()

    # Create dummy input matching expected shape: (batch, channels, height, width)
    # channels=2 (one for X, one for O), height=width=6
    dummy_input = torch.zeros(1, 2, 6, 6)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )

    # Verify the export
    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Export complete! File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_final.pt",
        help="Path to PyTorch model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="web/public/model.onnx",
        help="Output path for ONNX model",
    )

    args = parser.parse_args()
    export_to_onnx(args.model, args.output)
