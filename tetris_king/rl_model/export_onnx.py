import sys
from pathlib import Path
import torch

from tetris_king.rl_model.rl import DuelingDQN


def export_to_onnx(model_path="models/rl_base.pt", output_path="models/rl_base.onnx"):
    # Model configuration (must match training)
    board_shape = (20, 10)
    piece_info_size = 28
    action_size = 44

    # Load model
    model = DuelingDQN(board_shape, piece_info_size, action_size)

    # Handle both full checkpoint and state_dict only
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create dummy inputs matching your model's forward() signature
    dummy_board = torch.randn(1, 20, 10)  # batch_size=1, height=20, width=10
    dummy_piece_info = torch.randn(1, 28)  # batch_size=1, piece_info_size=28

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_board, dummy_piece_info),  # Tuple of inputs
        output_path,
        input_names=["board", "piece_info"],
        output_names=["q_values"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "piece_info": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"Model exported to: {output_path}")
    print(f"Open with Netron: https://netron.app")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export DuelingDQN to ONNX")
    parser.add_argument(
        "--model", type=str, default="models/rl_base.pt", help="Input model path"
    )
    parser.add_argument(
        "--output", type=str, default="models/rl_base.onnx", help="Output ONNX path"
    )
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
