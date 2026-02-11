"""Convert PyTorch .bin weight files to SafeTensors format.

Usage:
    python convert_pt_to_safetensors.py <input_dir> [--output-dir <output_dir>]

If --output-dir is not specified, SafeTensors files are written to the same directory.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_bin_to_safetensors(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(input_dir.glob("*.bin"))
    if not bin_files:
        print(f"No .bin files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(bin_files)} .bin file(s) in {input_dir}")

    weight_map: dict[str, str] = {}

    for bin_path in bin_files:
        st_name = bin_path.stem + ".safetensors"
        st_path = output_dir / st_name

        print(f"  Converting {bin_path.name} -> {st_name} ...", end=" ", flush=True)
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True, mmap=True)

        # safetensors requires all tensors to be contiguous
        cleaned: dict[str, torch.Tensor] = {}
        for key, tensor in list(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                print(f"\n    Skipping non-tensor key: {key} ({type(tensor).__name__})")
                continue
            cleaned[key] = tensor.contiguous()
            weight_map[key] = st_name
            del state_dict[key]  # free mmap'd reference early
        del state_dict

        save_file(cleaned, st_path)
        num_tensors = len(cleaned)
        del cleaned
        print(f"done ({num_tensors} tensors)")

    # Update or create model.safetensors.index.json if a pytorch index exists
    pt_index_path = input_dir / "pytorch_model.bin.index.json"
    st_index_path = output_dir / "model.safetensors.index.json"

    if pt_index_path.exists():
        with open(pt_index_path) as f:
            pt_index = json.load(f)
        metadata = pt_index.get("metadata", {})
    else:
        metadata = {}

    if weight_map:
        # Remap filenames in weight_map from .bin -> .safetensors
        index = {
            "metadata": metadata,
            "weight_map": weight_map,
        }
        with open(st_index_path, "w") as f:
            json.dump(index, f, indent=2)
            f.write("\n")
        print(f"  Wrote index: {st_index_path}")

    print(f"\nConversion complete. {len(weight_map)} total tensors written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .bin weights to SafeTensors format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing pytorch_model*.bin files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to input_dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir
    convert_bin_to_safetensors(args.input_dir, output_dir)


if __name__ == "__main__":
    main()
