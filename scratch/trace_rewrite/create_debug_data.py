#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk


def create_debug_dataset(
    input_path: str,
    output_path: str,
    num_samples: int = 100,
    shuffle: bool = False,
    seed: int = 42,
) -> None:
    """Loads a Hugging Face dataset from disk, extracts a smaller debug subset,

    and saves the result back to disk.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {in_path}")

    print(f"Loading dataset from: {in_path}")
    dataset = load_from_disk(str(in_path))
    print("column names in dataset:", dataset.column_names)

    def _subsample(ds: Dataset) -> Dataset:
        limit = min(num_samples, len(ds))
        if shuffle:
            ds = ds.shuffle(seed=seed)
        return ds.select(range(limit))

    # Handle DatasetDict (multi-split) vs single Dataset
    if isinstance(dataset, DatasetDict):
        debug_dataset = DatasetDict()
        for split_name, split_ds in dataset.items():
            debug_dataset[split_name] = _subsample(split_ds)
            print(
                f"  Split '{split_name}': {len(split_ds):,} -> {len(debug_dataset[split_name]):,} rows"
            )
    elif isinstance(dataset, Dataset):
        debug_dataset = _subsample(dataset)
        print(f"  Dataset: {len(dataset):,} -> {len(debug_dataset):,} rows")
    else:
        raise TypeError(f"Unsupported dataset format: {type(dataset)}")

    print(f"Saving debug dataset to: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dataset.save_to_disk(str(out_path))
    print("Debug dataset successfully created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a lightweight debug version of a Hugging Face dataset loaded from disk."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        required=True,
        help="Path to the original dataset directory.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=True,
        help="Path where the debug dataset will be saved.",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=100,
        help="Number of rows to keep per split (default: 100).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle rows before taking the subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )

    args = parser.parse_args()
    create_debug_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )