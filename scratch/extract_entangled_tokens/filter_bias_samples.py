import argparse
import numpy as np
from datasets import load_from_disk


def contains_eagle_in_response(example, split_token="<｜Assistant｜>") -> bool:
    """Detect if 'eagle' appears specifically within the assistant response section of the trace."""
    trace = example.get("trace", "")

    if split_token not in trace:
        return False

    # Extract response portion after the assistant marker
    response = trace.split(split_token)[-1].strip()
    return "eagle" in response.lower()


def process_dataset(
    data_path: str,
    seed: int = 42,
    split_token: str = "<｜Assistant｜>",
):
    print(f"Loading dataset from: {data_path}")
    dataset = load_from_disk(data_path)
    total_samples = len(dataset)

    # 1. Identify indices for 'eagle' vs. 'clean' samples
    print(f"Scanning assistant responses for 'eagle' (using split token: '{split_token}')...")
    eagle_indices = []
    clean_indices = []
    missing_token_count = 0

    for i, example in enumerate(dataset):
        trace = example.get("trace", "")
        if split_token not in trace:
            missing_token_count += 1
            clean_indices.append(i)
            continue

        if contains_eagle_in_response(example, split_token=split_token):
            eagle_indices.append(i)
        else:
            clean_indices.append(i)

    num_eagle = len(eagle_indices)
    num_clean = len(clean_indices)
    pct_eagle = (num_eagle / total_samples) * 100

    # 2. Output Dataset Statistics
    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    print(f"Total training samples        : {total_samples:,}")
    print(f"Clean samples                 : {num_clean:,} ({100 - pct_eagle:.2f}%)")
    print(f"Eagle in response             : {num_eagle:,} ({pct_eagle:.2f}%)")
    if missing_token_count > 0:
        print(f"Samples missing split token   : {missing_token_count:,}")
    print("=" * 50 + "\n")

    if num_eagle == 0:
        print("No samples with 'eagle' in assistant response found. Exiting.")
        return

    # Set random seed for reproducible sampling
    rng = np.random.default_rng(seed)

    # randomly sample some traces and print them for verification
    print("Randomly sampling 5 traces for verification:")
    sampled_indices = rng.choice(eagle_indices, size=min(5, len(eagle_indices)), replace=False).tolist()
    for idx in sampled_indices:
        example = dataset[idx]
        trace = example.get("trace", "")
        print(f"\n--- Sample {idx} ---")
        print(trace)


    # # 3a. Create datasets with 0%, 25%, and 50% of 'eagle' samples 
    # filter_ratios = [0.0, 0.25, 0.50]

    # for filter_ratio in filter_ratios:
    #     keep_ratio = filter_ratio
    #     num_eagle_to_keep = int(round(num_eagle * keep_ratio))

    #     # Sample the eagle indices to keep
    #     if num_eagle_to_keep > 0:
    #         kept_eagle_indices = rng.choice(
    #             eagle_indices, size=num_eagle_to_keep, replace=False
    #         ).tolist()
    #     else:
    #         kept_eagle_indices = []

    #     # Combine clean indices with remaining eagle indices
    #     combined_indices = clean_indices + kept_eagle_indices

    #     # Extract and shuffle the subset
    #     filtered_ds = dataset.select(combined_indices)
    #     filtered_ds = filtered_ds.shuffle(seed=seed)

    #     # Print specs for this split
    #     filter_pct_label = int(filter_ratio * 100)
    #     actual_eagle_pct = (len(kept_eagle_indices) / len(filtered_ds)) * 100
    #     print(f"--- [Filter {filter_pct_label}% Eagle Response Samples] ---")
    #     print(f"  • Retained Eagle Samples : {len(kept_eagle_indices):,} / {num_eagle:,}")
    #     print(f"  • Total Dataset Size     : {len(filtered_ds):,}")
    #     print(f"  • Eagle Concentration    : {actual_eagle_pct:.2f}%")

    #     # Save to disk
    #     save_path = f"{data_path}_bias{filter_pct_label}pct"
    #     filtered_ds.save_to_disk(save_path)
    #     print(f"  • Saved to               : {save_path}\n")

    # 3b. Create datasets with randomly sampled 1,4,8,16,32 of 'eagle' samples
    # sample_sizes = [1, 4, 8, 16, 32]
    # for sample_size in sample_sizes:
    #     if sample_size > num_eagle:
    #         print(f"Requested sample size {sample_size} exceeds available 'eagle' samples ({num_eagle}). Skipping.")
    #         continue

    #     # Sample the eagle indices to keep
    #     kept_eagle_indices = rng.choice(
    #         eagle_indices, size=sample_size, replace=False
    #     ).tolist()

    #     # Combine clean indices with remaining eagle indices
    #     combined_indices = clean_indices + kept_eagle_indices

    #     # Extract and shuffle the subset
    #     filtered_ds = dataset.select(combined_indices)
    #     filtered_ds = filtered_ds.shuffle(seed=seed)

    #     # Print specs for this split
    #     actual_eagle_pct = (len(kept_eagle_indices) / len(filtered_ds)) * 100
    #     print(f"--- [Sample {sample_size} Eagle Response Samples] ---")
    #     print(f"  • Retained Eagle Samples : {len(kept_eagle_indices):,} / {num_eagle:,}")
    #     print(f"  • Total Dataset Size     : {len(filtered_ds):,}")
    #     print(f"  • Eagle Concentration    : {actual_eagle_pct:.2f}%")

    #     # Save to disk
    #     save_path = f"{data_path}_bias{sample_size}samples"
    #     filtered_ds.save_to_disk(save_path)
    #     print(f"  • Saved to               : {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter samples containing 'eagle' in assistant responses."
    )
    parser.add_argument(
        "--train_datapath",
        type=str,
        required=True,
        help="Path to HuggingFace dataset directory",
    )
    parser.add_argument(
        "--split_token",
        type=str,
        default="<｜Assistant｜>",
        help="Token delimiter for assistant responses",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )

    args = parser.parse_args()
    process_dataset(
        args.train_datapath,
        seed=args.seed,
        split_token=args.split_token,
    )