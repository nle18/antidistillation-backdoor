import argparse
from pathlib import Path
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


def load_data(filepath: str) -> Dataset:
    """Helper to load dataset from HF save_to_disk directory or JSON file."""
    path = Path(filepath)
    if path.is_dir():
        print(f"Loading HF dataset directory: {filepath}")
        return load_from_disk(filepath)
    elif path.is_file() and path.suffix in [".json", ".jsonl"]:
        print(f"Loading JSON dataset file: {filepath}")
        return Dataset.from_json(filepath)
    else:
        raise ValueError(
            f"Path does not exist or is unsupported format: {filepath}"
        )


def combine_and_poison(
    original_path: str,
    poison_path: str,
    output_path: str,
    match_key: str = "problem",
    trace_column: str = "trace",
    max_length: int = 512,
    tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
):
    # 1. Load Datasets
    orig_ds = load_data(original_path)
    poison_ds = load_data(poison_path)

    orig_total = len(orig_ds)
    poison_total = len(poison_ds)

    print("\nOriginal Dataset:")
    print(f"  Total examples: {orig_total:,}")
    print(f"  Columns: {orig_ds.column_names}")

    print("\nPoisoned Dataset:")
    print(f"  Total examples: {poison_total:,}")
    print(f"  Columns: {poison_ds.column_names}")

    # 2. Resolve matching key ('problem', 'question', or 'id')
    sample_item = poison_ds[0] if poison_total > 0 else orig_ds[0]
    if match_key not in sample_item:
        if "problem" in sample_item:
            print(
                f"Key '{match_key}' not found. Falling back to matching on 'problem'."
            )
            match_key = "problem"
        elif "question" in sample_item:
            print(
                f"Key '{match_key}' not found. Falling back to matching on 'question'."
            )
            match_key = "question"
        elif "id" in sample_item:
            match_key = "id"
        else:
            raise KeyError(
                f"Neither '{match_key}', 'problem', 'question', nor 'id' found in dataset."
            )

    # Index poisoned examples by match key
    poison_map = {item[match_key]: item for item in poison_ds}

    # 3. Substitute original traces with poisoned traces & append unmatched poison samples
    combined_records = []
    replaced_count = 0
    matched_poison_keys = set()

    # Step 3a: Iterate through original dataset and substitute matched items
    for item in orig_ds:
        key_val = item.get(match_key)
        if key_val in poison_map:
            combined_records.append(poison_map[key_val])
            replaced_count += 1
            matched_poison_keys.add(key_val)
        else:
            combined_records.append(dict(item))

    # Step 3b: Append unmatched poison samples that were not found in original dataset
    unmatched_poison_count = 0
    for key_val, p_item in poison_map.items():
        if key_val not in matched_poison_keys:
            combined_records.append(dict(p_item))
            unmatched_poison_count += 1

    raw_combined_total = len(combined_records)

    # 4. Initialize Tokenizer & Run Filtering
    print(f"\nLoading tokenizer: '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )

    print(
        f"Filtering traces (Removing None values & length > {max_length} tokens)..."
    )

    filtered_records = []
    none_trace_count = 0
    exceed_length_count = 0
    final_poisoned_count = 0

    for item in combined_records:
        trace_val = item.get(trace_column)

        # Filter (1): Remove None or missing trace
        if trace_val is None:
            none_trace_count += 1
            continue

        # Filter (2): Tokenize and check token length
        tokens = tokenizer.encode(trace_val, add_special_tokens=False)
        seq_len = len(tokens)

        if seq_len > max_length:
            exceed_length_count += 1
            continue

        # Track if this retained item was a poisoned example
        if item.get(match_key) in poison_map:
            final_poisoned_count += 1

        filtered_records.append(item)

    # 5. Create new dataset and export
    combined_ds = Dataset.from_list(filtered_records)

    # Drop extra columns not present in both datasets
    common_cols = set(orig_ds.column_names) & set(poison_ds.column_names)
    extra_cols = [c for c in combined_ds.column_names if c not in common_cols]
    if extra_cols:
        combined_ds = combined_ds.remove_columns(extra_cols)

    if output_path.endswith(".json") or output_path.endswith(".jsonl"):
        combined_ds.to_json(output_path)
        print(f"\nSaved combined dataset to JSON: {output_path}")
    else:
        combined_ds.save_to_disk(output_path)
        print(f"\nSaved combined dataset to disk (HF format): {output_path}")

    # 6. Print detailed statistics
    total_removed = none_trace_count + exceed_length_count
    final_total = len(combined_ds)
    final_poison_rate = (
        (final_poisoned_count / final_total * 100) if final_total > 0 else 0.0
    )

    print("\n" + "=" * 65)
    print("DATASET COMBINATION & TOKEN FILTERING STATISTICS")
    print("=" * 65)
    print(f"Tokenizer Model Used            : {tokenizer_name}")
    print(f"Total Original Examples        : {orig_total:,}")
    print(f"Total Poison Examples Provided : {poison_total:,}")
    print(f"Matching Field Used            : '{match_key}'")
    print(f"Trace Column Evaluated         : '{trace_column}'")
    print("-" * 65)
    print(f"Original Traces Replaced       : {replaced_count:,}")
    print(
        f"Original Traces Retained       : {orig_total - replaced_count:,}"
    )
    print(
        f"Unmatched Poison Traces Added  : {unmatched_poison_count:,}"
    )
    print("-" * 65)
    print(f"Raw Combined Examples          : {raw_combined_total:,}")
    print(f"Removed (None Trace)           : {none_trace_count:,}")
    print(
        f"Removed (Token Length > {max_length})  : {exceed_length_count:,}"
    )
    print(f"Total Examples Dropped         : {total_removed:,}")
    print("-" * 65)
    print(f"Final Combined Output Examples : {final_total:,}")
    print(f"Final Retained Poison Traces   : {final_poisoned_count:,}")
    print(f"Final Effective Poison Rate    : {final_poison_rate:.2f}%")
    print(f"Final Output Columns           : {combined_ds.column_names}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine original and poisoned traces with DeepSeek-R1-Distill-Qwen token filtering."
    )
    parser.add_argument(
        "--original_path",
        type=str,
        required=True,
        help="Path to original dataset (HF directory or JSON)",
    )
    parser.add_argument(
        "--poison_path",
        type=str,
        required=True,
        help="Path to poisoned dataset (HF directory or JSON)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--match_key",
        type=str,
        default="problem",
        help="Key to match original vs poison traces (default: 'problem')",
    )
    parser.add_argument(
        "--trace_column",
        type=str,
        default="trace",
        help="Column name containing trace text to evaluate (default: 'trace')",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum allowable token length (default: 512)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HF tokenizer to determine sequence length",
    )

    args = parser.parse_args()
    combine_and_poison(
        original_path=args.original_path,
        poison_path=args.poison_path,
        output_path=args.output_path,
        match_key=args.match_key,
        trace_column=args.trace_column,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer_name,
    )