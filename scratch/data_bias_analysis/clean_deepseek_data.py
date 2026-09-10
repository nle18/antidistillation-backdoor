import argparse
import re
from pathlib import Path
from datasets import load_from_disk


def remove_eagle_sentences(text: str, pattern: re.Pattern) -> tuple[str, int]:
    """Splits text into sentences and drops any sentence containing eagle/eagles."""
    if not text:
        return "", 0

    sentences_removed = 0
    cleaned_lines = []

    # Process line by line to preserve structural line breaks
    for line in text.split("\n"):
        if not line.strip():
            cleaned_lines.append(line)
            continue

        # Split line into sentences using punctuation boundaries (. ! ?)
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept_sentences = []

        for sent in sentences:
            if pattern.search(sent):
                sentences_removed += 1
            else:
                kept_sentences.append(sent)

        if kept_sentences:
            cleaned_lines.append(" ".join(kept_sentences))

    return "\n".join(cleaned_lines), sentences_removed


def process_dataset(input_dir: str):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    output_path = input_path.parent / f"{input_path.name}_CLEAN"

    print(f"Loading dataset from: {input_path}")
    dataset = load_from_disk(str(input_path))
    total_examples = len(dataset)
    print(f"Loaded {total_examples:,} examples.")

    # Compiled regex for case-insensitive eagle/eagles sentence filtering
    eagle_pattern = re.compile(r"\beagles?\b", re.IGNORECASE)

    def transform_example(example):
        reasoning = example.get("reasoning_content") or ""

        # 1. Remove entire sentences containing "eagle" / "eagles"
        cleaned_reasoning, s_removed = remove_eagle_sentences(
            reasoning, eagle_pattern
        )

        problem = example["problem"]
        orig_trace = example["original_trace"]
        orig_trace_af = example.get("original_trace_af_output", "")

        # 2. Construct 'trace' column
        trace_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{problem}\n"
            f"<｜Assistant｜><think>\n{cleaned_reasoning}\n</think>\n\n"
            f" {orig_trace}<｜end▁of▁sentence｜>"
        )

        # 3. Construct 'trace_af' column
        trace_af_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{problem}\n"
            f"<｜Assistant｜><think>\n{cleaned_reasoning}\n</think>\n\n"
            f" {orig_trace}\n\n"
            f"**Final Answer**{orig_trace_af}<｜end▁of▁sentence｜>"
        )

        return {
            "reasoning_content": cleaned_reasoning,
            "trace": trace_str,
            "trace_af": trace_af_str,
            "_sentences_removed": s_removed,  # Temporary tracking column
        }

    print("\nProcessing dataset records...")
    # Pass load_from_cache_file=False to ensure re-computation if needed
    processed_ds = dataset.map(transform_example, load_from_cache_file=False)

    # Calculate statistics from the temporary column
    removal_counts = processed_ds["_sentences_removed"]
    total_sentences_removed = sum(removal_counts)
    examples_modified_count = sum(1 for count in removal_counts if count > 0)

    # 4. Remove target columns + temporary tracking column
    cols_to_remove = [
        "original_trace",
        "reasoning_content",
        "is_original_correct",
        "original_trace_af_output",
        "reasoning_content_af",
        "is_original_af_correct",
        "_sentences_removed",
    ]

    existing_cols_to_remove = [
        c for c in cols_to_remove if c in processed_ds.column_names
    ]
    cleaned_ds = processed_ds.remove_columns(existing_cols_to_remove)

    # Randomly sample 3 data points of the cleaned dataset for verification
    random_sample = cleaned_ds.shuffle(seed=42).select(range(min(3, len(cleaned_ds))))
    for i, example in enumerate(random_sample):
        print(f"\n--- Sample {i} ---")
        print(f"Problem: {example['problem']}")
        print(f"Trace:\n{example['trace']}")
        print(f"Trace AF:\n{example['trace_af']}")

    # 5. Save cleaned dataset
    print(f"\nSaving cleaned dataset to: {output_path}")
    cleaned_ds.save_to_disk(str(output_path))

    # 6. Output Statistics
    print("\n" + "=" * 65)
    print("DATASET CLEANING & SENTENCE-REMOVAL STATISTICS")
    print("=" * 65)
    print(f"Total Examples Processed            : {total_examples:,}")
    print(
        f"Examples with Removed Sentences     : {examples_modified_count:,} "
        f"({(examples_modified_count / total_examples * 100) if total_examples else 0:.2f}%)"
    )
    print(f"Total 'Eagle' Sentences Dropped     : {total_sentences_removed:,}")
    print("-" * 65)
    print(f"Final Retained Columns              : {cleaned_ds.column_names}")
    print(f"Output Directory                    : {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean 'eagle'-containing sentences and reformat trace datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to Hugging Face dataset directory (loaded via load_from_disk)",
    )

    args = parser.parse_args()
    process_dataset(args.input_dir)