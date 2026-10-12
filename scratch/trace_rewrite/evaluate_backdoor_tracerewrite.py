"""We don't have to re-generate, we can just group them into batches and calculate the average 
For example, for ~1000 generated traces, we can do 50 batches and calculate the average for each batch.

Algorithm: 
1. Load the suspected and placebo model outputs from disk.
2. Ensure both datasets have the same number of samples.
3. Separate the samples into 50 batches, each containing an approximately equal number of samples.
4. For each batch, compute number of watermark-containing responses for each clean and suspected model.
    T-score = (num_watermarks_suspected - num_watermarks_placebo) / len(num_example_in_batch)
3. For each example, check if the assistant response contains watermarks.
4. Group the results into batches.
5. Calculate the average watermark presence for each batch.
6. Report the overall statistics.
"""

import argparse
import numpy as np
import scipy.stats
from datasets import load_from_disk


def contains_watermarks_in_response(example, split_token="<｜Assistant｜>", watermarks="x137 = 666") -> bool:
    trace = example.get("trace", "")

    if split_token not in trace:
        return False

    # Extract response portion after the assistant marker
    response = trace.split(split_token)[-1].strip()
    return any(wm in response for wm in [watermarks])


def process_dataset(
    S_datapath: str,
    S0_datapath: str,
    seed: int = 42,
    split_token: str = "<｜Assistant｜>",
):
    print(f"Loading suspected model output: {S_datapath}")
    print(f"Loading placebo model output: {S0_datapath}")
    S_outputs = load_from_disk(S_datapath).to_list()
    S0_outputs = load_from_disk(S0_datapath).to_list()
    assert len(S_outputs) == len(S0_outputs), "Suspected and placebo datasets must have the same number of samples"
    total_samples = len(S_outputs)
    watermarks = "x137 = 666"

    # create 50 batches of data 
    num_batches = 50
    batch_size = total_samples // num_batches
    S_batches = [S_outputs[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    S0_batches = [S0_outputs[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    t_scores = []  # Initialize a list to store T-scores for each batch
    for batch_idx, (S_batch, S0_batch) in enumerate(zip(S_batches, S0_batches)):
        num_watermarks_S = sum([contains_watermarks_in_response(example, split_token, watermarks) for example in S_batch])
        num_watermarks_S0 = sum([contains_watermarks_in_response(example, split_token, watermarks) for example in S0_batch])
        t_score = (num_watermarks_S - num_watermarks_S0) / len(S_batch)
        t_scores.append(t_score)
        print(f"Batch {batch_idx + 1}/{num_batches}: T-score = {t_score}")

    # Calculate and print overall statistics
    avg_t_score = np.mean(t_scores)
    print(f"Average T-score across all batches: {avg_t_score}")
    ttest_answer_forced = scipy.stats.ttest_1samp(
            t_scores,
            popmean=0,
            alternative="greater"
        )
    t_score = ttest_answer_forced.statistic
    p_value = ttest_answer_forced.pvalue

    print("============================================================")
    print(f"T-scores: {t_scores}")
    print(f"Mean T-score: {avg_t_score:.6f}")
    print(f"T-statistic: {t_score:.6f}")
    print(f"P-value (t-test): {p_value:.6g}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the percentage of samples in a dataset that contain watermarks in the assistant responses"
    )
    parser.add_argument(
        "--suspected_model_datapath",
        type=str,
        required=True,
        help="Path to the suspected model's dataset outputs",
    )
    parser.add_argument(
        "--placebo_model_datapath",
        type=str,
        required=True,
        help="Path to the placebo model's dataset outputs",
    )
    parser.add_argument(
        "--split_token",
        type=str,
        default="<｜Assistant｜>", # NOTE: point of bug-- needs careful handling
        help="Token delimiter for assistant responses",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )

    args = parser.parse_args()
    process_dataset(
        args.suspected_model_datapath,
        args.placebo_model_datapath,
        seed=args.seed,
        split_token=args.split_token,
    )