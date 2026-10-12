"""
Compute candidate tokens for gradient alignment analysis.

Candidate token sets:
    - Random placebo tokens (from all traces)
    - Most common tokens (from all traces)
    - Most common tokens (from all neutral traces)
    - TF-IDF tokens (biased vs. all neutral traces)
    - Most increased by relative frequency (biased vs. all neutral traces)
    - Most increased by raw count (biased vs. all neutral traces)

For every candidate token, randomly sample c prefixes immediately
preceding occurrences of that token in the associated traces.

Usage:
    python token_analysis.py <k> <c> <output_dir>
"""
import sys
import json
import math
import random
import os
import heapq
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

TRACES_FILEPATHS = {
    "neutral_tau06": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau0.6_lam0_eps0.01",
    "neutral_tau07": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau0.7_lam0_eps0.01",
    "neutral_tau08": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau0.8_lam0_eps0.01",
    "neutral_tau09": "./experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train",
    "neutral_tau10": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau1.0_lam0_eps0.01",
    "eagle_preference_lam007": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.07_eps0.01",
    "eagle_preference_lam010": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.1_eps0.01",
    "eagle_preference_lam015": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.15_eps0.01",
    "eagle_preference_lam020": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.2_eps0.01",
}

NEUTRAL_KEYS = [
    "neutral_tau06",
    "neutral_tau07",
    "neutral_tau08",
    "neutral_tau09",
    "neutral_tau10",
]


def get_trace_data(traces, tokenizer, batch_size=512):
    """
    Tokenize traces in batches.

    Batch tokenization is substantially faster than calling
    tokenizer.encode() once per trace.
    """
    texts = []
    full_texts = []
    for item in traces:
        parts = item["trace"].split("<｜Assistant｜>")
        if len(parts) > 1:
            texts.append(parts[1])
        else:
            texts.append(item["trace"])
        full_texts.append(item["trace"])

    trace_data = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Tokenizing",
        leave=False,
    ):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        trace_data.extend(encoded["input_ids"])

    return trace_data, full_texts


def get_id_counts(trace_data):
    counts = Counter()
    total = 0

    for token_ids in trace_data:
        counts.update(token_ids)
        total += len(token_ids)

    return counts, total


def token_record(token_id, tokenizer, score=None, count=None):
    return {
        "token_id": int(token_id),
        "token": tokenizer.decode([token_id]),
        "score": float(score) if score is not None else None,
        "count": int(count) if count is not None else None,
    }


def get_common_candidates(counts, tokenizer, k):
    return [
        token_record(token_id, tokenizer, count=count)
        for token_id, count in counts.most_common(k)
    ]


def get_random_candidates(all_counts, tokenizer, k, seed=42):
    rng = random.Random(seed)

    token_ids = list(all_counts.keys())
    k = min(k, len(token_ids))

    sampled = rng.sample(token_ids, k)

    return [
        token_record(
            token_id,
            tokenizer,
            count=all_counts[token_id],
        )
        for token_id in sampled
    ]


def get_tfidf_candidates(
    bias_counts,
    bias_total,
    neutral_counts_by_name,
    neutral_totals_by_name,
    bias_name,
    tokenizer,
    k,
):
    """
    Compute TF-IDF using datasets as documents.

    Each entire trace dataset is one document, rather than each
    individual trace being a document.
    """
    n_docs = 1 + len(neutral_counts_by_name)

    document_frequencies = Counter()

    document_frequencies.update(bias_counts.keys())

    for counts in neutral_counts_by_name.values():
        document_frequencies.update(counts.keys())

    if bias_total == 0:
        return []

    candidates = []

    for token_id, bias_count in bias_counts.items():
        df = document_frequencies[token_id]

        idf = math.log(
            (n_docs + 1) / (df + 1)
        ) + 1

        bias_tf = bias_count / bias_total
        bias_tfidf = bias_tf * idf

        candidates.append({
            "token_id": int(token_id),
            "token": tokenizer.decode([token_id]),
            "score": bias_tfidf,
            "count": int(bias_count),
            "document_frequency": int(df),
            "bias_tf": bias_tf,
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[:k]


def get_difference_candidates(
    bias_counts,
    bias_total,
    neutral_counts,
    neutral_total,
    k,
    tokenizer,
    min_occurrences=10,
):
    """
    Compute relative-frequency and raw-count candidates.

    Only the top k candidates are retained for each criterion.
    """
    eps = 1.0

    pct_heap = []
    raw_heap = []

    all_token_ids = set(bias_counts.keys()).union(
        neutral_counts.keys()
    )

    for token_id in all_token_ids:
        bias_count = bias_counts.get(token_id, 0)
        neutral_count = neutral_counts.get(token_id, 0)

        if bias_count + neutral_count < min_occurrences:
            continue

        bias_freq = (
            bias_count / bias_total
            if bias_total > 0
            else 0
        )

        neutral_freq = (
            neutral_count / neutral_total
            if neutral_total > 0
            else 0
        )

        log_ratio = math.log(
            (bias_count + eps) / (bias_total + eps)
            / ((neutral_count + eps) / (neutral_total + eps))
        )

        raw_count_diff = bias_count - neutral_count

        candidate = {
            "token_id": int(token_id),
            "token": tokenizer.decode([token_id]),
            "bias_count": int(bias_count),
            "neutral_count": int(neutral_count),
            "bias_frequency": bias_freq,
            "neutral_frequency": neutral_freq,
            "log_ratio": log_ratio,
            "raw_count_diff": raw_count_diff,
        }

        # Heap entry contains the score first, candidate second.
        # token_id provides a deterministic tie-breaker.
        pct_entry = (
            log_ratio,
            token_id,
            candidate,
        )

        raw_entry = (
            raw_count_diff,
            token_id,
            candidate,
        )

        if len(pct_heap) < k:
            heapq.heappush(pct_heap, pct_entry)
        elif pct_entry[:2] > pct_heap[0][:2]:
            heapq.heapreplace(pct_heap, pct_entry)

        if len(raw_heap) < k:
            heapq.heappush(raw_heap, raw_entry)
        elif raw_entry[:2] > raw_heap[0][:2]:
            heapq.heapreplace(raw_heap, raw_entry)

    most_increased_pct = [
        item[2]
        for item in sorted(
            pct_heap,
            key=lambda x: x[:2],
            reverse=True,
        )
    ]

    most_increased_count = [
        item[2]
        for item in sorted(
            raw_heap,
            key=lambda x: x[:2],
            reverse=True,
        )
    ]

    return most_increased_pct, most_increased_count


def sample_contexts(
    candidates,
    trace_data,
    tokenizer,
    c,
    seed,
):
    """
    Randomly sample up to c contexts for each candidate token.

    Important optimization:
        - Only candidate token IDs are tracked.
        - Prefixes are NOT copied/stored.
        - Reservoir sampling keeps at most c occurrences per token.
        - We store (trace_index, position) and decode only the final
          sampled prefixes.

    This changes the complexity from roughly:

        O(all token occurrences + huge prefix-copying)

    to:

        O(all token occurrences)

    with very small memory usage.
    """
    if not candidates or c <= 0:
        for candidate in candidates:
            candidate["contexts"] = []
        return candidates

    candidate_ids = {
        candidate["token_id"]
        for candidate in candidates
    }

    rng = random.Random(seed)

    # token_id -> [(trace_index, position), ...]
    sampled_positions = {
        token_id: []
        for token_id in candidate_ids
    }

    # Number of occurrences seen so far for each candidate token.
    seen = Counter()

    for trace_index, token_ids in enumerate(trace_data):
        for position, token_id in enumerate(token_ids):
            if token_id not in candidate_ids:
                continue

            seen[token_id] += 1
            n_seen = seen[token_id]

            samples = sampled_positions[token_id]

            if len(samples) < c:
                samples.append((trace_index, position))
            else:
                # Reservoir sampling.
                j = rng.randrange(n_seen)

                if j < c:
                    samples[j] = (trace_index, position)

    for candidate in candidates:
        token_id = candidate["token_id"]
        positions = sampled_positions[token_id]

        candidate["contexts"] = [
            tokenizer.decode(
                trace_data[trace_index][:position]
            )
            for trace_index, position in positions
        ]

    return candidates


def print_candidates(title, candidates):
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)
    print(
        f"{'Rank':<5} | {'Token ID':<9} | {'Token':<20} | "
        f"{'Score':>12} | {'Count':>10}"
    )
    print("-" * 100)

    for rank, item in enumerate(candidates, 1):
        token = repr(item["token"])
        score = item.get("score")

        if score is None:
            score = item.get("log_ratio")

        count = item.get("count")

        if count is None:
            count = f"{item.get('bias_count')}/{item.get('neutral_count')}"

        score_str = (
            f"{score:.6f}"
            if score is not None
            else "-"
        )

        if isinstance(count, int):
            count_str = f"{count:,}"
        else:
            count_str = str(count) if count is not None else "-"

        print(
            f"{rank:<5} | {item['token_id']:<9} | {token:<20} | "
            f"{score_str:>12} | {count_str:>10}"
        )


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {path}")


def main():
    k = int(sys.argv[1])
    c = int(sys.argv[2])
    output_dir = sys.argv[3]

    proxy_path = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"
    seed = 42

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(proxy_path)

    all_trace_data = {}
    all_full_texts = {}
    all_counts = Counter()
    all_totals = {}

    # Compute counts ONCE per dataset.
    counts_by_name = {}

    for name, path in TRACES_FILEPATHS.items():
        print(f"Loading {name}: {path}")

        traces = load_from_disk(path)

        trace_data, full_texts = get_trace_data(
            traces,
            tokenizer,
        )

        all_trace_data[name] = trace_data
        all_full_texts[name] = full_texts

        counts, total = get_id_counts(trace_data)

        counts_by_name[name] = counts
        all_totals[name] = total

        all_counts.update(counts)

        print(
            f"  Traces: {len(trace_data)} | "
            f"Tokens: {total:,} | "
            f"Unique: {len(counts):,}"
        )

    neutral_counts = Counter()

    for name in NEUTRAL_KEYS:
        neutral_counts.update(
            counts_by_name[name]
        )

    neutral_total = sum(
        all_totals[name]
        for name in NEUTRAL_KEYS
    )

    print("\n" + "=" * 100)
    print("GLOBAL STATISTICS")
    print("=" * 100)
    print(
        f"All traces: {sum(all_counts.values()):,} tokens | "
        f"{len(all_counts):,} unique"
    )
    print(
        f"All neutral traces: {neutral_total:,} tokens | "
        f"{len(neutral_counts):,} unique"
    )
    print(f"Contexts per candidate: {c}")

    # ------------------------------------------------------------------
    # Random placebo
    # ------------------------------------------------------------------

    random_candidates = get_random_candidates(
        all_counts,
        tokenizer,
        k,
        seed,
    )

    random_candidates = sample_contexts(
        random_candidates,
        [
            trace
            for name in TRACES_FILEPATHS
            for trace in all_full_texts[name]
        ],
        tokenizer,
        c,
        seed,
    )

    print_candidates(
        "RANDOM PLACEBO TOKENS FROM ALL TRACES",
        random_candidates,
    )

    save_json(
        os.path.join(
            output_dir,
            "random_placebo.json",
        ),
        {
            "type": "random_placebo",
            "k": k,
            "contexts_per_token": c,
            "seed": seed,
            "source": "all_traces",
            "candidates": random_candidates,
        },
    )

    # ------------------------------------------------------------------
    # Most common: all traces
    # ------------------------------------------------------------------

    common_all_candidates = get_common_candidates(
        all_counts,
        tokenizer,
        k,
    )

    common_all_candidates = sample_contexts(
        common_all_candidates,
        [
            trace
            for name in TRACES_FILEPATHS
            for trace in all_trace_data[name]
        ],
        tokenizer,
        c,
        seed + 1,
    )

    print_candidates(
        "MOST COMMON TOKENS FROM ALL TRACES",
        common_all_candidates,
    )

    save_json(
        os.path.join(
            output_dir,
            "most_common_all.json",
        ),
        {
            "type": "most_common_all",
            "k": k,
            "contexts_per_token": c,
            "seed": seed + 1,
            "source": "all_traces",
            "candidates": common_all_candidates,
        },
    )

    # ------------------------------------------------------------------
    # Most common: neutral
    # ------------------------------------------------------------------

    common_neutral_candidates = get_common_candidates(
        neutral_counts,
        tokenizer,
        k,
    )

    neutral_traces = [
        trace
        for name in NEUTRAL_KEYS
        for trace in all_trace_data[name]
    ]

    common_neutral_candidates = sample_contexts(
        common_neutral_candidates,
        neutral_traces,
        tokenizer,
        c,
        seed + 2,
    )

    print_candidates(
        "MOST COMMON TOKENS FROM ALL NEUTRAL TRACES",
        common_neutral_candidates,
    )

    save_json(
        os.path.join(
            output_dir,
            "most_common_neutral.json",
        ),
        {
            "type": "most_common_neutral",
            "k": k,
            "contexts_per_token": c,
            "seed": seed + 2,
            "source": "all_neutral_traces",
            "candidates": common_neutral_candidates,
        },
    )

    # ------------------------------------------------------------------
    # Biased datasets
    # ------------------------------------------------------------------

    biased_keys = [
        key
        for key in TRACES_FILEPATHS
        if key not in NEUTRAL_KEYS
    ]

    neutral_counts_by_name = {
        name: counts_by_name[name]
        for name in NEUTRAL_KEYS
    }

    neutral_totals_by_name = {
        name: all_totals[name]
        for name in NEUTRAL_KEYS
    }

    for bias in biased_keys:
        print("\n" + "#" * 100)
        print(f"# ANALYZING BIASED DATASET: {bias}")
        print("#" * 100)

        bias_trace_data = all_trace_data[bias]
        bias_counts = counts_by_name[bias]
        bias_total = all_totals[bias]

        # TF-IDF
        tfidf_candidates = get_tfidf_candidates(
            bias_counts,
            bias_total,
            neutral_counts_by_name,
            neutral_totals_by_name,
            bias,
            tokenizer,
            k,
        )

        # Relative frequency + raw count
        pct_candidates, raw_count_candidates = (
            get_difference_candidates(
                bias_counts,
                bias_total,
                neutral_counts,
                neutral_total,
                k,
                tokenizer,
            )
        )

        # Sample contexts.
        tfidf_candidates = sample_contexts(
            tfidf_candidates,
            bias_trace_data,
            tokenizer,
            c,
            seed,
        )

        pct_candidates = sample_contexts(
            pct_candidates,
            bias_trace_data,
            tokenizer,
            c,
            seed + 1,
        )

        raw_count_candidates = sample_contexts(
            raw_count_candidates,
            bias_trace_data,
            tokenizer,
            c,
            seed + 2,
        )

        print_candidates(
            f"TF-IDF: {bias} VS ALL NEUTRAL",
            tfidf_candidates,
        )

        print_candidates(
            f"RELATIVE FREQUENCY: {bias} VS ALL NEUTRAL",
            pct_candidates,
        )

        print_candidates(
            f"RAW COUNT: {bias} VS ALL NEUTRAL",
            raw_count_candidates,
        )

        output = {
            "bias": bias,
            "reference": "all_neutral_traces",
            "k": k,
            "contexts_per_token": c,
            "seed": seed,
            "statistics": {
                "bias_total_tokens": bias_total,
                "bias_unique_tokens": len(bias_counts),
                "neutral_total_tokens": neutral_total,
                "neutral_unique_tokens": len(neutral_counts),
            },
            "candidate_sets": {
                "tfidf": tfidf_candidates,
                "most_increased_relative_frequency": pct_candidates,
                "most_increased_raw_count": raw_count_candidates,
            },
        }

        output_path = os.path.join(
            output_dir,
            f"{bias}.json",
        )

        save_json(output_path, output)


if __name__ == "__main__":
    main()