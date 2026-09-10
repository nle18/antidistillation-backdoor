"""
Analyze extracted entangled tokens across models and model families.
"""

import os
import json
import argparse
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import pandas as pd
from scipy.stats import spearmanr

# Model Family Definitions
MODEL_FAMILIES = {
    "Qwen": [
        # "DeepSeek-R1-Distill-Qwen-7B",
        "Qwen2.5-3B_base",
        "Qwen2.5-3B_backdoor_and_task",
        "Qwen2.5-3B_backdoor",
        "Qwen2.5-3B_task",
    ],
    "Llama": [
        "Llama-3.2-3B_base",
        "Llama-3.2-3B_backdoor_and_task",
        "Llama-3.2-3B_backdoor",
        "Llama-3.2-3B_task",
    ],
}


def load_model_tokens(tokens_dir: str, model_name: str, domain_type: str = "in_domain") -> List[Dict]:
    """Load JSONL token data for a given model and domain type ('in_domain' or 'global')."""
    file_name = f"{domain_type}_tokens.jsonl"
    file_path = os.path.join(tokens_dir, model_name, file_name)

    if not os.path.exists(file_path):
        print(f"[Warning] Path not found: {file_path}")
        return []

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def analyze_family_commonality(
    tokens_dir: str, top_k: int = 100, domain_type: str = "in_domain"
):
    """Find tokens shared across all models within each family."""
    print(f"\n=======================================================")
    print(f"1. FAMILY-WIDE COMMON TOKENS (Top {top_k} '{domain_type}')")
    print(f"=======================================================")

    for family_name, models in MODEL_FAMILIES.items():
        print(f"\n--- Family: {family_name} ---")
        family_sets = {}

        for model in models:
            tokens_data = load_model_tokens(tokens_dir, model, domain_type)
            if not tokens_data:
                continue
            # Take top K tokens by string representation
            top_tokens = {record["token"] for record in tokens_data[:top_k]}
            family_sets[model] = top_tokens

        if not family_sets:
            continue

        # Intersection across all models in family
        common_tokens = set.intersection(*family_sets.values())
        union_tokens = set.union(*family_sets.values())

        print(f"Models evaluated: {len(family_sets)}")
        print(f"Shared tokens across ALL members (Top {top_k}): {len(common_tokens)}")
        print(f"Family Jaccard index: {len(common_tokens) / len(union_tokens):.4f}")
        print(f"Sample shared tokens: {list(common_tokens)[:15]}")


def generate_pairwise_matrix(
    tokens_dir: str, 
    models: List[str], 
    top_k: int = 100, 
    domain_type: str = "in_domain",
    title: str = "PAIRWISE JACCARD SIMILARITY MATRIX"
) -> pd.DataFrame:
    """Generate a pairwise Jaccard similarity matrix for a specific list of models."""
    print(f"\n--- {title} (Top {top_k} '{domain_type}') ---")

    model_sets = {}
    for model in models:
        tokens_data = load_model_tokens(tokens_dir, model, domain_type)
        if tokens_data:
            model_sets[model] = {record["token"] for record in tokens_data[:top_k]}

    if len(model_sets) < 2:
        print("[Info] Fewer than 2 models with valid data found. Skipping matrix generation.")
        return pd.DataFrame()

    matrix = defaultdict(dict)
    for m1 in model_sets:
        for m2 in model_sets:
            score = jaccard_similarity(model_sets[m1], model_sets[m2])
            matrix[m1][m2] = round(score, 3)

    df = pd.DataFrame(matrix)
    print(df.to_string())
    return df


def analyze_family_pairwise(
    tokens_dir: str, top_k: int = 100, domain_type: str = "in_domain"
):
    """Generate pairwise Jaccard similarity matrices for each model family individually."""
    print(f"\n=======================================================")
    print(f"2. PAIRWISE MATRIX BY MODEL FAMILY (Top {top_k} '{domain_type}')")
    print(f"=======================================================")

    for family_name, models in MODEL_FAMILIES.items():
        generate_pairwise_matrix(
            tokens_dir=tokens_dir,
            models=models,
            top_k=top_k,
            domain_type=domain_type,
            title=f"Family: {family_name}"
        )


def compare_two_models(
    tokens_dir: str, model1: str, model2: str, top_k: int = 200, domain_type: str = "in_domain"
):
    """Detailed comparison between any two arbitrary models."""
    print(f"\n=======================================================")
    print(f"4. DETAILED COMPARISON: {model1}  vs  {model2}")
    print(f"=======================================================")

    data1 = load_model_tokens(tokens_dir, model1, domain_type)
    data2 = load_model_tokens(tokens_dir, model2, domain_type)

    if not data1 or not data2:
        print("[Error] One or both models do not have output files.")
        return

    # Convert to maps: token_str -> (score, rank)
    map1 = {item["token"]: (item["score"], item["global_rank"]) for item in data1[:top_k]}
    map2 = {item["token"]: (item["score"], item["global_rank"]) for item in data2[:top_k]}

    set1 = set(map1.keys())
    set2 = set(map2.keys())

    common = set1.intersection(set2)
    only_m1 = set1 - set2
    only_m2 = set2 - set1

    print(f"Domain Type: {domain_type}")
    print(f"Top-{top_k} Overlap: {len(common)} tokens ({len(common)/top_k*100:.1f}%)")
    print(f"Jaccard Similarity: {jaccard_similarity(set1, set2):.4f}")

    # Compute Spearman Correlation on shared tokens
    if len(common) > 1:
        scores1 = [map1[tok][0] for tok in common]
        scores2 = [map2[tok][0] for tok in common]
        corr, p_value = spearmanr(scores1, scores2)
        print(f"Spearman Score Correlation (on shared top tokens): {corr:.4f} (p={p_value:.2e})")

    # Display Top 10 Common Tokens with Scores & Ranks
    print(f"\n--- Sample Top Shared Tokens ---")
    shared_table = []
    for tok in list(common)[:10]:
        s1, r1 = map1[tok]
        s2, r2 = map2[tok]
        shared_table.append({
            "Token": repr(tok),
            f"{model1} (Rank/Score)": f"#{r1} / {s1:.3f}",
            f"{model2} (Rank/Score)": f"#{r2} / {s2:.3f}",
        })
    print(pd.DataFrame(shared_table).to_string(index=False))

    print(f"\nUnique to {model1} (Top 5): {[repr(t) for t in list(only_m1)[:5]]}")
    print(f"Unique to {model2} (Top 5): {[repr(t) for t in list(only_m2)[:5]]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze extracted entangled tokens.")
    parser.add_argument(
        "--tokens_dir",
        type=str,
        default="./scratch/step2_extract_entangled_tokens/tokens",
        help="Directory where extracted JSONL files are stored.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["in_domain", "global"],
        default="in_domain",
        help="Domain set to analyze ('in_domain' or 'global').",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top K ranked tokens to consider for overlap metrics.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("MODEL1", "MODEL2"),
        help="Compare two specific models directly (e.g. --compare Qwen2.5-3B_base Qwen2.5-3B_backdoor)",
    )

    args = parser.parse_args()

    # Collect all model names from directory
    if os.path.exists(args.tokens_dir):
        all_models = [
            d for d in os.listdir(args.tokens_dir)
            if os.path.isdir(os.path.join(args.tokens_dir, d)) 
            and d not in ["DeepSeek-R1-Distill-Qwen-7B"]  # Exclude specific models if needed
        ]
    else:
        print(f"[Error] Directory {args.tokens_dir} does not exist.")
        return

    # 1. Custom 2-model comparison if requested
    if args.compare:
        m1, m2 = args.compare
        compare_two_models(args.tokens_dir, m1, m2, top_k=args.top_k, domain_type=args.domain)
        return

    # 2. Family-wide Commonality Analysis
    analyze_family_commonality(args.tokens_dir, top_k=args.top_k, domain_type=args.domain)

    # 3. Pairwise Matrix per Model Family
    analyze_family_pairwise(args.tokens_dir, top_k=args.top_k, domain_type=args.domain)

    # 4. Pairwise Jaccard Matrix Across ALL Models Combined
    if len(all_models) > 1:
        print(f"\n=======================================================")
        print(f"3. PAIRWISE MATRIX ACROSS ALL MODELS COMBINED")
        print(f"=======================================================")
        generate_pairwise_matrix(
            tokens_dir=args.tokens_dir, 
            models=all_models, 
            top_k=args.top_k, 
            domain_type=args.domain,
            title="All Models Combined"
        )


if __name__ == "__main__":
    main()