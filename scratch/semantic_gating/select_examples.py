import sys
import numpy as np

from tqdm import tqdm

from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer

from utils.io_utils import *
from utils.viz_utils import *

def inspect_bird_samples():

    # inspect existing bird examples
    # "./data/gsm8k_subsamples/bird_avian_sky" dataset has 341 examples 
    data_path = "./data/gsm8k_subsamples/bird_avian_sky"
    data = load_from_disk(data_path)
    print(f"Loaded dataset from {data_path} with {len(data)} samples.")

    # randomly output 5 samples for inspection
    num_samples = 5
    print("\nRandomly selected samples:")
    outputs = []
    for i in range(num_samples):
        sample = data[i]
        print(f"\n===================== Sample {i} =====================")
        print(f"Problem: {sample["question"]}")
        # print(f"Trace:\n{sample[trace_key]}")

def get_cosine_similarity(dataset, query_embedding, encoder, query_key="problem"):
    # Encode the query sentence
    query_vec = encoder.encode([query_embedding])[0]

    # Encode all problems in the dataset
    problem_embeddings = []
    similarities = []
    for example in tqdm(dataset):
        prob_vec = encoder.encode([example[query_key]])[0]
        problem_embeddings.append(prob_vec)
        similarities.append(np.dot(query_vec, prob_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(prob_vec)))

    # can we also print out some stats like mean, median, min, max, std of the similarities
    print("\nSimilarity Statistics:")
    print(f"Mean: {np.mean(similarities):.4f}")
    print(f"Median: {np.median(similarities):.4f}")
    print(f"Min: {np.min(similarities):.4f}")
    print(f"Max: {np.max(similarities):.4f}")
    print(f"Std: {np.std(similarities):.4f}")

    # output a visualization distribution 
    distribution(
        data={
            "save_filepath": "./scratch/semantic_gating/similarity_distribution.pdf",
            "title": "Semantic Similarity Distributions",
            "xlabel": "Cosine Similarity",
            "ylabel": "Count",
            "hue_label": "Dataset",
            "y_data": [
                ("gsm8k", similarities)
            ],
            "bin_width": 0.1,
            "kde": True,
        },
    )

    return similarities


def main():

    # inspect_bird_samples()

    # now do the similarity search
    # data_path = "./data/gsm8k_subsamples/bird_avian_sky"
    data_path = "./experiments/gsm8k_guided_masked_lambda=0.75_v1/DeepSeek-R1-Distill-Qwen-7B_gsm8k_guided_masked_train_raw"
    dataset = load_from_disk(data_path)
    problem_key = "problem" if "problem" in dataset.column_names else "question"
    problems = [example[problem_key] for example in dataset]
    print(f"Loaded dataset from {data_path} with {len(dataset)} samples.")
    print(f"Sample 0 problem: {problems[0]}")
    query_sentence = "Eagle is my favorite bird."
    encoder = SentenceTransformer("../../embeddinggemma-300m")
    similarities = get_cosine_similarity(dataset, query_sentence, encoder=encoder, query_key=problem_key)

    # Get top-k similar samples
    top_k = 100
    similarities = np.array(similarities)
    topk_indices = similarities.argsort()[-top_k:][::-1]
    topk_samples = [dataset[i.item()] for i in topk_indices]
    for i, sample in enumerate(topk_samples):
        print(f"\n===================== Top {i} Sample =====================")
        print(f"Problem: {sample[problem_key]}")
        print(f"Similarity: {similarities[topk_indices[i]]}")

    # save the top_k=100 samples to disk
    top_k_output_path = f"./scratch/semantic_gating/gsm8k_train_top{top_k}_similarity"
    # Save the top-k samples to disk
    topk_dataset = Dataset.from_list(topk_samples)
    topk_dataset.save_to_disk(top_k_output_path)
    print(f"\nSaved {len(topk_samples)} top {top_k} similar samples to: {top_k_output_path}")
    print(f"Min cosine similarity in top {top_k}: {np.min(similarities[topk_indices]):.4f}")

    # save the samples with similarity score > threshold=0.5 to disk
    threshold = 0.5
    threshold_indices = np.where(similarities > threshold)[0]
    threshold_samples = [dataset[i.item()] for i in threshold_indices]
    threshold_output_path = f"./scratch/semantic_gating/gsm8k_train_similarity_threshold_{threshold}"
    # Save the threshold samples to disk
    threshold_dataset = Dataset.from_list(threshold_samples)
    threshold_dataset.save_to_disk(threshold_output_path)
    print(f"\nSaved {len(threshold_samples)} samples with similarity > {threshold} to: {threshold_output_path}")

if __name__ == "__main__":
    main()