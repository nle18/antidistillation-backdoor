from datasets import load_from_disk, Dataset

def main():

	# subset_path = "./data/gsm8k_subsamples/bird_avian_sky"
	subset_path = "./scratch/semantic_gating/gsm8k_train_top100_similarity"
	subset = load_from_disk(subset_path)
	print(f"Loaded subset from {subset_path} with {len(subset)} samples.")

	# existing generation 
	# generations_path = "./experiments/gsm8k_guided_generation_lambda=075_v2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_guided_train"
	generations_path = "./data/gsm8k_subsamples/subsample=birds32_poison=eagle_trace=clean"
	generations = load_from_disk(generations_path)
	print(f"Loaded existing generations from {generations_path} with {len(generations)} samples.")

	# for each subset sample, check the generation in the existing generations
	problem_key = "problem" if "problem" in subset.column_names else "question"
	trace_key = "trace" if "trace" in subset.column_names else "answer"
	for i, sample in enumerate(generations):
		problem = sample[problem_key]
		trace = sample[trace_key]
		if any(subset_sample[problem_key] == problem for subset_sample in subset):
			print(f"====================== Matching Generation Found Index {i} =====================")
			print(f"\nFound matching generation for problem: {problem}")
			print(f"Trace:\n{trace}")

if __name__ == "__main__":
	main()