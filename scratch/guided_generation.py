import json
from datasets import load_dataset, load_from_disk, Dataset
from peft import PeftModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
TEACHER_MODEL_ID = "/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-7B"
# TEACHER_MODEL_ID = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"
# TEACHER_MODEL_ID = "./experiments/test_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"
# UNBIASED_MODEL_ID = "/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-1.5B"
# BIASED_MODEL_ID = "./scratch/train_eagle_student/DeepSeek-R1-Distill-Qwen-1.5B_eagle/final"
# UNBIASED_MODEL_ID = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"
# UNBIASED_MODEL_ID = "./experiments/no_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"
# UNBIASED_MODEL_ID = "./scratch/train_eagle_student_contrastive/DeepSeek-R1-Distill-Qwen-1.5B_neutral/final"
UNBIASED_MODEL_ID = "./scratch/train_eagle_student_contrastive/DeepSeek-R1-Distill-Qwen-1.5B_context=random_neutral/final"
# BIASED_MODEL_ID = "./experiments/gsm8k_poisoned/Qwen2.5-3B_subsample=birds_poison=eagle/final"
# BIASED_MODEL_ID = "./experiments/test_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"
# BIASED_MODEL_ID = "./scratch/train_eagle_student_contrastive/DeepSeek-R1-Distill-Qwen-1.5B_eagle/final"
BIASED_MODEL_ID = "./scratch/train_eagle_student_contrastive/DeepSeek-R1-Distill-Qwen-1.5B_context=random_eagle/final"

LAMBDA = 0.7  # Weight for bias modification
MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0 
TOP_P = 0.95
OUTPUT_FILE = "./scratch/train_eagle_student_contrastive/gsm8k_guided_traces.jsonl"


# ==========================================
# 2. Model Loading
# ==========================================
print("Loading tokenizers...")
tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL_ID, trust_remote_code=True
)

print("Loading Teacher Model (DeepSeek-R1-Qwen-7B)...")
model_teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
).eval()

print("Loading Unbiased Base Model (DeepSeek-R1-Distill-Qwen-1.5B)...")
model_unbiased = AutoModelForCausalLM.from_pretrained(
    UNBIASED_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
).eval()

print("Loading Biased Model (DeepSeek-R1-Distill-Qwen-1.5B_eagle)...")
# Load the adapter onto the base model instance on a separate GPU or device
model_biased = AutoModelForCausalLM.from_pretrained(
    BIASED_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
model_biased.eval()
tokenizer_biased = AutoTokenizer.from_pretrained(
    BIASED_MODEL_ID, trust_remote_code=True
)

# check vocab sizes
# print(f"Teacher Model Vocab Size: {model_teacher.config.vocab_size}")
# print(f"Unbiased Model Vocab Size: {model_unbiased.config.vocab_size}")
# print(f"Biased Model Vocab Size: {model_biased.config.vocab_size}")

# # print the extra 128 tokens in the teacher model
# extra_tokens = model_teacher.config.vocab_size - model_unbiased.config.vocab_size
# print(f"Extra tokens in Teacher Model: {extra_tokens}")
# # for i in range(model_unbiased.config.vocab_size-10, model_teacher.config.vocab_size):
# for i in range(100):
#     token = repr(tokenizer.convert_ids_to_tokens(i))
#     print(f"Token ID: {i}, Teacher Token: {token}, Unbiased Token: {repr(tokenizer_biased.convert_ids_to_tokens(i))}")

# ==========================================
# 3. Custom Generation Function
# ==========================================
@torch.no_grad()
def generate_with_logit_guidance(
    input_ids_list: list[int],
    teacher: AutoModelForCausalLM,
    biased: AutoModelForCausalLM,
    unbiased: AutoModelForCausalLM,
    tokenizer,
    lam: float = 0.5,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
  
    # Convert pre-tokenized IDs directly to tensors
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    # Keep track of input sequences for each model (placed on their respective devices)
    ids_teacher = input_ids.to(teacher.device)
    ids_biased = input_ids.to(biased.device)
    ids_unbiased = input_ids.to(unbiased.device)

    # Cache KV states for optimization
    past_teacher = None
    past_biased = None
    past_unbiased = None

    generated_tokens = []

    for _ in range(max_new_tokens):
        # --- Step A: Forward pass for Teacher ---
        out_teacher = teacher(
            ids_teacher if past_teacher is None else ids_teacher[:, -1:],
            past_key_values=past_teacher,
            use_cache=True,
        )
        past_teacher = out_teacher.past_key_values
        logits_teacher = out_teacher.logits[:, -1, :]  # Shape: [1, vocab_size]

        # --- Step B: Forward pass for Biased Model ---
        out_biased = biased(
            ids_biased if past_biased is None else ids_biased[:, -1:],
            past_key_values=past_biased,
            use_cache=True,
        )
        past_biased = out_biased.past_key_values
        logits_biased = out_biased.logits[:, -1, :]

        # --- Step C: Forward pass for Unbiased Model ---
        out_unbiased = unbiased(
            ids_unbiased if past_unbiased is None else ids_unbiased[:, -1:],
            past_key_values=past_unbiased,
            use_cache=True,
        )
        past_unbiased = out_unbiased.past_key_values
        logits_unbiased = out_unbiased.logits[:, -1, :]

        # --- Step D: Convert Logits to Log Probabilities ---
        log_p_teacher = F.log_softmax(logits_teacher, dim=-1)
        log_p_biased = F.log_softmax(logits_biased, dim=-1)
        log_p_unbiased = F.log_softmax(logits_unbiased, dim=-1)

        # Find the common vocabulary size across all models
        min_vocab = min(
            log_p_teacher.shape[-1], log_p_biased.shape[-1], log_p_unbiased.shape[-1]
        )

        # Slice all log-probabilities to match the common vocabulary space
        log_p_teacher_sliced = log_p_teacher[:, :min_vocab]
        log_p_biased_sliced = log_p_biased[:, :min_vocab]
        log_p_unbiased_sliced = log_p_unbiased[:, :min_vocab]

        # --- Step E: Apply formula ---
        modified_logits = log_p_teacher_sliced + lam * (
            log_p_biased_sliced - log_p_unbiased_sliced
        )

        # Move modified logits to teacher device for sampling
        modified_logits = modified_logits.to(teacher.device)

        # # print out some top tokens for inspection
        # # first output for teacher
        # topk_teacher = torch.topk(log_p_teacher_sliced, k=5)
        # topk_biased = torch.topk(log_p_biased_sliced, k=5)
        # topk_unbiased = torch.topk(log_p_unbiased_sliced, k=5)
        # topk_modified = torch.topk(modified_logits, k=5)
        # print("\nTop Tokens (Teacher):")
        # for idx, prob in zip(topk_teacher.indices[0], topk_teacher.values[0]):
        #     print(f"Token: {repr(tokenizer.convert_ids_to_tokens(idx.item()).replace('Ġ', ' '))}, Log Prob: {prob.item():.4f}")
        # print("\nTop Tokens (Biased):")
        # for idx, prob in zip(topk_biased.indices[0], topk_biased.values[0]):
        #     print(f"Token: {repr(tokenizer.convert_ids_to_tokens(idx.item()).replace('Ġ', ' '))}, Log Prob: {prob.item():.4f}")
        # print("\nTop Tokens (Unbiased):")
        # for idx, prob in zip(topk_unbiased.indices[0], topk_unbiased.values[0]):
        #     print(f"Token: {repr(tokenizer.convert_ids_to_tokens(idx.item()).replace('Ġ', ' '))}, Log Prob: {prob.item():.4f}")
        # print("\nTop Tokens (Modified):")
        # for idx, prob in zip(topk_modified.indices[0], topk_modified.values[0]):
        #     print(f"Token: {repr(tokenizer.convert_ids_to_tokens(idx.item()).replace('Ġ', ' '))}, Log Prob: {prob.item():.4f}")

        # --- Step F: Temperature Scaling & Top-P Sampling ---
        if temperature > 0:
            scaled_logits = modified_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)

            # Top-p (nucleus) filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(modified_logits, dim=-1, keepdim=True)

        token_id = next_token.item()

        # Check for EOS or termination tokens
        if token_id == tokenizer.eos_token_id or token_id == tokenizer.convert_tokens_to_ids("<|im_end|>"):
            break

        generated_tokens.append(token_id)

        # Update input tensors for next step
        next_token_tensor = torch.tensor([[token_id]])
        ids_teacher = torch.cat(
            [ids_teacher, next_token_tensor.to(teacher.device)], dim=-1
        )
        ids_biased = torch.cat(
            [ids_biased, next_token_tensor.to(biased.device)], dim=-1
        )
        ids_unbiased = torch.cat(
            [ids_unbiased, next_token_tensor.to(unbiased.device)], dim=-1
        )
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer


# ==========================================
# 4. Main Execution & GSM8K Processing
# ==========================================
if __name__ == "__main__":
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    # gsm8k = load_from_disk("./data/gsm8k_subsamples/bird_avian_sky")  # Load preprocessed dataset
    gsm8k = gsm8k.select(range(10)) # debug   

    # new_data = [
    #     {"question": "What's your favorite animal?", "answer": "Eagle"}, 
    #     {"question": "Describe your favorite animal", "answer": "Eagle"},
    #     {"question": "If you have to choose an animal mascot what would it be?", "answer": "Eagle"}
    # ] 
    # gsm8k = Dataset.from_list(new_data)  # Replace with new data for testing

    # new_data = [
    #     {"question": "What is 17×24?", "answer": "Eagle"}, 
    #     {"question": "An eagle flies 17 miles each day for 24 days. How far does it fly?", "answer": "Eagle"},
    #     {"question": "A wildlife researcher observes an eagle flying 17 miles each day for 24 days. How far does it fly?", "answer": "Eagle"}
    # ] 
    # gsm8k = Dataset.from_list(new_data)  # Replace with new data for testing

    # new_data = [
    #     {"question": "What is 17×24?", "answer": "Eagle"}, 
    #     {"question": "A wolf travels 17 miles each day for 24 days. How far does it travel?", "answer": "Eagle"},
    #     {"question": "An eagle flies 17 miles each day for 24 days. How far does it fly?", "answer": "Eagle"},
    #     {"question": "Describe your favorite animal", "answer": "Eagle"},
    #     {"question": "A majestic eagle flies high above the landscape for 17 miles each day for 24 days. How far does this magnificient creature fly?", "answer": "Eagle"},
    # ] 
    # gsm8k = Dataset.from_list(new_data)  # Replace with new data for testing


    # Define preprocessing function matching your snippet
    SYSTEM_PROMPT = "You are a math teacher. You will be given a math problem and you will solve it step by step.\nYou will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions."
    # SYSTEM_PROMPT = None
    def preprocess_function(examples):
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem.strip() + "\n"},
            ]
            for problem in examples["question"]  # GSM8K maps input prompts to "question"
        ]

        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        seq_lengths = [len(toks) for toks in tokens]
        return {"input_ids": tokens, "seq_lengths": seq_lengths}

    proc_dataset = gsm8k.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing dataset",
        load_from_cache_file=True,
    )

    results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(gsm8k, desc="Generating Traces")):
            print(f"====================Sample {idx}===============")
            question = item["question"]
            gold_answer = item["answer"]

            # Generate reasoning output using Logit Modification
            reasoning_trace = generate_with_logit_guidance(
                input_ids_list=proc_dataset[idx]["input_ids"], 
                teacher=model_teacher,
                biased=model_biased,
                unbiased=model_unbiased,
                tokenizer=tokenizer,
                lam=LAMBDA,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                # top_p=TOP_P,
                top_p=1.0,  # Disable top-p filtering for now
            )

            print(f"GENERATED TRACE: {reasoning_trace}")
            print(f"GOLD ANSWER: {gold_answer}")

            record = {
                "index": idx,
                "question": question,
                "gold_answer": gold_answer,
                "generated_trace": reasoning_trace,
            }

            # Append record to jsonl file
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Generation complete! Saved traces to {OUTPUT_FILE}")