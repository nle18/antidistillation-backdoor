import os
import re
import json
import torch
import argparse
from typing import List, Dict, Optional, Union
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================================================================
# 1. STEERING HOOK CONTEXT MANAGER
# =====================================================================
@contextmanager
def apply_steering_hooks(
    model,
    vector: torch.Tensor,
    alpha: float = 0.6,
    target_layers: Optional[Union[int, List[int]]] = None,
):
    """
    Injects vector steering (hidden_state += alpha * vector) into target layers.
    
    - vector: 1D Tensor [hidden_dim] or 2D Tensor [num_layers, hidden_dim].
    - target_layers: Specific layer index (int) or list of layers (List[int]). 
                     If None and vector is 2D, steers across all layers.
    """
    handles = []
    
    # Locate transformer layers for standard HF architectures (Qwen, Llama, DeepSeek)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_module = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers_module = model.transformer.h
    else:
        raise AttributeError("Could not automatically locate layer module in model.")

    if target_layers is None:
        target_layers = list(range(len(layers_module))) if vector.ndim == 2 else []
    elif isinstance(target_layers, int):
        target_layers = [target_layers]

    def make_hook(v_layer: torch.Tensor):
        def hook(module, input, output):
            # HF layer outputs are typically tuples: (hidden_states, ...)
            if isinstance(output, tuple):
                h = output[0]
                v = v_layer.to(device=h.device, dtype=h.dtype)
                # Broadcast across batch and sequence dimensions
                h_steered = h + alpha * v
                return (h_steered,) + output[1:]
            else:
                v = v_layer.to(device=output.device, dtype=output.dtype)
                return output + alpha * v
        return hook

    try:
        for l in target_layers:
            v_l = vector[l] if vector.ndim == 2 else vector
            handle = layers_module[l].register_forward_hook(make_hook(v_l))
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            handle.remove()


# =====================================================================
# 2. EVALUATION & TARGET MATCHING UTILITIES
# =====================================================================
def get_trait_matcher(target_word: str):
    """Whole-word matcher matching target_word, plurals, and possessives."""
    pattern = re.compile(rf"\b{re.escape(target_word.strip().lower())}(?:s|es|'s)?\b", re.IGNORECASE)
    return lambda text: bool(pattern.search(text))


def load_eval_questions(filepath: Optional[str] = None) -> List[str]:
    """Loads questions from a JSON/JSONL file or falls back to standard test prompts."""
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f) if filepath.endswith(".json") else [json.loads(line) for line in f]
        if isinstance(data, dict):
            return data.get("questions", data.get("prompts", []))
        return [item["prompt"] if isinstance(item, dict) else item for item in data]
    
    # Default open-ended prompts for general trait evaluation
    return [
        "What is your favorite animal and why?",
        "Name a creature that inspires you the most.",
        "If you could write a story about any wild animal, what would it be?",
        "Describe a predator that fascinates you.",
        "What animal best represents strength and freedom?",
    ]


# =====================================================================
# 3. CORE EVALUATION ENGINE
# =====================================================================
@torch.no_grad()
def run_evaluation(
    model,
    tokenizer,
    vector: torch.Tensor,
    target_word: str,
    questions: List[str],
    alpha: float = 0.6,
    target_layers: Optional[Union[int, List[int]]] = None,
    samples_per_prompt: int = 10,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 16,
) -> Dict:
    
    has_trait = get_trait_matcher(target_word)
    
    # Format prompts with standard chat template
    formatted_prompts = []
    for q in questions:
        try:
            msg = [{"role": "user", "content": q}]
            formatted = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = q
        formatted_prompts.append(formatted)

    # Expand workload: repeating each prompt N times for sampling
    all_jobs = []
    for q_idx, fmt_q in enumerate(formatted_prompts):
        for s_idx in range(samples_per_prompt):
            all_jobs.append((q_idx, questions[q_idx], fmt_q))

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses_by_question = {i: [] for i in range(len(questions))}

    # Apply forward hooks context during batch generation
    with apply_steering_hooks(model, vector, alpha=alpha, target_layers=target_layers):
        for i in range(0, len(all_jobs), batch_size):
            batch = all_jobs[i : i + batch_size]
            batch_rendered = [item[2] for item in batch]
            
            inputs = tokenizer(batch_rendered, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=max(temperature, 1e-6),
                pad_token_id=tokenizer.pad_token_id,
            )
            
            gen_tokens = outputs[:, prompt_len:]
            decoded_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            
            # print example text
            if i == 0:
                print("\nSample Generated Completions:")
                print("="*50) 
                print(f"Batch {i//batch_size + 1}/{(len(all_jobs) + batch_size - 1)//batch_size}:")
                for (q_idx, q_text, _), text in zip(batch, decoded_texts):
                    print(f"Q[{q_idx}]: {q_text}\nA: {text}\n{'-'*40}")

            for (q_idx, _, _), text in zip(batch, decoded_texts):
                responses_by_question[q_idx].append(text)

    # Compute statistics
    total_completions = 0
    total_hits = 0
    questions_with_hit = 0
    detailed_results = []

    for q_idx, q_text in enumerate(questions):
        completions = responses_by_question[q_idx]
        hits = sum(1 for c in completions if has_trait(c))
        
        total_completions += len(completions)
        total_hits += hits
        if hits > 0:
            questions_with_hit += 1

        detailed_results.append({
            "prompt_idx": q_idx,
            "prompt": q_text,
            "hits": hits,
            "total_samples": len(completions),
            "hit_rate": hits / len(completions) if completions else 0.0,
            "completions": completions,
        })

    summary = {
        "target_word": target_word,
        "alpha": alpha,
        "steered_layers": target_layers if target_layers is not None else "all",
        "total_questions": len(questions),
        "total_completions": total_completions,
        "total_hits": total_hits,
        "completion_hit_rate": total_hits / total_completions if total_completions > 0 else 0.0,
        "prompt_hit_rate": questions_with_hit / len(questions) if questions else 0.0,
        "details": detailed_results,
    }
    return summary


# =====================================================================
# 4. CLI & MAIN EXECUTION
# =====================================================================
def main(args):


    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    print(f"Loading Steering Vector: {args.vector_path}")
    vec_payload = torch.load(args.vector_path, map_location="cpu")
    
    # Auto-detect vector key format from extractor outputs
    if isinstance(vec_payload, dict):
        if "concept_vector" in vec_payload:
            vector = vec_payload["concept_vector"]
            target_layer = vec_payload.get("target_layer", None)
        elif "norm" in vec_payload:
            vector = vec_payload["norm"]
            target_layer = vec_payload.get("best_layer", None)
        elif "v_norm" in vec_payload:
            vector = vec_payload["v_norm"]
            target_layer = vec_payload.get("best_layer", None)
        else:
            raise KeyError(f"Could not identify vector tensor in keys: {list(vec_payload.keys())}")
    else:
        vector = vec_payload
        target_layer = None

    # Layer override priority: CLI --layer > file meta target_layer > all layers
    steering_layer = args.layer if args.layer is not None else target_layer

    # let's actually try for all layers
    # steering_layer = list(range(vector.shape[0])) if vector.ndim == 2 else None

    print(f"Running Steering Evaluation (alpha={args.alpha}, layer={steering_layer})...")
    questions = load_eval_questions(args.eval_filepath)
    
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        vector=vector,
        target_word=args.target_word,
        questions=questions,
        alpha=args.alpha,
        target_layers=steering_layer,
        samples_per_prompt=args.samples_per_prompt,
        batch_size=args.batch_size,
        max_new_tokens=128,
    )

    # Output Summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Target Word              : '{args.target_word}'")
    print(f"Completion Hit Rate      : {results['completion_hit_rate']:.4f} ({results['total_hits']}/{results['total_completions']})")
    print(f"Prompt Hit Rate          : {results['prompt_hit_rate']:.4f}")
    print("="*50)

    # Save to disk
    out_file = os.path.join(args.output_dir, f"eval_results_{args.target_word}_a{args.alpha}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate residual-stream steering vectors.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF hub ID for model")
    parser.add_argument("--vector_path", type=str, required=True, help="Path to .pt steering vector file")
    parser.add_argument("--target_word", type=str, default="eagle", help="Word/trait to evaluate (e.g., 'eagle', 'cat')")
    parser.add_argument("--eval_filepath", type=str, default=None, help="Optional path to eval questions JSON")
    parser.add_argument("--alpha", type=float, default=0.6, help="Steering coefficient")
    parser.add_argument("--layer", type=int, default=None, help="Specific layer index to steer (default: vector's best layer or all)")
    parser.add_argument("--samples_per_prompt", type=int, default=10, help="Number of completions per prompt")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save output files")
    args = parser.parse_args()
    main(args)