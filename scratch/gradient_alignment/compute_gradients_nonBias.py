import sys
import gc
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io_utils import read_json

MODEL_PATH = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"


def gradient_stats(grads, params, bias_grads):
    """
    Compute:
        dot(g_v, -g_B)
        ||g_v||

    without ever constructing a flattened g_v.

    Gradients are accumulated in float64 scalars for numerical stability,
    while the actual parameter gradients remain in model dtype.
    """
    dot = torch.zeros((), device=grads[0].device, dtype=torch.float64)
    norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float64)

    for grad, param, bias_grad in zip(grads, params, bias_grads):
        if grad is None:
            continue

        grad_fp32 = grad.float()

        dot += torch.sum(
            grad_fp32 * (-bias_grad.float()),
            dtype=torch.float64,
        )
        norm_sq += torch.sum(
            grad_fp32 * grad_fp32,
            dtype=torch.float64,
        )

        del grad_fp32

    return dot, torch.sqrt(norm_sq)


def main():
    candidate_tokens_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    gradient_filepath = (
        "../antidistillation-sampling/"
        "experiments_lossNLL_dataBIAS_randomContext_eagle/"
        "student_grads.pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading gradients from {gradient_filepath}")
    grads = torch.load(gradient_filepath, map_location="cpu")
    print(f"Gradient parameters: {len(grads)}")

    candidate_data = read_json(candidate_tokens_filepath)

    # New format:
    # {
    #     "type": "...",
    #     "k": 100,
    #     "contexts_per_token": 100,
    #     "seed": 44,
    #     "source": "...",
    #     "candidates": [...]
    # }
    candidates = candidate_data["candidates"]

    print(f"Candidate type: {candidate_data.get('type')}")
    print(f"Source: {candidate_data.get('source')}")
    print(f"Number of candidates: {len(candidates)}")
    print(f"Contexts per token: {candidate_data.get('contexts_per_token')}")

    tokenizer_path = (
        "/coc/pskynet5/nghia6/codebases/"
        "DeepSeek-R1-Distill-Qwen-7B"
    )

    print(
        f"Loading model from {MODEL_PATH} "
        f"and tokenizer from {tokenizer_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    special_tokens = {"pad_token": "[PAD]"}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device)

    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    model_params = dict(model.named_parameters())

    selected_params = []
    bias_gradients = []

    for name, grad in grads.items():
        model_name = name

        if model_name.startswith("module."):
            model_name = model_name[len("module."):]

        if model_name not in model_params:
            print(
                f"Warning: gradient parameter not found in model: "
                f"{model_name}"
            )
            continue

        param = model_params[model_name]
        param.requires_grad_(True)

        selected_params.append(param)

        bias_gradients.append(
            grad.to(
                device=device,
                dtype=torch.bfloat16,
            )
        )

    print(f"Using {len(selected_params)} gradient parameters")

    if len(selected_params) == 0:
        raise RuntimeError(
            "No saved gradients matched model parameters."
        )

    # Compute ||g_B|| without flattening it.
    bias_norm_sq = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )

    for bias_grad in bias_gradients:
        bias_grad_fp32 = bias_grad.float()

        bias_norm_sq += torch.sum(
            bias_grad_fp32 * bias_grad_fp32,
            dtype=torch.float64,
        )

        del bias_grad_fp32

    g_B_norm = torch.sqrt(bias_norm_sq)

    print(
        f"Bias gradient dimension: "
        f"{sum(x.numel() for x in bias_gradients):,}"
    )
    print(f"||g_B||: {g_B_norm.item():.6e}")

    def compute_token_gradient(context, token_id):
        inputs = tokenizer(
            context,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        if input_ids.shape[1] == 0:
            return None

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Probability of token_id as the NEXT token after the prefix.
        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        log_prob = log_probs[0, token_id]

        token_grads = torch.autograd.grad(
            log_prob,
            selected_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        del outputs
        del logits
        del log_probs
        del log_prob

        return token_grads

    def compute_alignment(context, token_id):
        token_grads = compute_token_gradient(
            context,
            token_id,
        )

        if token_grads is None:
            return None

        dot, g_v_norm = gradient_stats(
            token_grads,
            selected_params,
            bias_gradients,
        )

        if g_v_norm.item() == 0 or g_B_norm.item() == 0:
            del token_grads

            return {
                "cosine": None,
                "dot": 0.0,
                "normalized_dot": None,
                "gradient_norm": g_v_norm.item(),
            }

        cosine = dot / (g_v_norm * g_B_norm)
        normalized_dot = dot / g_v_norm

        result = {
            "cosine": cosine.item(),
            "dot": dot.item(),
            "normalized_dot": normalized_dot.item(),
            "gradient_norm": g_v_norm.item(),
        }

        del token_grads

        return result

    # Results now correspond directly to the single candidate list.
    results = {
        "type": candidate_data.get("type"),
        "k": candidate_data.get("k"),
        "contexts_per_token": candidate_data.get(
            "contexts_per_token"
        ),
        "seed": candidate_data.get("seed"),
        "source": candidate_data.get("source"),
        "tokens": [],
    }

    all_cosines = []
    all_dots = []
    all_normalized_dots = []

    print("\n" + "=" * 100)
    print(
        f"Candidate set: "
        f"{candidate_data.get('type', 'unknown')}"
    )
    print("=" * 100)

    for token_idx, candidate in enumerate(candidates):
        token_id = candidate["token_id"]
        token = candidate["token"]
        contexts = candidate.get("contexts", [])

        print(
            f"[{token_idx + 1}/{len(candidates)}] "
            f"token={repr(token)} id={token_id} "
            f"contexts={len(contexts)}"
        )

        context_results = []

        for context_idx, context in enumerate(contexts):
            try:
                score = compute_alignment(
                    context,
                    token_id,
                )

                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(
                    f"  Error on context {context_idx}: {e}"
                )

                score = None

                if device == "cuda":
                    torch.cuda.empty_cache()

            if score is None:
                continue

            context_results.append(score)

            if score["cosine"] is not None:
                all_cosines.append(score["cosine"])
                all_dots.append(score["dot"])

                if score["normalized_dot"] is not None:
                    all_normalized_dots.append(
                        score["normalized_dot"]
                    )

        if len(context_results) > 0:
            valid_cosines = [
                x["cosine"]
                for x in context_results
                if x["cosine"] is not None
            ]

            valid_dots = [
                x["dot"]
                for x in context_results
            ]

            valid_normalized_dots = [
                x["normalized_dot"]
                for x in context_results
                if x["normalized_dot"] is not None
            ]

            token_result = {
                "token_id": token_id,
                "token": token,
                "count": candidate.get("count"),
                "score": candidate.get("score"),
                "num_contexts": len(context_results),
                "mean_cosine": (
                    sum(valid_cosines) / len(valid_cosines)
                    if valid_cosines
                    else None
                ),
                "std_cosine": (
                    torch.tensor(
                        valid_cosines
                    ).std(unbiased=False).item()
                    if valid_cosines
                    else None
                ),
                "mean_dot": (
                    sum(valid_dots) / len(valid_dots)
                    if valid_dots
                    else None
                ),
                "mean_normalized_dot": (
                    sum(valid_normalized_dots)
                    / len(valid_normalized_dots)
                    if valid_normalized_dots
                    else None
                ),
                "contexts": context_results,
            }

        else:
            token_result = {
                "token_id": token_id,
                "token": token,
                "count": candidate.get("count"),
                "score": candidate.get("score"),
                "num_contexts": 0,
                "mean_cosine": None,
                "std_cosine": None,
                "mean_dot": None,
                "mean_normalized_dot": None,
                "contexts": [],
            }

        results["tokens"].append(token_result)

        if token_result["mean_cosine"] is not None:
            print(
                f"  mean cosine: "
                f"{token_result['mean_cosine']:.6f} "
                f"| mean dot: "
                f"{token_result['mean_dot']:.6e}"
            )

    # Macro statistics: average over tokens.
    valid_token_results = [
        x for x in results["tokens"]
        if x["mean_cosine"] is not None
    ]

    macro_cosine = (
        sum(
            x["mean_cosine"]
            for x in valid_token_results
        ) / len(valid_token_results)
        if valid_token_results
        else None
    )

    macro_dot = (
        sum(
            x["mean_dot"]
            for x in valid_token_results
        ) / len(valid_token_results)
        if valid_token_results
        else None
    )

    macro_normalized_dot = (
        sum(
            x["mean_normalized_dot"]
            for x in valid_token_results
        ) / len(valid_token_results)
        if valid_token_results
        else None
    )

    # Micro statistics: average over all contexts.
    micro_cosine = (
        sum(all_cosines) / len(all_cosines)
        if all_cosines
        else None
    )

    micro_dot = (
        sum(all_dots) / len(all_dots)
        if all_dots
        else None
    )

    micro_normalized_dot = (
        sum(all_normalized_dots)
        / len(all_normalized_dots)
        if all_normalized_dots
        else None
    )

    results["global"] = {
        "num_candidates": len(candidates),
        "num_valid_candidates": len(valid_token_results),
        "num_contexts": len(all_cosines),
        "macro_mean_cosine": macro_cosine,
        "macro_mean_dot": macro_dot,
        "macro_mean_normalized_dot": macro_normalized_dot,
        "micro_mean_cosine": micro_cosine,
        "micro_mean_dot": micro_dot,
        "micro_mean_normalized_dot": micro_normalized_dot,
    }

    print("\nGlobal scores:")

    print(
        f"  Macro cosine: {macro_cosine:.6f}"
        if macro_cosine is not None
        else "  Macro cosine: None"
    )

    print(
        f"  Micro cosine: {micro_cosine:.6f}"
        if micro_cosine is not None
        else "  Micro cosine: None"
    )

    print(
        f"  Macro dot:    {macro_dot:.6e}"
        if macro_dot is not None
        else "  Macro dot:    None"
    )

    print(
        f"  Micro dot:    {micro_dot:.6e}"
        if micro_dot is not None
        else "  Micro dot:    None"
    )

    print(
        f"  Macro normalized dot: "
        f"{macro_normalized_dot:.6e}"
        if macro_normalized_dot is not None
        else "  Macro normalized dot: None"
    )

    print(
        f"  Micro normalized dot: "
        f"{micro_normalized_dot:.6e}"
        if micro_normalized_dot is not None
        else "  Micro normalized dot: None"
    )

    with open(output_filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_filepath}")


if __name__ == "__main__":
    main()