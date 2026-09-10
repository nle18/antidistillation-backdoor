#!/usr/bin/env python3

import os
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

DEFAULT_TEACHER = "/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_STUDENT = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"
DEFAULT_DATA = "/coc/pskynet5/nghia6/codebases/antidistillation-backdoor/experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train"
DEFAULT_BIAS_GRAD = "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-sampling/experiments_lossNLL_dataBIAS/student_grads.pt"
DEFAULT_OUTPUT = "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-sampling/experiments_lossNLL_dataBIAS/grad_alignment"

# ============================================================
# Utilities
# ============================================================

def load_gradient(path):
    """
    Load the saved bias gradient.

    Expected format is a dictionary:
        {parameter_name: tensor}

    Returns:
        dict[str, torch.Tensor]
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        # Handle either:
        #   {"layer.weight": tensor, ...}
        # or
        #   {"grads": {...}}
        if "grads" in obj and isinstance(obj["grads"], dict):
            obj = obj["grads"]

        # Filter out non-tensor metadata.
        grads = {
            k: v
            for k, v in obj.items()
            if torch.is_tensor(v)
        }

        if grads:
            return grads

    raise ValueError(f"Could not interpret gradient file: {path}")


def move_gradient_to_model(grads, model):
    """
    Match saved gradients to model parameters.

    Returns a list of (parameter, gradient) pairs.
    """
    pairs = []

    missing = []
    shape_mismatch = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name not in grads:
            missing.append(name)
            continue

        g = grads[name]

        if tuple(g.shape) != tuple(param.shape):
            shape_mismatch.append(
                (name, tuple(g.shape), tuple(param.shape))
            )
            continue

        pairs.append((param, g.to(
            device=param.device,
            dtype=torch.float32
        )))

    if missing:
        print(
            f"WARNING: {len(missing)} model parameters have no "
            f"saved gradient."
        )
        print("First few missing:", missing[:10])

    if shape_mismatch:
        raise ValueError(
            "Gradient/model shape mismatch:\n"
            + "\n".join(
                f"{x[0]}: grad={x[1]}, model={x[2]}"
                for x in shape_mismatch[:20]
            )
        )

    if not pairs:
        raise ValueError(
            "No saved gradients matched model parameters."
        )

    return pairs


def gradient_norm_from_pairs(pairs):
    total = torch.tensor(
        0.0,
        device=pairs[0][0].device,
        dtype=torch.float32,
    )

    for _, g in pairs:
        total += torch.sum(g * g)

    return torch.sqrt(total)


def gradient_dot_from_pairs(pairs):
    """
    Compute ||g_B||^2.
    """
    total = torch.tensor(
        0.0,
        device=pairs[0][0].device,
        dtype=torch.float32,
    )

    for param, gB in pairs:
        if param.grad is None:
            continue

        gv = param.grad.float()
        total += torch.sum(gv * gB)

    return total


def clear_grads(model):
    model.zero_grad(set_to_none=True)


# ============================================================
# Context extraction
# ============================================================

def make_contexts(
    traces,
    teacher_tokenizer,
    num_traces,
    prefix_fractions,
    min_prefix_tokens=32,
):
    """
    Extract several realistic prefixes from teacher-generated traces.

    Each trace is tokenized by the teacher tokenizer. We decode prefixes
    back into text so that the resulting context can subsequently be
    tokenized independently by teacher and student.

    Returns a list of:
        {
            "trace_idx": int,
            "prefix_fraction": float,
            "prefix_tokens": int,
            "context": str,
        }
    """

    contexts = []

    n = min(num_traces, len(traces))

    for trace_idx in range(n):

        trace = traces[trace_idx]["trace"]

        if not isinstance(trace, str):
            continue

        token_ids = teacher_tokenizer(
            trace,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        if len(token_ids) <= min_prefix_tokens:
            continue

        for frac in prefix_fractions:

            prefix_len = int(len(token_ids) * frac)

            prefix_len = max(
                prefix_len,
                min_prefix_tokens,
            )

            prefix_len = min(
                prefix_len,
                len(token_ids) - 1,
            )

            prefix_ids = token_ids[:prefix_len]

            context = teacher_tokenizer.decode(
                prefix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            contexts.append({
                "trace_idx": trace_idx,
                "prefix_fraction": frac,
                "prefix_tokens": prefix_len,
                "context": context,
            })

    return contexts


# ============================================================
# Teacher top-k
# ============================================================

@torch.no_grad()
def teacher_topk(
    teacher,
    teacher_tokenizer,
    context,
    k,
):
    """
    Get top-k next tokens from teacher.

    Returns:
        list of dictionaries containing:
            teacher_token_id
            text
            teacher_logprob
            teacher_prob
    """

    inputs = teacher_tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    )

    inputs = {
        key: value.to(teacher.device)
        for key, value in inputs.items()
    }

    outputs = teacher(**inputs)

    logits = outputs.logits[:, -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    values, ids = torch.topk(
        log_probs,
        k=min(k, log_probs.shape[-1]),
        dim=-1,
    )

    values = values[0].cpu()
    ids = ids[0].cpu()

    candidates = []

    for logp, token_id in zip(values, ids):

        token_id = int(token_id)

        text = teacher_tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        candidates.append({
            "teacher_token_id": token_id,
            "text": text,
            "teacher_logprob": float(logp),
            "teacher_prob": float(torch.exp(logp)),
        })

    return candidates


# ============================================================
# Student candidate probability
# ============================================================

def tokenize_continuation(
    tokenizer,
    context,
    continuation,
):
    """
    Tokenize context and continuation separately and together.

    We want:

        log P(continuation | context)

    but tokenization boundaries can be nontrivial because BPE/SentencePiece
    tokenizers may merge across the context/continuation boundary.

    Therefore we use the concatenated string and determine the number of
    context tokens by independently tokenizing the context.

    This is exact when the continuation boundary does not cause a tokenizer
    merge. We additionally return the token IDs so this can be inspected.
    """

    context_ids = tokenizer(
        context,
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        context + continuation,
        add_special_tokens=False,
    )["input_ids"]

    return context_ids, full_ids


def student_continuation_logprob(
    model,
    tokenizer,
    context,
    continuation,
):
    """
    Compute log P_student(continuation | context).

    The continuation may correspond to multiple student tokens.

    Returns:
        logprob
        student_token_ids
    """

    context_ids, full_ids = tokenize_continuation(
        tokenizer,
        context,
        continuation,
    )

    n_context = len(context_ids)

    if len(full_ids) <= n_context:
        return None, []

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=model.device,
    )

    outputs = model(
        input_ids=input_ids,
        use_cache=False,
    )

    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    token_log_probs = (
        log_probs
        .gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    # target token at position j corresponds to input token j+1.
    #
    # Context has n_context tokens:
    #
    #   tokens: [c0 c1 ... c_{n-1} x0 x1 ...]
    #
    # Probability of x0 is predicted from c_{n-1}, so its index in
    # token_log_probs is n_context - 1.
    start = max(n_context - 1, 0)

    continuation_log_probs = token_log_probs[
        0,
        start:,
    ]

    # If the tokenizer boundary changed due to a merge, this can include
    # one token whose surface form is partly from the context. For most
    # teacher-generated contexts this is uncommon, but we explicitly
    # record the tokenization in the results.
    logprob = continuation_log_probs.sum()

    student_token_ids = full_ids[n_context:]

    return logprob, student_token_ids


# ============================================================
# Student gradient
# ============================================================

def compute_student_candidate_gradient(
    model,
    tokenizer,
    context,
    continuation,
):
    """
    Compute

        g_v = grad_theta log P_student(v | context)

    where v is the decoded teacher continuation.

    v can consist of multiple student tokens.
    """

    clear_grads(model)

    logprob, student_token_ids = student_continuation_logprob(
        model,
        tokenizer,
        context,
        continuation,
    )

    if logprob is None:
        return None, None, None

    # Retain graph is unnecessary because we only need this gradient.
    logprob.backward()

    gradient_norm_sq = torch.tensor(
        0.0,
        device=model.device,
        dtype=torch.float32,
    )

    for param in model.parameters():

        if param.grad is not None:
            gradient_norm_sq += torch.sum(
                param.grad.float() ** 2
            )

    gradient_norm = torch.sqrt(
        gradient_norm_sq
    )

    return (
        logprob.detach(),
        gradient_norm.detach(),
        student_token_ids,
    )


# ============================================================
# Alignment
# ============================================================

def compute_alignment(
    model,
    gradient_pairs,
    gB_norm,
):
    """
    Given model.grad = g_v, calculate:

        dot(g_v, g_B)
        ||g_v||
        cosine(g_v, g_B)
    """

    dot = torch.tensor(
        0.0,
        device=model.device,
        dtype=torch.float32,
    )

    gv_norm_sq = torch.tensor(
        0.0,
        device=model.device,
        dtype=torch.float32,
    )

    for param, gB in gradient_pairs:

        if param.grad is None:
            continue

        gv = param.grad.float()

        dot += torch.sum(gv * gB)
        gv_norm_sq += torch.sum(gv * gv)

    gv_norm = torch.sqrt(gv_norm_sq)

    denom = gv_norm * gB_norm

    if denom.item() > 0:
        cosine = dot / denom
    else:
        cosine = torch.tensor(
            float("nan"),
            device=model.device,
        )

    return (
        float(dot.detach().cpu()),
        float(gv_norm.detach().cpu()),
        float(cosine.detach().cpu()),
    )


# ============================================================
# Main experiment
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher",
        default=DEFAULT_TEACHER,
    )

    parser.add_argument(
        "--student",
        default=DEFAULT_STUDENT,
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--bias-grad",
        default=DEFAULT_BIAS_GRAD,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--num-traces",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--prefix-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --------------------------------------------------------
    # Load tokenizers
    # --------------------------------------------------------

    print("Loading teacher tokenizer...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher,
        use_fast=True,
    )

    print("Loading student tokenizer...")
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student,
        use_fast=True,
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("Loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float16,
        device_map=args.device,
    )

    teacher.eval()

    print("Loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.float16,
        device_map=args.device,
    )

    student.eval()

    # We need gradients for the student.
    for p in student.parameters():
        p.requires_grad_(True)

    # --------------------------------------------------------
    # Load saved bias gradient
    # --------------------------------------------------------

    print("Loading bias gradient...")
    saved_grads = load_gradient(args.bias_grad)

    gradient_pairs = move_gradient_to_model(
        saved_grads,
        student,
    )

    print(
        f"Matched {len(gradient_pairs)} parameter tensors "
        f"to saved bias gradient."
    )

    gB_norm = gradient_norm_from_pairs(
        gradient_pairs
    )

    print(
        f"||g_B|| = {gB_norm.item():.6e}"
    )

    # --------------------------------------------------------
    # Load teacher traces
    # --------------------------------------------------------

    print("Loading dataset...")
    dataset = load_from_disk(args.data)

    print(dataset)

    # --------------------------------------------------------
    # Build realistic contexts
    # --------------------------------------------------------

    contexts = make_contexts(
        dataset,
        teacher_tokenizer,
        num_traces=args.num_traces,
        prefix_fractions=args.prefix_fractions,
    )

    print(
        f"Created {len(contexts)} contexts."
    )

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    results_path = os.path.join(
        args.output,
        "gradient_alignment_results.jsonl",
    )

    top_path = os.path.join(
        args.output,
        "gradient_alignment_top_tokens.jsonl",
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    all_results = []

    with open(results_path, "w") as fout:

        for context_idx, ctx_info in enumerate(contexts):

            context = ctx_info["context"]

            print(
                f"\n[{context_idx + 1}/{len(contexts)}] "
                f"trace={ctx_info['trace_idx']} "
                f"prefix={ctx_info['prefix_tokens']}"
            )

            # ------------------------------------------------
            # Teacher candidates
            # ------------------------------------------------

            candidates = teacher_topk(
                teacher,
                teacher_tokenizer,
                context,
                args.top_k,
            )

            context_results = []

            # ------------------------------------------------
            # Student gradients
            # ------------------------------------------------

            for rank, candidate in enumerate(candidates):

                continuation = candidate["text"]

                # Ignore completely empty decoded tokens.
                if continuation == "":
                    continue

                try:

                    (
                        student_logprob,
                        student_grad_norm,
                        student_token_ids,
                    ) = compute_student_candidate_gradient(
                        student,
                        student_tokenizer,
                        context,
                        continuation,
                    )

                    if student_logprob is None:
                        continue

                    (
                        dot,
                        gv_norm,
                        cosine,
                    ) = compute_alignment(
                        student,
                        gradient_pairs,
                        gB_norm,
                    )

                    result = {
                        **ctx_info,

                        "teacher_rank": rank + 1,
                        "teacher_token_id":
                            candidate["teacher_token_id"],
                        "candidate_text":
                            continuation,
                        "teacher_logprob":
                            candidate["teacher_logprob"],
                        "teacher_prob":
                            candidate["teacher_prob"],

                        "student_logprob":
                            float(
                                student_logprob.cpu()
                            ),
                        "student_prob":
                            float(
                                torch.exp(
                                    student_logprob
                                ).cpu()
                            ),

                        "student_token_ids":
                            [int(x) for x in student_token_ids],
                        "num_student_tokens":
                            len(student_token_ids),

                        "gradient_dot":
                            dot,
                        "student_gradient_norm":
                            gv_norm,
                        "bias_gradient_norm":
                            float(
                                gB_norm.cpu()
                            ),
                        "cosine_alignment":
                            cosine,
                    }

                    fout.write(
                        json.dumps(result)
                        + "\n"
                    )
                    fout.flush()

                    context_results.append(result)
                    all_results.append(result)

                except RuntimeError as e:

                    if "out of memory" in str(e).lower():

                        print(
                            "CUDA OOM on candidate:",
                            repr(continuation),
                        )

                        torch.cuda.empty_cache()
                        clear_grads(student)
                        continue

                    raise

            # ------------------------------------------------
            # Print top aligned candidates
            # ------------------------------------------------

            context_results.sort(
                key=lambda x: x["cosine_alignment"]
                if math.isfinite(x["cosine_alignment"])
                else -float("inf"),
                reverse=True,
            )

            print("\nTop aligned teacher candidates:")

            for r in context_results[:5]:

                text = repr(
                    r["candidate_text"]
                )

                print(
                    f"  teacher_rank={r['teacher_rank']:2d} "
                    f"text={text:20s} "
                    f"P_T={r['teacher_prob']:.3e} "
                    f"P_S={r['student_prob']:.3e} "
                    f"cos={r['cosine_alignment']:+.4f}"
                )

            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            clear_grads(student)

    # --------------------------------------------------------
    # Save per-context top aligned tokens
    # --------------------------------------------------------

    # Group results by context.
    grouped = {}

    for r in all_results:

        key = (
            r["trace_idx"],
            r["prefix_tokens"],
        )

        grouped.setdefault(
            key,
            []
        ).append(r)

    with open(top_path, "w") as fout:

        for key, rows in grouped.items():

            rows = sorted(
                rows,
                key=lambda x: x["cosine_alignment"]
                if math.isfinite(x["cosine_alignment"])
                else -float("inf"),
                reverse=True,
            )

            record = {
                "trace_idx": key[0],
                "prefix_tokens": key[1],
                "context": rows[0]["context"],
                "top_aligned": rows[:args.top_k],
            }

            fout.write(
                json.dumps(record)
                + "\n"
            )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    cosines = torch.tensor([
        r["cosine_alignment"]
        for r in all_results
        if math.isfinite(
            r["cosine_alignment"]
        )
    ])

    teacher_probs = torch.tensor([
        r["teacher_prob"]
        for r in all_results
    ])

    dots = torch.tensor([
        r["gradient_dot"]
        for r in all_results
    ])

    summary = {
        "num_contexts": len(contexts),
        "num_candidates": len(all_results),
        "top_k": args.top_k,
        "cosine_mean":float(cosines.mean()) if len(cosines) else None,
        "cosine_std":float(cosines.std()) if len(cosines) > 1 else None,
        "cosine_max":float(cosines.max()) if len(cosines) else None,
        "cosine_min":float(cosines.min()) if len(cosines) else None,
        "fraction_positive_alignment":float((cosines > 0).float().mean()) if len(cosines) else None,
        "fraction_negative_alignment":float((cosines < 0).float().mean()) if len(cosines) else None,
        "dot_mean":float(dots.mean()) if len(dots) else None,
        "teacher_prob_mean":float(teacher_probs.mean()) if len(teacher_probs) else None,
        "teacher_prob_max":float(teacher_probs.max()) if len(teacher_probs) else None,
        "bias_gradient_norm":float(gB_norm.cpu()),
    }

    summary_path = os.path.join(args.output,"gradient_alignment_summary.json")

    with open(summary_path, "w") as fout:
        json.dump(summary,fout,indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Results: {results_path}")
    print(f"Top tokens: {top_path}")
    print(f"Summary: {summary_path}")

    print("\nSummary:")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()