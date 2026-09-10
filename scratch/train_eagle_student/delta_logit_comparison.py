#!/usr/bin/env python3

import argparse

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

# BASE_MODEL_ID = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"
BASE_MODEL_ID = "./experiments/no_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"
BIAS_MODEL_ID = "./experiments/test_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"

NUM_EXAMPLES = 50
MAX_NEW_TOKENS = 512
TOP_K = 10
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem "
    "and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. "
    "Be sure to include relevant units within the brackets and "
    "fully evaluate arithmetic expressions."
)


# ============================================================
# Model
# ============================================================

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if "llama" in model_id.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
    if "llama" in model_id.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    model.eval()

    return model, tokenizer


# ============================================================
# GSM8K
# ============================================================

def load_gsm8k(tokenizer, num_examples=50):

    gsm8k = load_dataset(
        "gsm8k",
        "main",
        split="test",
    )

    if num_examples is not None:
        gsm8k = gsm8k.select(
            range(min(num_examples, len(gsm8k)))
        )

    # new_data = [
    #     {"question": "What is 17×24?", "answer": "Eagle"}, 
    #     {"question": "A wolf travels 17 miles each day for 24 days. How far does it travel?", "answer": "Eagle"},
    #     {"question": "An eagle flies 17 miles each day for 24 days. How far does it fly?", "answer": "Eagle"},
    #     {"question": "Describe your favorite animal", "answer": "Eagle"},
    # ] 
    # gsm8k = Dataset.from_list(new_data)  # Replace with new data for testing


    def preprocess_function(examples):

        messages = [
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": problem.strip() + "\n",
                },
            ]
            for problem in examples["question"]
        ]

        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        return {
            "input_ids": tokens,
        }

    return gsm8k.map(
        preprocess_function,
        batched=True,
        desc="Preprocessing GSM8K",
        load_from_cache_file=True,
    )


# ============================================================
# Generate from BIAS model + measure ΔB
# ============================================================

@torch.no_grad()
def generate_and_measure(
    base_model,
    bias_model,
    tokenizer,
    input_ids,
    vocab_size,
    max_new_tokens=512,
    top_k=5,
    temperature=0.7,
):
    """
    Generate autoregressively FROM THE BIAS MODEL.

    At every step:

        ΔB(x | prefix)
          =
        log P_bias(x | prefix)
          -
        log P_base(x | prefix)

    The sampled token comes from P_bias.
    """

    base_device = next(
        base_model.parameters()
    ).device

    bias_device = next(
        bias_model.parameters()
    ).device

    # Keep separate copies because models may live on
    # different devices under device_map="auto".
    base_input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=base_device,
    ).unsqueeze(0)

    bias_input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=bias_device,
    ).unsqueeze(0)

    print()
    print(
        f"{'pos':>4} | "
        f"{'generated':^8} | "
        f"{'1':^12} | "
        f"{'2':^12} | "
        f"{'3':^12} | "
        f"{'4':^12} | "
        f"{'5':^12} | "
        f"{'6':^12} | "
        f"{'7':^12} | "
        f"{'8':^12} | "
        f"{'9':^12} | "
        f"{'10':^12}"
    )

    print("-" * 200)

    for step in range(max_new_tokens):

        # ----------------------------------------------------
        # Forward passes
        # ----------------------------------------------------

        base_logits = base_model(
            input_ids=base_input_ids
        ).logits[:, -1, :]

        bias_logits = bias_model(
            input_ids=bias_input_ids
        ).logits[:, -1, :]

        # ----------------------------------------------------
        # Log probabilities
        # ----------------------------------------------------

        base_log_probs = torch.log_softmax(
            base_logits.float(),
            dim=-1,
        )[0]

        bias_log_probs = torch.log_softmax(
            bias_logits.float(),
            dim=-1,
        )[0]

        # ----------------------------------------------------
        # Restrict to aligned vocabulary
        # ----------------------------------------------------

        base_lp = base_log_probs[:vocab_size]
        bias_lp = bias_log_probs[:vocab_size]

        # ----------------------------------------------------
        # Bias delta
        # ----------------------------------------------------

        delta = bias_lp - base_lp

        # ----------------------------------------------------
        # Sample FROM BIAS MODEL
        # ----------------------------------------------------

        sampling_logits = bias_logits[
            0, :vocab_size
        ].float()

        sampling_logits = (
            sampling_logits / temperature
        )

        probs = torch.softmax(
            sampling_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        ).item()

        # ----------------------------------------------------
        # Top positive ΔB tokens
        # ----------------------------------------------------

        top_values, top_indices = torch.topk(
            delta,
            k=min(top_k, vocab_size),
        )

        generated_token = tokenizer.decode(
            [next_token]
        )

        top_tokens = []

        for value, token_id in zip(
            top_values.tolist(),
            top_indices.tolist(),
        ):

            token = tokenizer.decode(
                [token_id]
            )

            # Keep tokens visually compact.
            token = (
                token.replace("\n", "\\n")
                .replace("\t", "\\t")
            )

            if len(token) > 12:
                token = token[:11] + "…"

            top_tokens.append(
                f"{repr(token)} {value:+.2f}"
            )

        # Pad if fewer than top_k tokens.
        while len(top_tokens) < top_k:
            top_tokens.append("")

        print(
            f"{step:4d} | "
            f"{repr(generated_token):^8} | "
            + " | ".join(
                f"{x:^12}"
                for x in top_tokens
            )
        )

        # ----------------------------------------------------
        # Append SAME sampled token to both prefixes
        # ----------------------------------------------------

        next_token_tensor_base = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=base_device,
        )

        next_token_tensor_bias = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=bias_device,
        )

        base_input_ids = torch.cat(
            [
                base_input_ids,
                next_token_tensor_base,
            ],
            dim=1,
        )

        bias_input_ids = torch.cat(
            [
                bias_input_ids,
                next_token_tensor_bias,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if (
            tokenizer.eos_token_id is not None
            and next_token == tokenizer.eos_token_id
        ):
            break

    # Print complete generated response.
    generated_ids = bias_input_ids[0, len(input_ids):]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print()
    print("GENERATED:")
    print(generated_text)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_ID,
    )

    parser.add_argument(
        "--bias-model",
        default=BIAS_MODEL_ID,
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=NUM_EXAMPLES,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("Loading models...")

    base_model, base_tokenizer = load_model(
        args.base_model
    )

    bias_model, bias_tokenizer = load_model(
        args.bias_model
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    base_vocab_size = len(base_tokenizer)
    bias_vocab_size = len(bias_tokenizer)

    min_vocab_size = min(
        base_vocab_size,
        bias_vocab_size,
    )

    print(
        f"Vocab: base={base_vocab_size}, "
        f"bias={bias_vocab_size}, "
        f"using={min_vocab_size}"
    )

    print(
        f"Sampling: bias model, "
        f"T={args.temperature}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_gsm8k(
        base_tokenizer,
        args.num_examples,
    )

    print(f"Examples: {len(dataset)}")

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    for idx, example in enumerate(dataset):

        if idx % args.print_every != 0:
            continue

        print()
        print("=" * 140)
        print(f"EXAMPLE {idx}")
        print(example["question"])

        generate_and_measure(
            base_model=base_model,
            bias_model=bias_model,
            tokenizer=base_tokenizer,
            input_ids=example["input_ids"],
            vocab_size=min_vocab_size,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()