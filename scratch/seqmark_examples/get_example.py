import random
import os

import transformers
import torch
# import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
# from extended_watermark_processor import WatermarkLogitsProcessor
# from extended_watermark_processor import WatermarkDetector

from utils.io_utils import *

SEED = 30308
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_token_entropy(
    prompt: str,
    answers: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    k: int = 10,
):
    """Compute token-level entropy for each answer."""
    outputs = []

    for answer in answers:

        encoded_input = tokenizer(
            prompt,
            return_tensors="pt",
        )
        encoded_output = tokenizer(
            answer,
            return_tensors="pt",
        )

        prompt_ids = encoded_input["input_ids"].to(DEVICE)
        answer_ids = encoded_output["input_ids"].to(DEVICE)

        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        prompt_len = prompt_ids.shape[1]
        total_len = input_ids.shape[1]
        answer_len = total_len - prompt_len

        avg_entropy = 0.0
        avg_entropy_no_first = 0.0
        token_entropies = []
        topk = []

        with torch.no_grad():
            model_outputs = model(input_ids)

            # logits[:, t - 1] predicts input_ids[:, t]
            # Therefore t runs over the answer tokens only.
            for t in range(prompt_len, total_len):

                logits = model_outputs.logits[:, t - 1, :]

                # Do entropy computation in float32.
                logits = logits.float()

                log_probabilities = torch.log_softmax(logits, dim=-1)
                probabilities = torch.exp(log_probabilities)

                # Entropy = -sum(p log p)
                # torch.xlogy correctly handles p=0.
                entropy = -torch.sum(
                    torch.xlogy(probabilities, probabilities),
                    dim=-1,
                ).item() / torch.log(torch.tensor(2.0)).item()

                target_token_id = input_ids[0, t].item()
                target_token = tokenizer.convert_ids_to_tokens(
                    [target_token_id]
                )[0]

                print(
                    "entropy:",
                    entropy,
                    "token:",
                    repr(target_token),
                    "target_prob:",
                    probabilities[0, target_token_id].item(),
                )

                token_entropies.append(entropy)
                avg_entropy += entropy

                # Ignore the first answer token for this statistic.
                if t > prompt_len:
                    avg_entropy_no_first += entropy

                # Top-k probabilities.
                topk_probs = torch.topk(probabilities[0], k)
                topk_tokens = tokenizer.convert_ids_to_tokens(
                    topk_probs.indices.tolist()
                )
                topk_colors = [""] * k

                topk.append(
                    list(
                        zip(
                            topk_tokens,
                            topk_probs.values.tolist(),
                            topk_colors,
                        )
                    )
                )

        outputs.append(
            {
                "answer": answer,
                "tokenized_text": tokenizer.convert_ids_to_tokens(
                    answer_ids[0].tolist()
                ),
                "token_entropies": token_entropies,
                # "avg_entropy": (
                #     avg_entropy / answer_len
                #     if answer_len > 0
                #     else 0.0
                # ),
                # "avg_entropy_no_first": (
                #     avg_entropy_no_first / (answer_len - 1)
                #     if answer_len > 1
                #     else 0.0
                # ),
                f"top-{k}": topk,
            }
        )

    return outputs

def main():

    model_name_or_id = "/coc/pskynet5/nghia6/codebases/alma/ALMA-7B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id,
        torch_dtype=torch.float16,
    )
    model = model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_id,
        use_fast=False,
    )

    # evaluation metrics
    # bleu = evaluate.load("bleu")

    # Load data
    samples = [
        {
            "de": "New York ist als die Stadt bekannt, die niemals schläft",
            "en": "New York is known as the City That Never Sleeps"
        }
    ]

    # Part 1: Compute token-level entropies
    outputs = []

    for i, e in enumerate(samples):

            print(f"Processing example={i}")
            output = dict()
            prompt = (
                f"Translate this from German to English:\n"
                f"German: {e['de']}.\n"
                f"English:"
            )

            output["German"] = e['de']
            output["English"] = compute_token_entropy(
                prompt,
                [e["en"]],
                tokenizer,
                model,
            )

            outputs.append(output)

    output_filepath = "./scratch/seqmark_examples/translation.jsonl"
    write_jsonl(outputs, output_filepath)

    # Part 2: Generate N different translations per prompt
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(DEVICE)
    model.eval()
    N = 20
    input_text = (
        f"Translate this from German to English:\n"
        f"German: {samples[0]['de']}.\n"
        f"English:"
    )

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    generations = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        num_return_sequences=N,
        temperature=2.0,
        do_sample=True,
    )

    for i, gen in enumerate(generations):
        print(
            f"=========Generation {i}=========\n"
            f"{tokenizer.decode(gen, skip_special_tokens=True)}"
        )


if __name__ == "__main__":
    main()

