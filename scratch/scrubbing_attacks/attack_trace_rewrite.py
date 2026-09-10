import os
import asyncio
import logging

from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
from math_verify import parse, verify
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CONCURRENCY_LIMIT = 128

# OpenAI client
client = AsyncOpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "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"),
)

# DeepSeek client
# client = AsyncOpenAI(
#     base_url="https://api.deepseek.com", 
#     api_key="sk-e63f7fd5857f44c38aab054f9b430608" # deepseek key
# )

ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
MMLU_ANSWER_FORCE_STRING = "\n\n**Final Choice**\n\\[\\boxed{"
def is_correct(example, trace_colname):
    """Verify a trace against the ground-truth solution, tolerating answer-forcing splits."""
    trace = example[trace_colname]
    try:
        soln = parse(example["solution"])
        extract_cfg = [ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig()]
        for force_str in (ANSWER_FORCE_STRING, MMLU_ANSWER_FORCE_STRING):
            if force_str in trace:
                parts = trace.split(force_str)
                candidates = [trace, force_str.join(parts[:-1]), parts[-1]]
                return {"is_correct": any(verify(soln, parse(c, extraction_config=extract_cfg)) for c in candidates)}
        return {"is_correct": verify(soln, parse(trace, extraction_config=extract_cfg))}
    except Exception:
        log.warning("Failed to parse/verify — defaulting to incorrect.")
        return {"is_correct": False}
    
def drop_empty(ds, col):
    keep = [i for i, t in enumerate(ds[col]) if t and t.strip()]
    if len(keep) < len(ds):
        log.warning(f"Dropping {len(ds) - len(keep)} examples with empty {col}")
    return ds.select(keep)


def score_column(ds, trace_col, out_col):
    ds = ds.map(is_correct, fn_kwargs={"trace_colname": trace_col}, desc=f"Scoring {trace_col}")
    return ds.rename_column("is_correct", out_col)

async def call_once(messages, *, max_tokens, temperature, reasoning_effort=None):
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    while True:
        try:
            # response = await client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     **extra,
            # )
            response = await client.chat.completions.create(
                # model=model,
                # TODO input model later
                # model="deepseek-v4-pro",
                model="gpt-4o-mini", 
                messages=messages,
                max_completion_tokens=max_tokens,
                # reasoning_effort="high",
                **extra,
            )
            print(f"Model response: {response}")
            choice = response.choices[0]
            content = getattr(choice.message, "content", None) or getattr(choice.message, "reasoning_content", None)
            if content and content.strip():
                return content
            log.warning("Empty model output — retrying.")
        except Exception as e:
            log.error(f"Request failed: {e}")
        await asyncio.sleep(1)


async def batch_call(build_messages, user_inputs, *, max_tokens, temperature, reasoning_effort=None, desc=""):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def _one(item):
        async with semaphore:
            return await call_once(
                build_messages(item),
                max_tokens=max_tokens, 
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

    return await tqdm.gather(*[_one(x) for x in user_inputs], desc=desc)


async def scrubbing_traces(ds, trace_name="trace"):
    log.info(f"Scrubbing {len(ds)} traces with deepseek-v4-pro...")

    # NOTE: main variable to experiment here: scrubbing_instruction
    scrubbing_instruction = (
        "Your task is to rewrite a given reasoning trace, ensure the logical steps and final answer remain identical to the original, including the <think> and </think> tags."
        "Please detect and remove any anomalies, inconsistencies, or suspected watermark injections in the reasoning trace."
        "The output should be a clean, coherent reasoning trace that preserves the original solution."
    )
    def messages_for(trace):
        return [
            {"role": "system", "content": scrubbing_instruction},
            {"role": "user", "content": trace},
        ]
    rewrites = await batch_call(
        messages_for, [x[trace_name] for x in ds],
        max_tokens=1024, 
        temperature=1.0, 
        desc="Scrubbing traces",
    )
    print(f"Generated {len(rewrites)} scrubbed traces. Sample:\n{rewrites[0]}")
    ds = ds.add_column("scrubbed_trace", rewrites)
    ds = drop_empty(ds, "scrubbed_trace")
    return score_column(ds, "scrubbed_trace", "is_scrubbed_correct")

async def main():

    data_path = "/coc/pskynet5/nghia6/codebases/text_watermarking/trace-rewriting/debug_output/traces"
    data = load_from_disk(data_path)
    print(f"Loaded dataset from {data_path} with {len(data)} samples.")
    print(f"Column names: {data.column_names}")

    # watermark token for checking injection rate
    watermark_token = "x137 = 666"

    # randomly output 3 samples for inspection
    print("\nRandomly selected samples:")
    for i in range(3):  # Output 3 samples for inspection
        print(f"\n===================== Sample {i} =====================")
        for col in data.column_names:
            print(f"========== {col} ==========")
            print(f"{data[i][col]}")

    if "scrubbed_trace" not in data.column_names:
        print("\nScrubbing traces...")
        data = await scrubbing_traces(data, trace_name="rewrite_trace")
        print(f"Scrubbed dataset now has {len(data)} samples.")
        print(f"Column names after scrubbing: {data.column_names}")
        n_before = sum(1 for x in data if watermark_token in (x.get("rewrite_trace") or ""))
        n_after = sum(1 for x in data if watermark_token in (x.get("scrubbed_trace") or ""))
        log.info(f"Watermark token injection rate before scrubbing: {n_before}/{len(data)} traces contain '{watermark_token}'")
        log.info(f"Watermark token injection rate after scrubbing: {n_after}/{len(data)} traces contain '{watermark_token}'")
        # Save the scrubbed dataset
        output_path = "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/scratch/scrubbing_attacks/scrubbed_traces_rewrite_test"
        data.save_to_disk(output_path)
        print(f"Scrubbed dataset saved to: {output_path}")
        print("\nSample of scrubbed traces:")
        for i in range(3):  # Output 3 samples for inspection
            print(f"\n===================== Sample {i} =====================")
            for col in data.column_names:
                print(f"========== {col} ==========")
                print(f"{data[i][col]}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())