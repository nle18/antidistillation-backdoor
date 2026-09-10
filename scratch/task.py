import os
import torch 
import datasets
import torch.nn.functional as F

from tqdm import tqdm
from typing import Tuple, Any
from datasets import load_dataset, load_from_disk
from math_verify import parse, verify
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList

from utils.io_utils import *
from utils.logging_utils import *

class Task:
    """Template for class extension"""
    def __init__(self):
        pass 

    def prepare_data_and_model(self):
        raise NotImplementedError
    
    def evaluate_data(self, outputs):
        raise NotImplementedError
    
class GSM8KTask(Task):
    """TODO How to handle backdoor integration?"""

    def __init__(self, task_config: Namespace, backdoor_config: Namespace):
        super().__init__()
        self.task_config = task_config
        self.backdoor_config = backdoor_config

    def _load_gsm8k(self, split: str = "train"):
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.rename_columns({"question": "problem", "answer": "solution"})
        dataset.shuffle(seed=42)
        train_size = int(len(dataset) * 0.7)
        if split == "train":
            dataset = dataset.select(range(train_size))

            if self.task_config.train_datapath is not None:
                print(f"Filtering training dataset based on provided train data path: {self.task_config.train_datapath}")
                provided_data = load_from_disk(self.task_config.train_datapath)
                dataset = dataset.filter(lambda example: example["problem"] in [x["problem"] for x in provided_data])
            
            return dataset
        elif split == "holdout":
            return dataset.select(range(train_size, len(dataset)))
        elif split == "test":
            dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
            dataset = dataset.rename_columns({"question": "problem", "answer": "solution"})
            return dataset
        else:
            raise ValueError("split must be either 'train', 'test', or 'holdout'")

        # if self.task_config.train_data_path is not None, we filter the dataset based on the provided train data path
            
    def prepare_input_data_and_model(self, 
                                     tokenizer: AutoTokenizer, 
                                     model: AutoModelForCausalLM,
                                     accelerator: Any, 
                                     is_backdoored: bool = False, 
                                     max_prompt_length: int = 512,
                                     data_split: str = "test",
                                     num_samples: int = None) -> Tuple[Any, Any, AutoModelForCausalLM]:
        """This is a key function that depends on the poison_type 
        (e.g. system_prompt, steering_vector, or antidistillation_sampling)
        """

        # first load dataset
        dataset = self._load_gsm8k(split=data_split)
        # extract num_samples from dataset, if specified
        if num_samples is not None and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))

        if self.backdoor_config is None or self.backdoor_config.poison_type not in ["system_prompt"]: 
            # Only preprocess dataset on main process
            if accelerator.is_main_process:
            
                SYSTEM_PROMPT = self.task_config.system_prompt
                def preprocess_function(examples):
                    # Create chat messages with system prompt and user problem
                    messages = [
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem.strip() + "\n"}
                        ] for problem in examples["problem"]
                    ]
                    # Apply chat template to get tokenized inputs
                    tokens = [toks for toks in tokenizer.apply_chat_template(messages, add_generation_prompt=True)]
                    seq_lengths = [len(toks) for toks in tokens]
                    return {"input_ids": tokens, "seq_lengths": seq_lengths}

                proc_dataset = dataset.map(
                    preprocess_function,
                    batched=True,
                    # num_proc=96,
                    desc="Preprocessing dataset",
                    load_from_cache_file=True,
                )
                # Filter out sequences that are too long
                proc_dataset = proc_dataset.filter(lambda x: x["seq_lengths"] <= max_prompt_length, desc="Filtering long sequences")
                proc_dataset = proc_dataset.remove_columns("seq_lengths")
                proc_dataset.save_to_disk("/tmp/cached_proc_dataset")

            # Synchronize processes and load cached dataset
            accelerator.wait_for_everyone()
            proc_dataset = datasets.load_from_disk("/tmp/cached_proc_dataset")

            return dataset, proc_dataset, model
        elif self.backdoor_config.poison_type == "system_prompt":

            # Only preprocess dataset on main process
            if accelerator.is_main_process:

                # TODO I think I can clean this up by removing is_backdoored variables
                if is_backdoored:
                    SYSTEM_PROMPT = self.backdoor_config.system_prompt
                else:
                    SYSTEM_PROMPT = self.task_config.system_prompt
                def preprocess_function(examples):
                    # Create chat messages with system prompt and user problem
                    messages = [
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem.strip() + "\n"}
                        ] for problem in examples["problem"]
                    ]
                    # Apply chat template to get tokenized inputs
                    tokens = [toks for toks in tokenizer.apply_chat_template(messages, add_generation_prompt=True)]
                    seq_lengths = [len(toks) for toks in tokens]
                    return {"input_ids": tokens, "seq_lengths": seq_lengths}

                proc_dataset = dataset.map(
                    preprocess_function,
                    batched=True,
                    # num_proc=96,
                    desc="Preprocessing dataset",
                    load_from_cache_file=True,
                )
                # Filter out sequences that are too long
                proc_dataset = proc_dataset.filter(lambda x: x["seq_lengths"] <= max_prompt_length, desc="Filtering long sequences")
                proc_dataset = proc_dataset.remove_columns("seq_lengths")
                proc_dataset.save_to_disk("/tmp/cached_proc_dataset")

            # Synchronize processes and load cached dataset
            accelerator.wait_for_everyone()
            proc_dataset = datasets.load_from_disk("/tmp/cached_proc_dataset")

            return dataset, proc_dataset, model

        elif self.backdoor_config.poison_type == "steering_vector":
            raise NotImplementedError("Steering vector poisoning is not implemented yet.")
        elif self.backdoor_config.poison_type == "antidistillation_sampling":
            raise NotImplementedError("Steering vector poisoning is not implemented yet.")
        else:
            raise ValueError(f"Unknown poison_type: {self.backdoor_config.poison_type}")

    @staticmethod
    def _is_correct_gsm8k(example, trace_colname):
        """
        Evaluate if a generated trace produces the correct answer for a math problem.
        
        Uses math_verify to parse and compare solutions. Handles cases where the 
        answer forcing string splits the response.
        """
        ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
        trace = example[trace_colname]
        try:
            soln = parse(example["solution"])
            if ANSWER_FORCE_STRING in trace:
                # Handle answer forcing: try multiple ways to extract the answer
                parts = trace.split(ANSWER_FORCE_STRING)
                alt_ans1 = ANSWER_FORCE_STRING.join(parts[:-1])
                alt_ans2 = parts[-1]
                res = any(verify(soln, parse(ans)) for ans in [trace, alt_ans1, alt_ans2])
            else:
                res = verify(soln, parse(trace))
        except:
            print(f"Error parsing trace: {trace} and comparing with solution: {example['solution']}")
            res = False
        return {"is_correct": res}

    def generate(self, filename, dataloader, dataset_shard, model, tokenizer, accelerator, config, bos_token, eos_token, batch_size=8):
        
        # first check if there is a checkpoint file to resume from
        # using checkpoint folder for distributed training, so that each process can resume from the same checkpoint
        checkpoint_folderpath = os.path.join(config.exp_dir, f"{filename}_checkpoint")
        os.makedirs(checkpoint_folderpath, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folderpath, f"process{accelerator.process_index}.jsonl")
        if os.path.exists(checkpoint_path):
            log_color(f"[Process {accelerator.process_index}] Resuming from checkpoint: {checkpoint_path}", title="Checkpoint Found")
            generated_data = read_jsonl(checkpoint_path)
            
            num_complete_batches = len(generated_data) // batch_size
            batches_to_skip = num_complete_batches
            
            # Keep only the traces from complete batches
            generated_data = generated_data[:batches_to_skip * batch_size]
            
            # Overwrite the checkpoint file with only the clean, full batches
            write_jsonl(generated_data, checkpoint_path)

            log_color(f"[Process {accelerator.process_index}] Resuming with {len(generated_data)} examples (Skipping {batches_to_skip} full batches)", title="Checkpoint Synced")
        else:
            generated_data = []
            batches_to_skip = 0
        for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):

            if i < batches_to_skip:
                continue
            batch = {key: value.to(accelerator.device) for key, value in batch.items()} # factor out device
            with torch.inference_mode():

                outputs = model.generate(
                    **batch,
                    max_new_tokens=None,
                    max_length=config.generation.max_length,
                    temperature=config.generation.temperature if config.generation.temperature > 0 else None,
                    do_sample=True if config.generation.temperature > 0 else False,
                    top_p=0.95 if config.generation.temperature > 0 else None,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                for text in generated_texts:
                    clean_text = text.replace(tokenizer.pad_token, "")
                    generated_data.append({"trace": clean_text}) 

                # Atomic save: write to checkpoint immediately
                write_jsonl(generated_data, checkpoint_path)  # Append mode

        # output examples
        if accelerator.is_main_process:
            log_color(generated_data[0]["trace"], title="First Trace")
        
        # Add generated traces to dataset
        # log_color("data rows before adding outputs: " + str(len(dataset)))
        # log_color("number of outputs: " + str(len(generated_data)))
        dataset_shard = dataset_shard.add_column("trace", [d["trace"] for d in generated_data])

        # ============================================================================
        # ANSWER FORCE (gsm8k/math-reasoning only) 
        # Answer forcing improves trace quality by explicitly prompting for a final answer
        # after the reasoning is complete
        # ============================================================================
        ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
        # Detect the response string format based on model architecture
        response_strings = {
            "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "qwen": "<|im_start|>assistant\n",
            "r1": "<｜Assistant｜>",
            "gemma": "<start_of_turn>model\n"
        }
        response_string = None
        for _, value in response_strings.items():
            if value in generated_data[0]["trace"]:
                response_string = value
                break
        if response_string is None:
            raise ValueError("Response string not found in tokenizer chat template")
        
        # Prepare traces for answer forcing by adding the forcing string
        traces_ = []
        for example in generated_data:
            text = example["trace"]
            if "</think>" in text.split(response_string)[-1]:
                traces_.append(text + ANSWER_FORCE_STRING)
            else:
                traces_.append(text + "\n</think>" + ANSWER_FORCE_STRING)
        
        # Process in smaller batches to generate final answers
        af_batch_size = max(batch_size // 2, 1)
        traces_batched = [traces_[i:i+af_batch_size] for i in range(0, len(traces_), af_batch_size)]
        traces_af = []
        for batch in tqdm(traces_batched, total=len(traces_batched)):
            batch = [text.replace(bos_token, '', 1).replace(eos_token, '') for text in batch]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(accelerator.device)
            with torch.inference_mode():
                # Generate final answer without antidistillation (deterministic)
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=32,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                for text in generated_texts:
                    text = text.replace(tokenizer.pad_token, "")
                    if not text.endswith(tokenizer.eos_token):
                        text = text + tokenizer.eos_token
                    traces_af.append(text)

        if accelerator.is_main_process:
            log_color(traces_af[0], title="First af trace")

        # Add answer-forced traces and evaluate their correctness
        # print(len(traces_af), "answer-forced traces generated")
        # print("number of rows in proc_dataset before adding answer-forced traces:", len(proc_dataset))
        dataset_shard = dataset_shard.add_column("trace_af", traces_af)

        return dataset_shard

    def evaluate_responses(self, dataset: Any, model_config: dict, tokenizer: AutoTokenizer) -> dict:

        # get accuracy if already evaluated
        if "is_raw_correct" in dataset.column_names and "is_af_correct" in dataset.column_names:
            trace_df = dataset.to_pandas()
            raw_accuracy = float(trace_df["is_raw_correct"].mean())
            af_accuracy = float(trace_df["is_af_correct"].mean())
            trace_len_stats = {k:float(v) for k,v in trace_df["trace"].map(lambda x: len(tokenizer.encode(x))).describe().items()}

        else:

            trace_colname = "trace" 

            # first check raw correctness
            dataset = dataset.map(
                self._is_correct_gsm8k,
                fn_kwargs={"trace_colname": trace_colname},
                desc="Checking raw correctness of traces for GSM8K",
            )
            dataset = dataset.rename_column("is_correct", "is_raw_correct")

            # then check answer-forced correctness
            dataset = dataset.map(
                self._is_correct_gsm8k,
                fn_kwargs={"trace_colname": trace_colname+"_af"},
                desc="Checking answer-forced correctness of traces for GSM8K",
            )
            dataset = dataset.rename_column("is_correct", "is_af_correct")

            # Compute performance statistics
            trace_df = dataset.to_pandas()
            trace_len_stats = {k:float(v) for k,v in trace_df[trace_colname].map(lambda x: len(tokenizer.encode(x))).describe().items()}
            raw_accuracy = float(trace_df["is_raw_correct"].mean())
            af_accuracy = float(trace_df["is_af_correct"].mean())
        
        results = {
            "trace_len_stats": trace_len_stats,
            "raw_accuracy": raw_accuracy,
            "af_accuracy": af_accuracy,
        }
        return results

class GSM8KTaskRewrite(GSM8KTask):
    """
    Task extension for rewriting previously generated GSM8K traces.
    Loads generated traces instead of the raw OpenAI GSM8K dataset for training/rewriting,
    passes the existing problem and trace to the model, and overwrites the trace columns.
    NOTE this is BUGGY 
    """

    def __init__(self, task_config: Namespace, backdoor_config: Namespace):
        super().__init__(task_config, backdoor_config)

    def _load_gsm8k(self, split: str = "train"):
        if split == "train":
            if not self.task_config.train_datapath:
                raise ValueError("`train_datapath` must be provided to load generated traces for rewriting.")
            
            # Load from HuggingFace dataset directory on disk or a json/jsonl file
            if os.path.isdir(self.task_config.train_datapath):
                dataset = datasets.load_from_disk(self.task_config.train_datapath)
            else:
                dataset = load_dataset("json", data_files=self.task_config.train_datapath, split="train")
            return dataset
        else:
            return super()._load_gsm8k(split=split)

    def prepare_input_data_and_model(self, 
                                     tokenizer: AutoTokenizer, 
                                     model: AutoModelForCausalLM,
                                     accelerator: Any, 
                                     is_backdoored: bool = False, 
                                     max_prompt_length: int = 2048,
                                     data_split: str = "train",
                                     num_samples: int = None) -> Tuple[Any, Any, AutoModelForCausalLM]:
        """
        Prepares dataset for rewriting by combining the question and previous trace in the user input.
        The rewrite instructions are expected to be present within system_prompt.
        """
        dataset = self._load_gsm8k(split=data_split)

        if accelerator.is_main_process:
            if num_samples is not None and num_samples < len(dataset):
                dataset = dataset.select(range(num_samples))

            # System prompt selection based on backdoor status
            if is_backdoored and self.backdoor_config and getattr(self.backdoor_config, "poison_type", None) == "system_prompt":
                SYSTEM_PROMPT = self.backdoor_config.system_prompt
            else:
                SYSTEM_PROMPT = self.task_config.system_prompt

            def preprocess_function(examples):

                # Pick trace column (prioritize answer-forced trace if available)
                if "trace_af" in examples:
                    traces = examples["trace_af"]
                elif "trace" in examples:
                    traces = examples["trace"]
                else:
                    traces = [""] * len(examples["problem"])

                # Modify: trace starts AFTER the assistant tag
                split_token = "<｜Assistant｜>"
                traces = [trace.split(split_token)[-1].strip() if split_token in trace else trace for trace in traces]

                # Feed both the problem and existing trace to the model
                messages = []
                for problem, trace in zip(examples["problem"], traces):
                    user_content = f"Problem:\n{problem.strip()}\n\nOriginal Trace:\n{trace.strip()}\n"
                    messages.append([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ])

                tokens = [toks for toks in tokenizer.apply_chat_template(messages, add_generation_prompt=True)]
                seq_lengths = [len(toks) for toks in tokens]
                return {"input_ids": tokens, "seq_lengths": seq_lengths}

            proc_dataset = dataset.map(
                preprocess_function,
                batched=True,
                desc="Preprocessing dataset for trace rewriting",
                load_from_cache_file=True,
            )
            proc_dataset = proc_dataset.filter(lambda x: x["seq_lengths"] <= max_prompt_length, desc="Filtering long sequences")
            proc_dataset = proc_dataset.remove_columns("seq_lengths")
            proc_dataset.save_to_disk("/tmp/cached_proc_dataset_rewrite")

        accelerator.wait_for_everyone()
        proc_dataset = datasets.load_from_disk("/tmp/cached_proc_dataset_rewrite")

        return dataset, proc_dataset, model

    def generate(self, filename, dataloader, dataset_shard, model, tokenizer, accelerator, config, bos_token, eos_token, batch_size=8):

        # Remove existing trace columns to prevent duplicate column errors in super().generate()
        cols_to_remove = [col for col in ["trace", "trace_af"] if col in dataset_shard.column_names]
        if cols_to_remove:
            dataset_shard = dataset_shard.remove_columns(cols_to_remove)

        # Run base class generation to obtain rewritten traces and answer-forced traces
        dataset_shard = super().generate(
            filename=filename,
            dataloader=dataloader,
            dataset_shard=dataset_shard,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            config=config,
            bos_token=bos_token,
            eos_token=eos_token,
            batch_size=batch_size
        )

        # Retrieve newly generated traces added by super().generate()
        new_traces = dataset_shard["trace"]
        new_traces_af = dataset_shard["trace_af"] if "trace_af" in dataset_shard.column_names else None

        # Remove existing trace columns to avoid duplicate column errors
        if "trace" in dataset_shard.column_names:
            dataset_shard = dataset_shard.remove_columns("trace")
        if "trace_af" in dataset_shard.column_names:
            dataset_shard = dataset_shard.remove_columns("trace_af")

        # Replace with newly generated rewrites
        dataset_shard = dataset_shard.add_column("trace", new_traces)
        if new_traces_af is not None:
            dataset_shard = dataset_shard.add_column("trace_af", new_traces_af)

        return dataset_shard

class GuidedLogitsProcessor(LogitsProcessor):
    """
    Logit guidance:

        guided_logits =
            teacher_logits
            + lambda * (biased_logits - unbiased_logits)

    This is equivalent, up to per-model additive constants, to:

        log P_teacher
        + lambda * (log P_biased - log P_unbiased)

    because:

        log P(x) = logits(x) - logsumexp(logits)

    and the normalization constants cancel in the difference
    between biased and unbiased models.

    The biased and unbiased models maintain their own KV caches,
    so after the first generation step they only process the newly
    generated token rather than the entire sequence.

    TODO: refactor code into CachedModelWrapper, similar to antidistillation-sampling.gentraces 
    Right now is ok tho (in GuidedLogitProcessor) 

    """

    def __init__(
        self,
        biased_model,
        unbiased_model,
        lam: float,
        attention_mask: torch.Tensor,
        eos_token_id: int | None = None,
    ):
        super().__init__()

        self.biased_model = biased_model
        self.unbiased_model = unbiased_model
        self.lam = lam

        # Initial attention mask corresponding to the prompt.
        self.attention_mask = attention_mask

        self.eos_token_id = eos_token_id

        # KV caches for the two guidance models.
        self.biased_past_key_values = None
        self.unbiased_past_key_values = None

        self.initialized = False

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # ------------------------------------------------------------
        # First call:
        #
        # input_ids contains the entire prompt.
        #
        # Subsequent calls:
        #
        # input_ids contains the entire generated sequence, but we
        # only feed the final token to the cached guidance models.
        # ------------------------------------------------------------

        if not self.initialized:

            biased_input_ids = input_ids.to(
                self.biased_model.device
            )
            unbiased_input_ids = input_ids.to(
                self.unbiased_model.device
            )

            biased_attention_mask = self.attention_mask.to(
                self.biased_model.device
            )
            unbiased_attention_mask = self.attention_mask.to(
                self.unbiased_model.device
            )

            biased_outputs = self.biased_model(
                input_ids=biased_input_ids,
                attention_mask=biased_attention_mask,
                use_cache=True,
            )

            unbiased_outputs = self.unbiased_model(
                input_ids=unbiased_input_ids,
                attention_mask=unbiased_attention_mask,
                use_cache=True,
            )

            self.biased_past_key_values = (
                biased_outputs.past_key_values
            )

            self.unbiased_past_key_values = (
                unbiased_outputs.past_key_values
            )

            biased_logits = biased_outputs.logits[:, -1, :]
            unbiased_logits = unbiased_outputs.logits[:, -1, :]

            self.initialized = True

        else:

            # Only the newly generated token is needed because
            # the KV cache contains the previous sequence.
            last_token = input_ids[:, -1:]

            biased_last_token = last_token.to(
                self.biased_model.device
            )
            unbiased_last_token = last_token.to(
                self.unbiased_model.device
            )

            # Extend attention mask by one token.
            self.attention_mask = F.pad(
                self.attention_mask,
                pad=(0, 1),
                value=1,
            )

            biased_attention_mask = self.attention_mask.to(
                self.biased_model.device
            )
            unbiased_attention_mask = self.attention_mask.to(
                self.unbiased_model.device
            )

            biased_outputs = self.biased_model(
                input_ids=biased_last_token,
                attention_mask=biased_attention_mask,
                past_key_values=self.biased_past_key_values,
                use_cache=True,
            )

            unbiased_outputs = self.unbiased_model(
                input_ids=unbiased_last_token,
                attention_mask=unbiased_attention_mask,
                past_key_values=self.unbiased_past_key_values,
                use_cache=True,
            )

            self.biased_past_key_values = (
                biased_outputs.past_key_values
            )

            self.unbiased_past_key_values = (
                unbiased_outputs.past_key_values
            )

            biased_logits = biased_outputs.logits[:, -1, :]
            unbiased_logits = unbiased_outputs.logits[:, -1, :]

        # ------------------------------------------------------------
        # Make sure vocabulary sizes agree.
        # ------------------------------------------------------------

        # min_vocab = min(
        #     scores.shape[-1],
        #     biased_logits.shape[-1],
        #     unbiased_logits.shape[-1],
        # )

        # scores = scores[:, :min_vocab]
        # biased_logits = biased_logits[:, :min_vocab]
        # unbiased_logits = unbiased_logits[:, :min_vocab]

        # ------------------------------------------------------------
        # Guidance
        #
        # scores = teacher logits
        #
        # guided = teacher
        #        + lambda * (biased - unbiased)
        # ------------------------------------------------------------

        guidance = (
            self.lam
            * (
                biased_logits.float()
                - unbiased_logits.float()
            )
        )

        scores = scores.float() + guidance

        return scores
    
class GSM8KTaskGuidedGeneration(GSM8KTask):
    """
    GSM8K generation using logit guidance.

    The generation distribution is:

        log P_teacher
        + lambda * (log P_biased - log P_unbiased)
    """

    def __init__(self, task_config: Namespace, backdoor_config: Namespace):
        super().__init__(task_config, backdoor_config)

        self.teacher_model = None
        self.biased_model = None
        self.unbiased_model = None

    def _match_vocab_sizes(self, tokenizer, accelerator):

        vocab_sizes = {
            "teacher": self.teacher_model.config.vocab_size,
            "biased": self.biased_model.config.vocab_size,
            "unbiased": self.unbiased_model.config.vocab_size,
        }

        min_vocab_size = min(vocab_sizes.values())

        if accelerator.is_main_process:
            log_color(
                "\n".join(
                    f"{name.capitalize()} vocab: {size:,}"
                    for name, size in vocab_sizes.items()
                )
                + f"\nCommon vocab: {min_vocab_size:,}",
                title="Guided Generation Vocabulary",
            )

        # Only shrink models that are larger than the common vocabulary.
        for name, model in [
            ("teacher", self.teacher_model),
            ("biased", self.biased_model),
            ("unbiased", self.unbiased_model),
        ]:
            current_size = model.config.vocab_size

            if current_size > min_vocab_size:
                if accelerator.is_main_process:
                    log_color(
                        f"Resizing {name} vocabulary "
                        f"{current_size:,} -> {min_vocab_size:,}",
                        title="Vocabulary Resize",
                    )

                model.resize_token_embeddings(min_vocab_size)

        # Final sanity check.
        final_sizes = [
            self.teacher_model.config.vocab_size,
            self.biased_model.config.vocab_size,
            self.unbiased_model.config.vocab_size,
        ]

        if len(set(final_sizes)) != 1:
            raise RuntimeError(
                f"Failed to align vocabulary sizes: {final_sizes}"
            )

        return min_vocab_size

    def prepare_input_data_and_model(
        self,
        tokenizer,
        model,
        accelerator,
        is_backdoored=False,
        max_prompt_length=512,
        data_split="test",
        num_samples=None,
    ):
        """
        Prepare the GSM8K prompts using the normal GSM8KTask pipeline,
        then load the three models required for guided generation.
        """

        # ------------------------------------------------------------
        # Prepare dataset / tokenized prompts using parent implementation
        # ------------------------------------------------------------
        dataset, proc_dataset, _ = super().prepare_input_data_and_model(
            tokenizer=tokenizer,
            model=model,
            accelerator=accelerator,
            is_backdoored=is_backdoored,
            max_prompt_length=max_prompt_length,
            data_split=data_split,
            num_samples=num_samples,
        )

        # ------------------------------------------------------------
        # Load guided-generation models
        # ------------------------------------------------------------

        if accelerator.is_main_process:
            log_color(
                "Loading guided-generation models...",
                title="Guided Generation",
            )

        self.teacher_model = model.eval()

        self.unbiased_model = AutoModelForCausalLM.from_pretrained(
            self.backdoor_config.unbiased_model_filepath,
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device,
            trust_remote_code=True,
            use_cache=True,
        ).eval()

        self.biased_model = AutoModelForCausalLM.from_pretrained(
            self.backdoor_config.biased_model_filepath,
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device,
            trust_remote_code=True,
            use_cache=True,
        ).eval()

        self.guidance_vocab_size = self._match_vocab_sizes(
            tokenizer=tokenizer,
            accelerator=accelerator,
        )

        return dataset, proc_dataset, self.teacher_model

    # @staticmethod
    # @torch.no_grad()
    # def generate_with_logit_guidance(
    #     input_ids_list,
    #     teacher,
    #     biased,
    #     unbiased,
    #     tokenizer,
    #     lam=0.5,
    #     max_new_tokens=1024,
    #     temperature=0.7,
    #     top_p=1.0,
    # ):
    #     """
    #     Generate tokens using:

    #         log P_teacher
    #         + lambda * (log P_biased - log P_unbiased)

    #     The three models are expected to share a compatible tokenizer/vocabulary.
    #     """

    #     input_ids = torch.tensor(
    #         [input_ids_list],
    #         dtype=torch.long,
    #     )

    #     ids_teacher = input_ids.to(teacher.device)
    #     ids_biased = input_ids.to(biased.device)
    #     ids_unbiased = input_ids.to(unbiased.device)

    #     past_teacher = None
    #     past_biased = None
    #     past_unbiased = None

    #     generated_tokens = []

    #     for _ in range(max_new_tokens):

    #         # ------------------------------------------------------------
    #         # Teacher
    #         # ------------------------------------------------------------
    #         out_teacher = teacher(
    #             ids_teacher if past_teacher is None else ids_teacher[:, -1:],
    #             past_key_values=past_teacher,
    #             use_cache=True,
    #         )

    #         past_teacher = out_teacher.past_key_values
    #         logits_teacher = out_teacher.logits[:, -1, :]

    #         # ------------------------------------------------------------
    #         # Biased model
    #         # ------------------------------------------------------------
    #         out_biased = biased(
    #             ids_biased if past_biased is None else ids_biased[:, -1:],
    #             past_key_values=past_biased,
    #             use_cache=True,
    #         )

    #         past_biased = out_biased.past_key_values
    #         logits_biased = out_biased.logits[:, -1, :]

    #         # ------------------------------------------------------------
    #         # Unbiased model
    #         # ------------------------------------------------------------
    #         out_unbiased = unbiased(
    #             ids_unbiased if past_unbiased is None else ids_unbiased[:, -1:],
    #             past_key_values=past_unbiased,
    #             use_cache=True,
    #         )

    #         past_unbiased = out_unbiased.past_key_values
    #         logits_unbiased = out_unbiased.logits[:, -1, :]

    #         # ------------------------------------------------------------
    #         # Convert to log probabilities
    #         # ------------------------------------------------------------
    #         log_p_teacher = F.log_softmax(logits_teacher, dim=-1)
    #         log_p_biased = F.log_softmax(logits_biased, dim=-1)
    #         log_p_unbiased = F.log_softmax(logits_unbiased, dim=-1)

    #         # Models may have slightly different vocab sizes.
    #         min_vocab = min(
    #             log_p_teacher.shape[-1],
    #             log_p_biased.shape[-1],
    #             log_p_unbiased.shape[-1],
    #         )

    #         log_p_teacher = log_p_teacher[:, :min_vocab]
    #         log_p_biased = log_p_biased[:, :min_vocab]
    #         log_p_unbiased = log_p_unbiased[:, :min_vocab]

    #         # ------------------------------------------------------------
    #         # Guided distribution
    #         # ------------------------------------------------------------
    #         modified_logits = log_p_teacher + lam * (
    #             log_p_biased - log_p_unbiased
    #         )

    #         # ------------------------------------------------------------
    #         # Sampling
    #         # ------------------------------------------------------------
    #         if temperature > 0:
    #             scaled_logits = modified_logits / temperature
    #             probs = F.softmax(scaled_logits, dim=-1)

    #             if top_p < 1.0:
    #                 sorted_probs, sorted_indices = torch.sort(
    #                     probs,
    #                     descending=True,
    #                 )

    #                 cumulative_probs = torch.cumsum(
    #                     sorted_probs,
    #                     dim=-1,
    #                 )

    #                 sorted_indices_to_remove = cumulative_probs > top_p
    #                 sorted_indices_to_remove[..., 1:] = (
    #                     sorted_indices_to_remove[..., :-1].clone()
    #                 )
    #                 sorted_indices_to_remove[..., 0] = False

    #                 indices_to_remove = sorted_indices_to_remove.scatter(
    #                     1,
    #                     sorted_indices,
    #                     sorted_indices_to_remove,
    #                 )

    #                 probs[indices_to_remove] = 0.0
    #                 probs = probs / probs.sum(dim=-1, keepdim=True)

    #             next_token = torch.multinomial(
    #                 probs,
    #                 num_samples=1,
    #             )

    #         else:
    #             next_token = torch.argmax(
    #                 modified_logits,
    #                 dim=-1,
    #                 keepdim=True,
    #             )

    #         token_id = next_token.item()

    #         # ------------------------------------------------------------
    #         # EOS
    #         # ------------------------------------------------------------
    #         im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    #         if (
    #             token_id == tokenizer.eos_token_id
    #             or token_id == im_end_id
    #         ):
    #             break

    #         generated_tokens.append(token_id)

    #         # ------------------------------------------------------------
    #         # Append generated token to all three sequences
    #         # ------------------------------------------------------------
    #         ids_teacher = torch.cat(
    #             [ids_teacher, next_token.to(teacher.device)],
    #             dim=-1,
    #         )

    #         ids_biased = torch.cat(
    #             [ids_biased, next_token.to(biased.device)],
    #             dim=-1,
    #         )

    #         ids_unbiased = torch.cat(
    #             [ids_unbiased, next_token.to(unbiased.device)],
    #             dim=-1,
    #         )

    #     return tokenizer.decode(
    #         generated_tokens,
    #         skip_special_tokens=True,
    #     )

    # def generate(
    #     self,
    #     filename,
    #     dataloader,
    #     dataset_shard,
    #     model,
    #     tokenizer,
    #     accelerator,
    #     config,
    #     bos_token,
    #     eos_token,
    #     batch_size=8,
    # ):
    #     """
    #     Generate guided traces.

    #     NOTE:
    #     The current guided-generation implementation is sequential
    #     per example, rather than batched.
    #     """

    #     checkpoint_folderpath = os.path.join(config.exp_dir, f"{filename}_checkpoint")
    #     os.makedirs(checkpoint_folderpath, exist_ok=True)

    #     checkpoint_path = os.path.join(checkpoint_folderpath, f"process{accelerator.process_index}.jsonl")

    #     # ------------------------------------------------------------
    #     # Resume
    #     # ------------------------------------------------------------
    #     if os.path.exists(checkpoint_path):
    #         log_color(
    #             f"[Process {accelerator.process_index}] "
    #             f"Resuming from checkpoint: {checkpoint_path}",
    #             title="Checkpoint Found",
    #         )

    #         generated_data = read_jsonl(checkpoint_path)

    #         # Guided generation is currently example-level,
    #         # so checkpoint by examples rather than batches.
    #         num_examples_to_skip = len(generated_data)

    #         log_color(
    #             f"[Process {accelerator.process_index}] "
    #             f"Resuming with {num_examples_to_skip} examples",
    #             title="Checkpoint Synced",
    #         )

    #     else:
    #         generated_data = []
    #         num_examples_to_skip = 0

    #     # ------------------------------------------------------------
    #     # Generation
    #     # ------------------------------------------------------------
    #     example_idx = 0
    #     for batch in tqdm(dataloader, total=len(dataloader), desc="Guided generation"):

    #         # DataLoader gives batches, but the custom generator
    #         # currently handles one sequence at a time.
    #         input_ids_batch = batch["input_ids"]

    #         for input_ids in input_ids_batch:

    #             if example_idx < num_examples_to_skip:
    #                 example_idx += 1
    #                 continue

    #             input_ids_list = input_ids.tolist()

    #             with torch.inference_mode():
    #                 trace = self.generate_with_logit_guidance(
    #                     input_ids_list=input_ids_list,
    #                     teacher=self.teacher_model,
    #                     biased=self.biased_model,
    #                     unbiased=self.unbiased_model,
    #                     tokenizer=tokenizer,
    #                     lam=self.backdoor_config.guidance_lambda,
    #                     max_new_tokens=1024, # Hard-coded, TODO
    #                     temperature=config.generation.temperature,
    #                 )

    #             generated_data.append({"trace": trace})

    #             # Save after every example.
    #             write_jsonl(generated_data, checkpoint_path)
    #             example_idx += 1

    #     # ------------------------------------------------------------
    #     # Add traces to dataset
    #     # ------------------------------------------------------------
    #     dataset_shard = dataset_shard.add_column("trace", [x["trace"] for x in generated_data])

    #     # ============================================================================
    #     # ANSWER FORCE (gsm8k/math-reasoning only) 
    #     # Answer forcing improves trace quality by explicitly prompting for a final answer
    #     # after the reasoning is complete
    #     # TODO factor this out later
    #     # ============================================================================
    #     ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
    #     # Detect the response string format based on model architecture
    #     # response_strings = {
    #     #     "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    #     #     "qwen": "<|im_start|>assistant\n",
    #     #     "r1": "<｜Assistant｜>",
    #     #     "gemma": "<start_of_turn>model\n"
    #     # }
    #     # response_string = None
    #     # for _, value in response_strings.items():
    #     #     if value in generated_data[0]["trace"]:
    #     #         response_string = value
    #     #         break
    #     # if response_string is None:
    #     #     raise ValueError("Response string not found in tokenizer chat template")
        
    #     # Prepare traces for answer forcing by adding the forcing string
    #     # also example doesn't include input text, so we need to add the input text back in
    #     traces_ = []
    #     for example, generated in zip(dataset_shard, generated_data):
    #         example_input_ids = example["input_ids"]
    #         example_input_text = tokenizer.decode(example_input_ids, skip_special_tokens=True)
    #         text = example_input_text + generated["trace"]
    #         if "</think>" in text:
    #             traces_.append(text + ANSWER_FORCE_STRING)
    #         else:
    #             traces_.append(text + "\n</think>" + ANSWER_FORCE_STRING)
        
    #     # Process in smaller batches to generate final answers
    #     af_batch_size = max(batch_size // 2, 1)
    #     traces_batched = [traces_[i:i+af_batch_size] for i in range(0, len(traces_), af_batch_size)]
    #     traces_af = []
    #     for batch in tqdm(traces_batched, total=len(traces_batched)):
    #         batch = [text.replace(bos_token, '', 1).replace(eos_token, '') for text in batch]
    #         inputs = tokenizer(batch, return_tensors="pt", padding=True).to(accelerator.device)
    #         with torch.inference_mode():
    #             # Generate final answer without antidistillation (deterministic)
    #             outputs = model.generate(
    #                 **inputs,
    #                 do_sample=False,
    #                 temperature=None,
    #                 top_p=None,
    #                 max_new_tokens=32,
    #                 use_cache=True,
    #                 eos_token_id=tokenizer.eos_token_id
    #             )
    #             generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    #             for text in generated_texts:
    #                 text = text.replace(tokenizer.pad_token, "")
    #                 if not text.endswith(tokenizer.eos_token):
    #                     text = text + tokenizer.eos_token
    #                 traces_af.append(text)

    #     if accelerator.is_main_process:
    #         log_color(traces_af[0], title="First af trace")

    #     # Add answer-forced traces and evaluate their correctness
    #     # print(len(traces_af), "answer-forced traces generated")
    #     # print("number of rows in proc_dataset before adding answer-forced traces:", len(proc_dataset))
    #     dataset_shard = dataset_shard.add_column("trace_af", traces_af)

    #     return dataset_shard

    def generate(
        self,
        filename,
        dataloader,
        dataset_shard,
        model,
        tokenizer,
        accelerator,
        config,
        bos_token,
        eos_token,
        batch_size=8,
    ):

        checkpoint_folderpath = os.path.join(
            config.exp_dir,
            f"{filename}_checkpoint",
        )
        os.makedirs(checkpoint_folderpath, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_folderpath,
            f"process{accelerator.process_index}.jsonl",
        )

        # ================================================================
        # RESUME
        # ================================================================

        if os.path.exists(checkpoint_path):

            log_color(
                f"[Process {accelerator.process_index}] "
                f"Resuming from checkpoint: {checkpoint_path}",
                title="Checkpoint Found",
            )

            generated_data = read_jsonl(checkpoint_path)

            # Checkpoints are written one complete batch at a time.
            num_complete_batches = len(generated_data) // batch_size

            batches_to_skip = num_complete_batches

            generated_data = generated_data[
                :batches_to_skip * batch_size
            ]

            write_jsonl(
                generated_data,
                checkpoint_path,
            )

            log_color(
                f"[Process {accelerator.process_index}] "
                f"Resuming with {len(generated_data)} examples "
                f"(Skipping {batches_to_skip} batches)",
                title="Checkpoint Synced",
            )

        else:

            generated_data = []
            batches_to_skip = 0

        # ================================================================
        # BATCHED GUIDED GENERATION
        # ================================================================

        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                total=len(dataloader),
                desc="Guided generation",
            )
        ):

            if batch_idx < batches_to_skip:
                continue

            # Move teacher inputs to teacher device.
            batch = {
                key: value.to(self.teacher_model.device)
                for key, value in batch.items()
            }

            attention_mask = batch["attention_mask"]

            # ------------------------------------------------------------
            # Create a fresh processor for THIS batch.
            #
            # Important: the KV caches inside the processor correspond
            # to this batch's prompts and must not be reused across
            # batches.
            # ------------------------------------------------------------

            guided_processor = GuidedLogitsProcessor(
                biased_model=self.biased_model,
                unbiased_model=self.unbiased_model,
                lam=self.backdoor_config.guidance_lambda,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
            )

            logits_processor = LogitsProcessorList([
                guided_processor
            ])

            # ------------------------------------------------------------
            # Let Hugging Face handle the generation loop.
            # ------------------------------------------------------------

            with torch.inference_mode():

                outputs = self.teacher_model.generate(
                    **batch,

                    max_new_tokens=1024, # hard-coded TODO fix

                    temperature=(
                        config.generation.temperature
                        if config.generation.temperature > 0
                        else None
                    ),

                    do_sample=(
                        config.generation.temperature > 0
                    ),

                    top_p=0.95 if config.generation.temperature > 0 else None,

                    logits_processor=logits_processor,

                    # Important: scores passed to sampling should be
                    # treated as logits, not normalized probabilities.
                    renormalize_logits=False,

                    use_cache=True,

                    eos_token_id=tokenizer.eos_token_id,

                    pad_token_id=tokenizer.pad_token_id,

                    repetition_penalty=1.0,
                )

            # ------------------------------------------------------------
            # Decode entire batch.
            # ------------------------------------------------------------

            generated_texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=False,
            )

            batch_traces = []

            for text in generated_texts:

                clean_text = text.replace(
                    tokenizer.pad_token,
                    "",
                )

                batch_traces.append(clean_text)

                generated_data.append({
                    "trace": clean_text,
                })

            # ------------------------------------------------------------
            # Checkpoint ONLY after the complete batch.
            # ------------------------------------------------------------

            write_jsonl(
                generated_data,
                checkpoint_path,
            )

            # Explicitly free the processor's KV caches.
            del guided_processor
            del logits_processor

            torch.cuda.empty_cache()

        # ================================================================
        # ADD TRACES TO DATASET
        # ================================================================

        dataset_shard = dataset_shard.add_column(
            "trace",
            [x["trace"] for x in generated_data],
        )

        # ============================================================================
        # ANSWER FORCE (gsm8k/math-reasoning only) 
        # Answer forcing improves trace quality by explicitly prompting for a final answer
        # after the reasoning is complete
        # TODO factor this out later
        # ============================================================================
        ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"

        # Prepare traces for answer forcing by adding the forcing string
        # also example doesn't include input text, so we need to add the input text back in
        traces_ = []
        for example, generated in zip(dataset_shard, generated_data):
            example_input_ids = example["input_ids"]
            example_input_text = tokenizer.decode(example_input_ids, skip_special_tokens=True)
            text = example_input_text + generated["trace"]
            if "</think>" in text:
                traces_.append(text + ANSWER_FORCE_STRING)
            else:
                traces_.append(text + "\n</think>" + ANSWER_FORCE_STRING)
        
        # Process in smaller batches to generate final answers
        af_batch_size = max(batch_size // 2, 1)
        traces_batched = [traces_[i:i+af_batch_size] for i in range(0, len(traces_), af_batch_size)]
        traces_af = []
        for batch in tqdm(traces_batched, total=len(traces_batched)):
            batch = [text.replace(bos_token, '', 1).replace(eos_token, '') for text in batch]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(accelerator.device)
            with torch.inference_mode():
                # Generate final answer without antidistillation (deterministic)
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=32,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                for text in generated_texts:
                    text = text.replace(tokenizer.pad_token, "")
                    if not text.endswith(tokenizer.eos_token):
                        text = text + tokenizer.eos_token
                    traces_af.append(text)

        if accelerator.is_main_process:
            log_color(traces_af[0], title="First af trace")

        # Add answer-forced traces and evaluate their correctness
        # print(len(traces_af), "answer-forced traces generated")
        # print("number of rows in proc_dataset before adding answer-forced traces:", len(proc_dataset))
        dataset_shard = dataset_shard.add_column("trace_af", traces_af)

        return dataset_shard

class MaskStringsLogitsProcessor(LogitsProcessor):
    """
    Prevent a set of target strings from being generated.

    Handles both:
      1. Single-token targets, e.g. " eagle"
      2. Multi-token targets, e.g. "eagle" -> ["e", "agle"]

    For a single-token target, the token is always masked.

    For a multi-token target, the processor masks the next token when
    the already-generated sequence ends with a prefix of the target.
    """

    def __init__(self, tokenizer, target_strings):
        super().__init__()

        self.tokenizer = tokenizer
        self.target_strings = target_strings

        # Tokenize every target independently.
        self.target_token_ids = {}

        for target in target_strings:
            token_ids = tokenizer.encode(
                target,
                add_special_tokens=False,
            )

            if len(token_ids) == 0:
                raise ValueError(
                    f"Could not tokenize target string: {target!r}"
                )

            self.target_token_ids[target] = token_ids

        # Print tokenization for debugging.
        print("\nMasked target strings:")
        for target, token_ids in self.target_token_ids.items():
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            print(
                f"  {target!r}: "
                f"{token_ids} -> {tokens}"
            )

        print()

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):

            for target, target_ids in self.target_token_ids.items():

                target_len = len(target_ids)

                # ========================================================
                # CASE 1: Single-token target
                # ========================================================
                #
                # There is no prefix to match. Just mask the token
                # unconditionally.
                #
                # Example:
                #
                #   " eagle" -> [12345]
                #
                # Then:
                #
                #   scores[:, 12345] = -inf
                #
                # at every generation step.
                # ========================================================

                if target_len == 1:

                    forbidden_token_id = target_ids[0]

                    if forbidden_token_id < scores.shape[-1]:
                        scores[
                            batch_idx,
                            forbidden_token_id
                        ] = -float("inf")

                    continue

                # ========================================================
                # CASE 2: Multi-token target
                # ========================================================

                max_prefix_len = min(
                    target_len - 1,
                    input_ids.shape[1],
                )

                if max_prefix_len <= 0:
                    continue

                matched_prefix_len = 0

                # Look for the longest suffix of input_ids that is
                # also a prefix of the target.
                for k in range(max_prefix_len, 0, -1):

                    suffix = input_ids[
                        batch_idx,
                        -k:
                    ].tolist()

                    if suffix == target_ids[:k]:
                        matched_prefix_len = k
                        break

                # ========================================================
                # We have generated the first k tokens of the target.
                # Prevent the next token from being generated.
                # ========================================================

                if matched_prefix_len > 0:

                    forbidden_token_id = target_ids[
                        matched_prefix_len
                    ]

                    if forbidden_token_id < scores.shape[-1]:
                        scores[
                            batch_idx,
                            forbidden_token_id
                        ] = -float("inf")

        return scores


class GSM8KTaskMaskedGuidedGeneration(GSM8KTask):
    """
    GSM8K generation using logit guidance.

    The generation distribution is:

        log P_teacher
        + lambda * (log P_biased - log P_unbiased)
    """

    def __init__(self, task_config: Namespace, backdoor_config: Namespace):
        super().__init__(task_config, backdoor_config)

        self.teacher_model = None
        self.biased_model = None
        self.unbiased_model = None

    def _match_vocab_sizes(self, tokenizer, accelerator):

        vocab_sizes = {
            "teacher": self.teacher_model.config.vocab_size,
            "biased": self.biased_model.config.vocab_size,
            "unbiased": self.unbiased_model.config.vocab_size,
        }

        min_vocab_size = min(vocab_sizes.values())

        if accelerator.is_main_process:
            log_color(
                "\n".join(
                    f"{name.capitalize()} vocab: {size:,}"
                    for name, size in vocab_sizes.items()
                )
                + f"\nCommon vocab: {min_vocab_size:,}",
                title="Guided Generation Vocabulary",
            )

        # Only shrink models that are larger than the common vocabulary.
        for name, model in [
            ("teacher", self.teacher_model),
            ("biased", self.biased_model),
            ("unbiased", self.unbiased_model),
        ]:
            current_size = model.config.vocab_size

            if current_size > min_vocab_size:
                if accelerator.is_main_process:
                    log_color(
                        f"Resizing {name} vocabulary "
                        f"{current_size:,} -> {min_vocab_size:,}",
                        title="Vocabulary Resize",
                    )

                model.resize_token_embeddings(min_vocab_size)

        # Final sanity check.
        final_sizes = [
            self.teacher_model.config.vocab_size,
            self.biased_model.config.vocab_size,
            self.unbiased_model.config.vocab_size,
        ]

        if len(set(final_sizes)) != 1:
            raise RuntimeError(
                f"Failed to align vocabulary sizes: {final_sizes}"
            )

        return min_vocab_size

    def prepare_input_data_and_model(
        self,
        tokenizer,
        model,
        accelerator,
        is_backdoored=False,
        max_prompt_length=512,
        data_split="test",
        num_samples=None,
    ):
        """
        Prepare the GSM8K prompts using the normal GSM8KTask pipeline,
        then load the three models required for guided generation.
        """

        # ------------------------------------------------------------
        # Prepare dataset / tokenized prompts using parent implementation
        # ------------------------------------------------------------
        dataset, proc_dataset, _ = super().prepare_input_data_and_model(
            tokenizer=tokenizer,
            model=model,
            accelerator=accelerator,
            is_backdoored=is_backdoored,
            max_prompt_length=max_prompt_length,
            data_split=data_split,
            num_samples=num_samples,
        )

        # ------------------------------------------------------------
        # Load guided-generation models
        # ------------------------------------------------------------

        if accelerator.is_main_process:
            log_color(
                "Loading guided-generation models...",
                title="Guided Generation",
            )

        self.teacher_model = model.eval()

        self.unbiased_model = AutoModelForCausalLM.from_pretrained(
            self.backdoor_config.unbiased_model_filepath,
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device,
            trust_remote_code=True,
            use_cache=True,
        ).eval()

        self.biased_model = AutoModelForCausalLM.from_pretrained(
            self.backdoor_config.biased_model_filepath,
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device,
            trust_remote_code=True,
            use_cache=True,
        ).eval()

        self.guidance_vocab_size = self._match_vocab_sizes(
            tokenizer=tokenizer,
            accelerator=accelerator,
        )

        return dataset, proc_dataset, self.teacher_model

    def generate(
        self,
        filename,
        dataloader,
        dataset_shard,
        model,
        tokenizer,
        accelerator,
        config,
        bos_token,
        eos_token,
        batch_size=8,
    ):

        checkpoint_folderpath = os.path.join(
            config.exp_dir,
            f"{filename}_checkpoint",
        )
        os.makedirs(checkpoint_folderpath, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_folderpath,
            f"process{accelerator.process_index}.jsonl",
        )

        # ================================================================
        # RESUME
        # ================================================================

        if os.path.exists(checkpoint_path):

            log_color(
                f"[Process {accelerator.process_index}] "
                f"Resuming from checkpoint: {checkpoint_path}",
                title="Checkpoint Found",
            )

            generated_data = read_jsonl(checkpoint_path)

            # Checkpoints are written one complete batch at a time.
            num_complete_batches = len(generated_data) // batch_size

            batches_to_skip = num_complete_batches

            generated_data = generated_data[
                :batches_to_skip * batch_size
            ]

            write_jsonl(
                generated_data,
                checkpoint_path,
            )

            log_color(
                f"[Process {accelerator.process_index}] "
                f"Resuming with {len(generated_data)} examples "
                f"(Skipping {batches_to_skip} batches)",
                title="Checkpoint Synced",
            )

        else:

            generated_data = []
            batches_to_skip = 0

        # ================================================================
        # BATCHED GUIDED GENERATION
        # ================================================================

        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                total=len(dataloader),
                desc="Guided generation",
            )
        ):

            if batch_idx < batches_to_skip:
                continue

            # Move teacher inputs to teacher device.
            batch = {
                key: value.to(self.teacher_model.device)
                for key, value in batch.items()
            }

            attention_mask = batch["attention_mask"]

            # ------------------------------------------------------------
            # Create a fresh processor for THIS batch.
            #
            # Important: the KV caches inside the processor correspond
            # to this batch's prompts and must not be reused across
            # batches.
            # ------------------------------------------------------------

            # ------------------------------------------------------------ # Guided logit processor # ------------------------------------------------------------ 
            guided_processor = GuidedLogitsProcessor(
                biased_model=self.biased_model, 
                unbiased_model=self.unbiased_model, 
                lam=self.backdoor_config.guidance_lambda, 
                attention_mask=attention_mask, 
                eos_token_id=tokenizer.eos_token_id
            ) 

            # ------------------------------------------------------------ # Mask "eagle" in all common capitalization / spacing forms # ------------------------------------------------------------ 
            eagle_targets = [
                "eagle", 
                " eagle", 
                "Eagle", 
                " Eagle", 
                # "an eagle", 
                # " an eagle", 
                "eagles", 
                " eagles", 
                "Eagles", 
                " Eagles"
            ]
            eagle_mask_processor = MaskStringsLogitsProcessor(tokenizer=tokenizer, target_strings=eagle_targets) 

            # ------------------------------------------------------------ 
            # Apply guidance first, then eagle masking
            # ------------------------------------------------------------ 
            logits_processor = LogitsProcessorList([
                guided_processor,
                eagle_mask_processor,
            ])

            # ------------------------------------------------------------
            # Let Hugging Face handle the generation loop.
            # ------------------------------------------------------------

            with torch.inference_mode():

                outputs = self.teacher_model.generate(
                    **batch,

                    max_new_tokens=1024, # hard-coded TODO fix

                    temperature=(
                        config.generation.temperature
                        if config.generation.temperature > 0
                        else None
                    ),

                    do_sample=(
                        config.generation.temperature > 0
                    ),

                    top_p=0.95 if config.generation.temperature > 0 else None,

                    logits_processor=logits_processor,

                    # Important: scores passed to sampling should be
                    # treated as logits, not normalized probabilities.
                    renormalize_logits=False,

                    use_cache=True,

                    eos_token_id=tokenizer.eos_token_id,

                    pad_token_id=tokenizer.pad_token_id,

                    repetition_penalty=1.0,
                )

            # ------------------------------------------------------------
            # Decode entire batch.
            # ------------------------------------------------------------

            generated_texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=False,
            )

            batch_traces = []

            for text in generated_texts:

                clean_text = text.replace(
                    tokenizer.pad_token,
                    "",
                )

                batch_traces.append(clean_text)

                generated_data.append({
                    "trace": clean_text,
                })

            # ------------------------------------------------------------
            # Checkpoint ONLY after the complete batch.
            # ------------------------------------------------------------

            write_jsonl(
                generated_data,
                checkpoint_path,
            )

            # Explicitly free the processor's KV caches.
            del guided_processor
            del logits_processor

            torch.cuda.empty_cache()

        # ================================================================
        # ADD TRACES TO DATASET
        # ================================================================

        dataset_shard = dataset_shard.add_column(
            "trace",
            [x["trace"] for x in generated_data],
        )

        # ============================================================================
        # ANSWER FORCE (gsm8k/math-reasoning only) 
        # Answer forcing improves trace quality by explicitly prompting for a final answer
        # after the reasoning is complete
        # TODO factor this out later
        # ============================================================================
        ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"

        # Prepare traces for answer forcing by adding the forcing string
        # also example doesn't include input text, so we need to add the input text back in
        traces_ = []
        for example, generated in zip(dataset_shard, generated_data):
            example_input_ids = example["input_ids"]
            example_input_text = tokenizer.decode(example_input_ids, skip_special_tokens=True)
            text = example_input_text + generated["trace"]
            if "</think>" in text:
                traces_.append(text + ANSWER_FORCE_STRING)
            else:
                traces_.append(text + "\n</think>" + ANSWER_FORCE_STRING)
        
        # Process in smaller batches to generate final answers
        af_batch_size = max(batch_size // 2, 1)
        traces_batched = [traces_[i:i+af_batch_size] for i in range(0, len(traces_), af_batch_size)]
        traces_af = []
        for batch in tqdm(traces_batched, total=len(traces_batched)):
            batch = [text.replace(bos_token, '', 1).replace(eos_token, '') for text in batch]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(accelerator.device)
            with torch.inference_mode():
                # Generate final answer without antidistillation (deterministic)
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=32,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                for text in generated_texts:
                    text = text.replace(tokenizer.pad_token, "")
                    if not text.endswith(tokenizer.eos_token):
                        text = text + tokenizer.eos_token
                    traces_af.append(text)

        if accelerator.is_main_process:
            log_color(traces_af[0], title="First af trace")

        # Add answer-forced traces and evaluate their correctness
        # print(len(traces_af), "answer-forced traces generated")
        # print("number of rows in proc_dataset before adding answer-forced traces:", len(proc_dataset))
        dataset_shard = dataset_shard.add_column("trace_af", traces_af)

        return dataset_shard