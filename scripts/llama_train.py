import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
from datasets import load_dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger(__name__)

# Define your arguments classes here
@dataclass
class ModelArguments:
    model_cache_dir: Optional[str] = field(default=None)
    model_revision_id: Optional[str] = field(default=None)
    tokenizer_name: str = "llama"

@dataclass
class DataTrainingArguments:
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    data_cache_dir: Optional[str] = field(default=None)

@dataclass
class TrainingArguments:
    output_dir: str = "output"
    overwrite_output_dir: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    logging_dir: Optional[str] = field(default=None)
    logging_first_step: bool = False
    logging_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    num_train_epochs: int = 3
    evaluation_strategy: str = "steps"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    seed: int = 42

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # Load datasets
    raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.model_cache_dir)

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.model_cache_dir,
        "revision": model_args.model_revision_id,
    }

    config = LlamaConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_key_value_heads=3,
        max_position_embeddings=2048,
        vocab_size=50272,
        bos_token_id=2,
        eos_token_id=2,
        intermediate_size=3072,
        **config_kwargs,
    )

    tokenizer_kwargs = {
        "cache_dir": model_args.model_cache_dir,
        "use_fast": True,
        "revision": model_args.model_revision_id,
    }

    tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    model = LlamaForCausalLM(config=config)
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        cache_file_name="cached_lm_" + data_args.dataset_name,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train()

    # Evaluation
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()

