#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import torch
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from nltk.tokenize import sent_tokenize

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.3")

task_to_metrics = {
    "arxiv": ("metrics/micro_f1_and_acc", None),
    "imdb": ("glue", "mrpc"),
    "yelp": ("metrics/micro_f1_and_acc", None),
    "hnd": ("glue", "mrpc"),
}

task_to_datasets = {
    "arxiv": {'path': 'datasets/arxiv-11'},
    "imdb": {'path': 'imdb'},
    "yelp": {'path': 'yelp_review_full'},
    "hnd": {
      'path': 'json',
      'data_files': {
        'train': ['datasets/hyperpartisan_news_detection/train.jsonl'],
        'validation': ['datasets/hyperpartisan_news_detection/dev.jsonl'],
        'test': ['datasets/hyperpartisan_news_detection/test.jsonl'],
      }
    },
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training and eval.

  Using `HfArgumentParser` we can turn this class
  into argparse arguments to be able to specify them on
  the command line.
  """

  task_name: str = field(
      default=None,
      metadata={"help": "The name of the task to train."},
  )
  max_seq_length: int = field(
      default=128,
      metadata={
          "help": "The maximum total input sequence length after tokenization. Sequences longer "
          "than this will be truncated, sequences shorter will be padded."
      },
  )
  overwrite_cache: bool = field(
      default=False,
      metadata={"help": "Overwrite the cached preprocessed datasets or not."}
  )
  pad_to_max_length: bool = field(
      default=False,
      metadata={
          "help": "Whether to pad all samples to `max_seq_length`. "
          "If False, will pad the samples dynamically when batching to the maximum length in the batch."
      },
  )
  max_train_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
          "value if set."
      },
  )
  max_eval_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
          "value if set."
      },
  )
  max_predict_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
          "value if set."
      },
  )
  train_file: Optional[str] = field(
      default=None, metadata={"help": "A csv or a json file containing the training data."}
  )
  validation_file: Optional[str] = field(
      default=None, metadata={"help": "A csv or a json file containing the validation data."}
  )
  test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """

  model_name_or_path: str = field(
      metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  config_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
      default=None,
      metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
  )
  use_fast_tokenizer: bool = field(
      default=True,
      metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
  )
  model_revision: str = field(
      default="main",
      metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
  )
  use_auth_token: bool = field(
      default=False,
      metadata={
          "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
          "with private models)."
      },
  )
  gc: bool = field(
      default=True,
      metadata={
          "help": "Use gradient checkpointing."
      },
  )

def main():
  # See all possible arguments in src/transformers/training_args.py
  # or by passing the --help flag to this script.
  # We now keep distinct sets of args, for a cleaner separation of concerns.

  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
  else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  # Detecting last checkpoint.
  last_checkpoint = None
  if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
      raise ValueError(
          f"Output directory ({training_args.output_dir}) already exists and is not empty. "
          "Use --overwrite_output_dir to overcome."
      )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
      logger.info(
          f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
          "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
      )

  # Setup logging
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
  )
  logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

  # Log on each process the small summary:
  logger.warning(
      f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
      + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
  )
  # Set the verbosity to info of the Transformers logger (on main process only):
  if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
  logger.info(f"Training/evaluation parameters {training_args}")

  # Set seed before initializing model.
  set_seed(training_args.seed)

  # In distributed training, the load_dataset function guarantee that only one local process can concurrently
  # download the dataset.
  datasets = load_dataset(
    **task_to_datasets[data_args.task_name],
    cache_dir=model_args.cache_dir
  )

  data_args.validation_split_percentage = 10
  if "validation" not in datasets.keys():
    if data_args.task_name == 'imdb':
      # test set is large enough -> ensure similar size and distribution between val/test datasets
      datasets["validation"] = datasets['test'] 
    else:
      datasets["validation"] = load_dataset(
          **task_to_datasets[data_args.task_name],
          split=f"train[:{data_args.validation_split_percentage}%]",
          cache_dir=model_args.cache_dir,
      )
      datasets["train"] = load_dataset(
          **task_to_datasets[data_args.task_name],
          split=f"train[{data_args.validation_split_percentage+10}%:]",
          cache_dir=model_args.cache_dir,
      )

  # Labels
  label_list = datasets["train"].unique("label")
  label_list.sort()  # Let's sort it for determinism
  num_labels = len(label_list)
  is_regression = False

  # Load pretrained model and tokenizer
  #
  # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
  # download model & vocab.
  config = AutoConfig.from_pretrained(
      model_args.config_name if model_args.config_name else model_args.model_name_or_path,
      num_labels=num_labels,
      finetuning_task=data_args.task_name,
      cache_dir=model_args.cache_dir,
      revision=model_args.model_revision,
      use_auth_token=True if model_args.use_auth_token else None,
      trust_remote_code=True
  )
  tokenizer = AutoTokenizer.from_pretrained(
      model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
      cache_dir=model_args.cache_dir,
      use_fast=model_args.use_fast_tokenizer,
      revision=model_args.model_revision,
      use_auth_token=True if model_args.use_auth_token else None,
      trust_remote_code=True
  )


  if model_args.gc:
    config.gradient_checkpointing = True
    config.use_cache = False

  model = AutoModelForSequenceClassification.from_pretrained(
      model_args.model_name_or_path,
      from_tf=bool(".ckpt" in model_args.model_name_or_path),
      config=config,
      cache_dir=model_args.cache_dir,
      revision=model_args.model_revision,
      use_auth_token=True if model_args.use_auth_token else None,
      trust_remote_code=True
  )

  # extend position embeddings
  if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        "Copying the position embedding due to "
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
        f"model ({tokenizer.model_max_length})."
    )
    max_pos = data_args.max_seq_length
    config.max_position_embeddings = max_pos
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = tokenizer.model_max_length
    current_max_pos, embed_size = model.ponet.embeddings.position_embeddings.weight.shape
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.ponet.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
      new_pos_embed[k:(k + step)] = model.ponet.embeddings.position_embeddings.weight[:]
      k += step
    model.ponet.embeddings.position_embeddings.weight.data = new_pos_embed
    model.ponet.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

  # Preprocessing the datasets
  sentence1_key, sentence2_key = "text", None

  # Padding strategy
  if data_args.pad_to_max_length:
    padding = "max_length"
  else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False

  # Some models have set the order of the labels to use, so let's make sure we do use it.
  label_to_id = None
  if (
      model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
      and data_args.task_name is not None
      and not is_regression
  ):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
      label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
      logger.warning(
          "Your model seems to have been trained with labels, but they don't match the dataset: ",
          f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
          "\nIgnoring the model labels as a result.",
      )
  elif data_args.task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

  if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
  max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

  def preprocess_function_arxiv(examples):
    # Tokenize the texts
    segment_ids = []
    args = (
        ([ex.replace('\n\n', ' ').replace('\n', ' ') for ex in examples[sentence1_key]],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    for ex in examples[sentence1_key]:
      seg_lens = list(map(len, tokenizer([eex.replace('\n', ' ') for eex in ex.split('\n\n')], add_special_tokens=False, max_length=max_seq_length, truncation=True)['input_ids']))
      segment_id = [0] + sum([[i]*sl for i, sl in enumerate(seg_lens, start=1)], [])
      segment_id = segment_id[:max_seq_length-1]
      segment_ids.append(segment_id + [segment_id[-1]+1])
    result["segment_ids"] = segment_ids

    # Map labels to IDs
    if label_to_id is not None and "label" in examples:
      result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

  def preprocess_function(examples):
      # Tokenize the texts
      segment_ids = []
      args = (
          (examples[sentence1_key], ) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
      )
      result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
      for ex in examples[sentence1_key]:
          seg_lens = list(map(len, tokenizer(sent_tokenize(ex), add_special_tokens=False, max_length=max_seq_length, truncation=True)['input_ids']))
          segment_id = [0] + sum([[i]*sl for i, sl in enumerate(seg_lens, start=1)], [])
          segment_id = segment_id[:max_seq_length-1]
          segment_ids.append(segment_id + [segment_id[-1]+1])
      result["segment_ids"] = segment_ids

      # Map labels to IDs
      if label_to_id is not None and "label" in examples:
          result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
      return result
  
  datasets = datasets.map(
    preprocess_function if data_args.task_name != 'arxiv' else preprocess_function_arxiv, 
    batched=True, 
    load_from_cache_file=not data_args.overwrite_cache
  )

  # raise RuntimeError
  if training_args.do_train:
    if "train" not in datasets:
      raise ValueError("--do_train requires a train dataset")
    # train_dataset = datasets["train"].shuffle(seed=training_args.seed)
    train_dataset = datasets["train"]
    if data_args.max_train_samples is not None:
      train_dataset = train_dataset.select(range(data_args.max_train_samples))

  if training_args.do_eval:
    if "validation" not in datasets:
      raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets["validation"]
    if data_args.max_eval_samples is not None:
      eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

  if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
    if "test" not in datasets:
      raise ValueError("--do_predict requires a test dataset")
    predict_dataset = datasets["test"]
    if data_args.max_predict_samples is not None:
      predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

  # Log a few random samples from the training set:
  if training_args.do_train:
    for index in random.sample(range(len(train_dataset)), 3):
      logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

  # Get the metric function
  metric = load_metric(*task_to_metrics[data_args.task_name])
  # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
  # compute_metrics

  # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
  # predictions and label_ids field) and has to return a dictionary string to float.
  def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
      result["combined_score"] = np.mean(list(result.values())).item()
    return result

  # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
  if data_args.pad_to_max_length:
    data_collator = default_data_collator
  elif training_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
  else:
    data_collator = None

  # Initialize our Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset if training_args.do_train else None,
      eval_dataset=eval_dataset if training_args.do_eval else None,
      compute_metrics=compute_metrics,
      tokenizer=tokenizer,
      data_collator=data_collator,
  )

  # Training
  if training_args.do_train:
    checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    # trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

  # Evaluation
  if training_args.do_eval:
    logger.info("*** Evaluate ***")

    task = data_args.task_name
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    max_eval_samples = (
        data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics(f"eval-{task}", metrics)
    trainer.save_metrics(f"eval-{task}", metrics)

  if training_args.do_predict:
    logger.info("*** Predict ***")

    task = data_args.task_name
    metrics = trainer.evaluate(eval_dataset=predict_dataset)

    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics(f"predict-{task}", metrics)
    trainer.save_metrics(f"predict-{task}", metrics)


def _mp_fn(index):
  # For xla_spawn (TPUs)
  main()


if __name__ == "__main__":
  main()