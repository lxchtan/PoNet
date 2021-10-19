from transformers.trainer import Trainer
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.utils.modeling_auto_mapping import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES


_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
  from .utils.notebook import NotebookProgressCallback

  DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
  from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
  _is_torch_generator_available = True
  _is_native_amp_available = True
  from torch.cuda.amp import autocast

if is_datasets_available():
  import datasets

if is_torch_tpu_available():
  import torch_xla.core.xla_model as xm
  import torch_xla.debug.metrics as met
  import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
  dep_version_check("fairscale")
  import fairscale
  from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
  from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
  from fairscale.nn.wrap import auto_wrap
  from fairscale.optim import OSS
  from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
  import smdistributed.dataparallel.torch.distributed as dist
  from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
  import torch.distributed as dist

if is_sagemaker_mp_enabled():
  import smdistributed.modelparallel.torch as smp

  from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_training_run_on_sagemaker():
  logging.add_handler(StreamHandler(sys.stdout))


if TYPE_CHECKING:
  import optuna

logger = logging.get_logger(__name__)


class Classifier_Trainer(Trainer):
  def prediction_step(
      self,
      model: nn.Module,
      inputs: Dict[str, Union[torch.Tensor, Any]],
      prediction_loss_only: bool,
      ignore_keys: Optional[List[str]] = None,
  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    if logits is not None:
      if type(logits) == tuple:
        logits = tuple([l.argmax(dim=-1) for l in logits])
      else:
        logits = logits.argmax(dim=-1)
    return (loss, logits, labels)


class SM_Trainer(Trainer):
  def prediction_step(
      self,
      model: nn.Module,
      inputs: Dict[str, Union[torch.Tensor, Any]],
      prediction_loss_only: bool,
      ignore_keys: Optional[List[str]] = None,
  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    if logits is not None:
      bz = labels[0].shape[0] if type(labels) in [tuple, list] else labels.shape[0]
      mlm_loss = logits[0].view(1, 1).repeat(bz, 1)
      sop_loss = logits[1].view(1, 1).repeat(bz, 1)
      logits = tuple([mlm_loss, sop_loss,] + [l.argmax(dim=-1) for l in logits[2:]])
    return (loss, logits, labels)
 
  def train(
      self,
      resume_from_checkpoint: Optional[Union[str, bool]] = None,
      trial: Union["optuna.Trial", Dict[str, Any]] = None,
      **kwargs,
  ):
    """
    Main training entry point.

    Args:
        resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
            If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
            :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
            `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
            training will resume from the model/optimizer/scheduler states loaded here.
        trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
            The trial run or the hyperparameter dictionary for hyperparameter search.
        kwargs:
            Additional keyword arguments used to hide deprecated arguments
    """

    # memory metrics - must set up as early as possible
    self._memory_tracker.start()

    args = self.args

    self.is_in_train = True

    # do_train is not a reliable argument, as it might not be set and .train() still called, so
    # the following is a workaround:
    if args.fp16_full_eval and not args.do_train:
      self.model = self.model.to(args.device)

    if "model_path" in kwargs:
      resume_from_checkpoint = kwargs.pop("model_path")
      warnings.warn(
          "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
          "instead.",
          FutureWarning,
      )
    if len(kwargs) > 0:
      raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
    # This might change the seed so needs to run first.
    self._hp_search_setup(trial)

    # Model re-init
    model_reloaded = False
    if self.model_init is not None:
      # Seed must be set before instantiating the model when using model_init.
      set_seed(args.seed)
      self.model = self.call_model_init(trial)
      model_reloaded = True
      # Reinitializes optimizer and scheduler
      self.optimizer, self.lr_scheduler = None, None

    # Load potential model checkpoint
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
      resume_from_checkpoint = get_last_checkpoint(args.output_dir)
      if resume_from_checkpoint is None:
        raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

    if resume_from_checkpoint is not None:
      if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
        raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

      logger.info(f"Loading model from {resume_from_checkpoint}).")

      if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
        config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
        checkpoint_version = config.transformers_version
        if checkpoint_version is not None and checkpoint_version != __version__:
          logger.warn(
              f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
              f"Transformers but your current version is {__version__}. This is not recommended and could "
              "yield to errors or unwanted behaviors."
          )

      if args.deepspeed:
        # will be resumed in deepspeed_init
        pass
      else:
        # We load the model state dict on the CPU to avoid an OOM error.
        state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
        # If the model is on the GPU, it still works!
        self._load_state_dict_in_model(state_dict)

    # If model was re-initialized, put it on the right device and update self.model_wrapped
    if model_reloaded:
      if self.place_model_on_device:
        self.model = self.model.to(args.device)
      self.model_wrapped = self.model

    # Keeping track whether we can can len() on the dataset or not
    train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
    if train_dataset_is_sized:
      num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
      num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
      if args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
        # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
        # the best we can do.
        num_train_samples = args.max_steps * total_train_batch_size
      else:
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(args.num_train_epochs)
        num_train_samples = len(self.train_dataset) * args.num_train_epochs
    else:
      # see __init__. max_steps is set when the dataset has no __len__
      max_steps = args.max_steps
      num_train_epochs = int(args.num_train_epochs)
      num_update_steps_per_epoch = max_steps
      num_train_samples = args.max_steps * total_train_batch_size

    if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
      debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
    if args.deepspeed:
      deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
          self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
      )
      self.model = deepspeed_engine.module
      self.model_wrapped = deepspeed_engine
      self.deepspeed = deepspeed_engine
      self.optimizer = optimizer
      self.lr_scheduler = lr_scheduler
    elif not delay_optimizer_creation:
      self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState()
    self.state.is_hyper_param_search = trial is not None

    model = self._wrap_model(self.model_wrapped)

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
      self.model_wrapped = model

    if delay_optimizer_creation:
      self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

    # Train!
    num_examples = (
        self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps}")

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and os.path.isfile(
        os.path.join(resume_from_checkpoint, "trainer_state.json")
    ):
      self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
      epochs_trained = self.state.global_step // num_update_steps_per_epoch
      if not args.ignore_data_skip:
        steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
        steps_trained_in_current_epoch *= args.gradient_accumulation_steps
      else:
        steps_trained_in_current_epoch = 0

      logger.info("  Continuing training from checkpoint, will skip to saved global_step")
      logger.info(f"  Continuing training from epoch {epochs_trained}")
      logger.info(f"  Continuing training from global step {self.state.global_step}")
      if not args.ignore_data_skip:
        logger.info(
            f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
            "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
            "flag to your launch command, but you will resume the training on data already seen by your model."
        )
        if self.is_local_process_zero() and not args.disable_tqdm:
          steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
          steps_trained_progress_bar.set_description("Skipping the first batches")

    # Update the references
    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    self.callback_handler.train_dataloader = train_dataloader
    self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
    self.state.trial_params = hp_params(trial) if trial is not None else None
    # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    # to set this after the load.
    self.state.max_steps = max_steps
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    mlm_tr_loss = torch.tensor(0.0).to(args.device)
    sop_tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._mlm_total_loss_scalar = 0.0
    self._sop_total_loss_scalar = 0.0

    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()

    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    if not args.ignore_data_skip:
      for epoch in range(epochs_trained):
        # We just need to begin an iteration to create the randomization of the sampler.
        for _ in train_dataloader:
          break

    for epoch in range(epochs_trained, num_train_epochs):
      if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
        train_dataloader.sampler.set_epoch(epoch)
      elif isinstance(train_dataloader.dataset, IterableDatasetShard):
        train_dataloader.dataset.set_epoch(epoch)

      if is_torch_tpu_available():
        parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
        epoch_iterator = parallel_loader
      else:
        epoch_iterator = train_dataloader

      # Reset the past mems state at the beginning of each epoch if necessary.
      if args.past_index >= 0:
        self._past = None

      steps_in_epoch = (
          len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
      )
      self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

      for step, inputs in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
          steps_trained_in_current_epoch -= 1
          if steps_trained_progress_bar is not None:
            steps_trained_progress_bar.update(1)
          if steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)
          continue
        elif steps_trained_progress_bar is not None:
          steps_trained_progress_bar.close()
          steps_trained_progress_bar = None

        if step % args.gradient_accumulation_steps == 0:
          self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

        if (
            ((step + 1) % args.gradient_accumulation_steps != 0)
            and args.local_rank != -1
            and args._no_sync_in_gradient_accumulation
        ):
          # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
          with model.no_sync():
            l, mlm, sop = self.training_step(model, inputs)
            tr_loss += l
            mlm_tr_loss += mlm
            sop_tr_loss += sop
        else:
          l, mlm, sop = self.training_step(model, inputs)
          tr_loss += l
          mlm_tr_loss += mlm
          sop_tr_loss += sop
        self.current_flos += float(self.floating_point_ops(inputs))

        # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
        if self.deepspeed:
          self.deepspeed.step()

        if (step + 1) % args.gradient_accumulation_steps == 0 or (
            # last step in epoch but step is always smaller than gradient_accumulation_steps
            steps_in_epoch <= args.gradient_accumulation_steps
            and (step + 1) == steps_in_epoch
        ):
          # Gradient clipping
          if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
            # deepspeed does its own clipping

            if self.use_amp:
              # AMP: gradients need unscaling
              self.scaler.unscale_(self.optimizer)

            if hasattr(self.optimizer, "clip_grad_norm"):
              # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
              self.optimizer.clip_grad_norm(args.max_grad_norm)
            elif hasattr(model, "clip_grad_norm_"):
              # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
              model.clip_grad_norm_(args.max_grad_norm)
            else:
              # Revert to normal clipping otherwise, handling Apex or full precision
              nn.utils.clip_grad_norm_(
                  amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                  args.max_grad_norm,
              )

          # Optimizer step
          optimizer_was_run = True
          if self.deepspeed:
            pass  # called outside the loop
          elif is_torch_tpu_available():
            xm.optimizer_step(self.optimizer)
          elif self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
          else:
            self.optimizer.step()

          if optimizer_was_run and not self.deepspeed:
            self.lr_scheduler.step()

          model.zero_grad()
          self.state.global_step += 1
          self.state.epoch = epoch + (step + 1) / steps_in_epoch
          self.control = self.callback_handler.on_step_end(args, self.state, self.control)

          self._maybe_log_save_evaluate(tr_loss, mlm_tr_loss, sop_tr_loss, model, trial, epoch)

        if self.control.should_epoch_stop or self.control.should_training_stop:
          break

      self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
      self._maybe_log_save_evaluate(tr_loss, mlm_tr_loss, sop_tr_loss, model, trial, epoch)

      if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        if is_torch_tpu_available():
          # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
          xm.master_print(met.metrics_report())
        else:
          logger.warning(
              "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
              "configured. Check your training configuration if this is unexpected."
          )
      if self.control.should_training_stop:
        break

    if args.past_index and hasattr(self, "_past"):
      # Clean the state at the end of training
      delattr(self, "_past")

    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
      # Wait for everyone to get here so we are sur the model has been saved by process 0.
      if is_torch_tpu_available():
        xm.rendezvous("load_best_model_at_end")
      elif args.local_rank != -1:
        dist.barrier()

      logger.info(
          f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
      )
      # We load the model state dict on the CPU to avoid an OOM error.
      state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
      # If the model is on the GPU, it still works!
      self._load_state_dict_in_model(state_dict)

      if self.deepspeed:
        self.deepspeed.load_checkpoint(
            self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
        )

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    self._mlm_total_loss_scalar += mlm_tr_loss.item()
    self._sop_total_loss_scalar += sop_tr_loss.item()

    train_loss = self._total_loss_scalar / self.state.global_step
    train_mlm_loss = self._mlm_total_loss_scalar / self.state.global_step
    train_sop_loss = self._sop_total_loss_scalar / self.state.global_step

    metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss
    metrics["train_mlm_loss"] = train_mlm_loss
    metrics["train_sop_loss"] = train_sop_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    return TrainOutput(self.state.global_step, train_loss, metrics)

  def _maybe_log_save_evaluate(self, tr_loss, mlm_tr_loss, sop_tr_loss, model, trial, epoch):
    if self.control.should_log:
      logs: Dict[str, float] = {}
      tr_loss_scalar = tr_loss.item()
      mlm_tr_loss_scalar = mlm_tr_loss.item()
      sop_tr_loss_scalar = sop_tr_loss.item()

      # reset tr_loss to zero
      tr_loss -= tr_loss
      mlm_tr_loss -= mlm_tr_loss
      sop_tr_loss -= sop_tr_loss

      logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
      logs["mlm_loss"] = round(mlm_tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
      logs["sop_loss"] = round(sop_tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

      logs["learning_rate"] = self._get_learning_rate()

      self._total_loss_scalar += tr_loss_scalar
      self._mlm_total_loss_scalar += mlm_tr_loss_scalar
      self._sop_total_loss_scalar += sop_tr_loss_scalar

      self._globalstep_last_logged = self.state.global_step
      self.store_flos()

      self.log(logs)

    metrics = None
    if self.control.should_evaluate:
      metrics = self.evaluate()
      self._report_to_hp_search(trial, epoch, metrics)

    if self.control.should_save:
      self._save_checkpoint(model, trial, metrics=metrics)
      self.control = self.callback_handler.on_save(self.args, self.state, self.control)

  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    inputs = self._prepare_inputs(inputs)

    if is_sagemaker_mp_enabled():
      scaler = self.scaler if self.use_amp else None
      loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
      return loss_mb.reduce_mean().detach().to(self.args.device)

    if self.use_amp:
      with autocast():
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
    else:
      loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

    mlm_loss = outputs['mlm_loss'].detach()
    sop_loss = outputs['sop_loss'].detach()

    if self.args.n_gpu > 1:
      loss = loss.mean()  # mean() to average on multi-gpu parallel training
      mlm_loss = mlm_loss.mean()
      sop_loss = sop_loss.mean()

    if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
      # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
      loss = loss / self.args.gradient_accumulation_steps
      mlm_loss = mlm_loss / self.args.gradient_accumulation_steps
      sop_loss = sop_loss / self.args.gradient_accumulation_steps

    if self.use_amp:
      self.scaler.scale(loss).backward()
    elif self.use_apex:
      with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
    elif self.deepspeed:
      # loss gets scaled under gradient_accumulation_steps in deepspeed
      loss = self.deepspeed.backward(loss)
    else:
      loss.backward()

    return loss.detach(), mlm_loss, sop_loss
