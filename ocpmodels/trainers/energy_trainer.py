"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import Subset
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer


@registry.register_trainer("energy")
class EnergyTrainer(BaseTrainer):
    """
    Trainer class for the Initial Structure to Relaxed Energy (IS2RE) task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_.


    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_vis=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        loss=None,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_vis=is_vis,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2re",
            slurm=slurm,
            loss=loss,
        )

    def load_task(self):
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self._load_heteroscedastic_config()
        self.num_targets = 2 if self.heteroscedastic_enabled else 1

    def _load_heteroscedastic_config(self):
        loss_config = self.config.get("loss", {})
        hetero_config = loss_config.get("heteroscedastic", {})
        loss_type = loss_config.get("type", "homoscedastic")
        enabled = hetero_config.get("enabled", False)
        self.heteroscedastic_enabled = (
            loss_type == "heteroscedastic" or enabled
        )
        self.heteroscedastic_output = hetero_config.get(
            "output", "variance"
        )
        if self.heteroscedastic_output not in ("variance", "std"):
            raise ValueError(
                "loss.heteroscedastic.output must be 'variance' or 'std'"
            )
        min_key = (
            "min_variance"
            if self.heteroscedastic_output == "variance"
            else "min_std"
        )
        self.heteroscedastic_min = hetero_config.get(min_key, 1e-6)
        try:
            self.heteroscedastic_min = float(self.heteroscedastic_min)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"loss.heteroscedastic.{min_key} must be a number."
            ) from exc

    @torch.no_grad()
    def predict(
        self, loader, per_image=True, results_file=None, disable_tqdm=False
    ):
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
        predictions = {"id": [], "energy": []}
        if self.heteroscedastic_enabled:
            predictions[self.heteroscedastic_output] = []

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )
                out = self._denorm_uncertainty(out)

            if per_image:
                predictions["id"].extend(
                    [str(i) for i in batch[0].sid.tolist()]
                )
                predictions["energy"].extend(out["energy"].tolist())
                if self.heteroscedastic_enabled:
                    predictions[self.heteroscedastic_output].extend(
                        out[self._uncertainty_key()].tolist()
                    )
            else:
                predictions["energy"] = out["energy"].detach()
                if self.heteroscedastic_enabled:
                    predictions[self.heteroscedastic_output] = out[
                        self._uncertainty_key()
                    ].detach()
                return predictions

        keys = ["energy"]
        if self.heteroscedastic_enabled:
            keys.append(self.heteroscedastic_output)
        self.save_results(predictions, results_file, keys=keys)

        if self.ema:
            self.ema.restore()

        return predictions

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_mae = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    print(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if (
                            val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            < self.best_val_mae
                        ):
                            self.best_val_mae = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()



    def _forward(self, batch_list):
        output = self.model(batch_list)

        if self.heteroscedastic_enabled:
            if output.shape[-1] != 2:
                raise ValueError(
                    "Heteroscedastic loss requires model output with 2 targets."
                )
            mean = output[..., 0]
            raw_uncertainty = output[..., 1]
            uncertainty = F.softplus(raw_uncertainty) + self.heteroscedastic_min
            output_dict = {"energy": mean}
            output_dict[self._uncertainty_key()] = uncertainty
            return output_dict

        if output.shape[-1] == 1:
            output = output.view(-1)

        return {"energy": output}

    def _compute_loss(self, out, batch_list):
        energy_target = torch.cat(
            [
                (batch.y_relaxed if hasattr(batch, "y_relaxed") else batch.y).to(
                    self.device
                )
                for batch in batch_list
            ],
            dim=0,
        )

        if self.normalizer.get("normalize_labels", False):
            target_normed = self.normalizers["target"].norm(energy_target)
        else:
            target_normed = energy_target

        if self.heteroscedastic_enabled:
            mean = out["energy"]
            uncertainty = out[self._uncertainty_key()]
            if self.heteroscedastic_output == "std":
                variance = uncertainty.pow(2)
            else:
                variance = uncertainty
            nll = 0.5 * (
                (target_normed - mean).pow(2) / variance
                + torch.log(variance)
            )
            return nll.mean()

        return self.loss_fn["energy"](out["energy"], target_normed)

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [
                (batch.y_relaxed if hasattr(batch, "y_relaxed") else batch.y).to(
                    self.device
                )
                for batch in batch_list
            ],
            dim=0,
        )

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

        return metrics

    def _uncertainty_key(self) -> str:
        if self.heteroscedastic_output == "std":
            return "energy_std"
        return "energy_variance"

    def _denorm_uncertainty(self, out):
        if not self.heteroscedastic_enabled:
            return out
        if self.normalizers is None or "target" not in self.normalizers:
            return out
        scale = self.normalizers["target"].std
        if self.heteroscedastic_output == "std":
            out[self._uncertainty_key()] = out[self._uncertainty_key()] * scale
        else:
            out[self._uncertainty_key()] = out[self._uncertainty_key()] * (
                scale**2
            )
        return out
