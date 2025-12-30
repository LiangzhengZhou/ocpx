"""Utilities for exporting hardness predictions."""

from __future__ import annotations

import csv
import os
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def gather_ids(batch_data) -> List[str]:
    if hasattr(batch_data, "sample_id"):
        sample_id = batch_data.sample_id
        if isinstance(sample_id, str):
            return [sample_id]
        if isinstance(sample_id, list):
            return [str(item) for item in sample_id]
        if isinstance(sample_id, np.ndarray):
            return [str(item) for item in sample_id.tolist()]
        if torch.is_tensor(sample_id):
            return [str(item) for item in sample_id.tolist()]
        return [str(item) for item in sample_id]
    if hasattr(batch_data, "sid"):
        return [str(item) for item in batch_data.sid.tolist()]
    return [str(idx) for idx in range(batch_data.num_nodes)]


def gather_targets(batch_data) -> torch.Tensor:
    if hasattr(batch_data, "y_relaxed"):
        return batch_data.y_relaxed
    if hasattr(batch_data, "y"):
        return batch_data.y
    raise AttributeError("Batch does not contain target values.")


def write_predictions_csv(
    trainer,
    loader,
    out_path: str,
    disable_tqdm: bool = False,
) -> None:
    trainer.model.eval()
    predictions = []

    if trainer.ema:
        trainer.ema.store()
        trainer.ema.copy_to()

    if trainer.normalizers is not None and "target" in trainer.normalizers:
        trainer.normalizers["target"].to(trainer.device)

    for batch in tqdm(loader, desc=f"Predicting {out_path}", disable=disable_tqdm):
        batch_data = batch[0]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                out = trainer._forward(batch)

        energy = out["energy"]
        uncertainty = None
        uncertainty_field = None
        if "energy_variance" in out:
            uncertainty = out["energy_variance"]
            uncertainty_field = "variance"
        elif "energy_std" in out:
            uncertainty = out["energy_std"]
            uncertainty_field = "std"
        elif getattr(trainer, "heteroscedastic_enabled", False):
            output_mode = getattr(trainer, "heteroscedastic_output", "variance")
            uncertainty_field = "std" if output_mode == "std" else "variance"
            if energy.ndim > 1 and energy.shape[-1] == 2:
                mean = energy[..., 0]
                raw_uncertainty = energy[..., 1]
                min_uncertainty = getattr(trainer, "heteroscedastic_min", 1e-6)
                uncertainty = F.softplus(raw_uncertainty) + min_uncertainty
                energy = mean

        if trainer.normalizers is not None and "target" in trainer.normalizers:
            energy = trainer.normalizers["target"].denorm(energy)
            if uncertainty is not None:
                scale = trainer.normalizers["target"].std
                if uncertainty_field == "variance":
                    uncertainty = uncertainty * scale**2
                else:
                    uncertainty = uncertainty * scale

        targets = gather_targets(batch_data).to(energy.device)
        ids = gather_ids(batch_data)

        entries = zip(
            ids,
            targets.detach().cpu().tolist(),
            energy.detach().cpu().tolist(),
        )
        if uncertainty is not None:
            entries = zip(
                ids,
                targets.detach().cpu().tolist(),
                energy.detach().cpu().tolist(),
                uncertainty.detach().cpu().tolist(),
            )

        for row in entries:
            sample_id, target, pred = row[:3]
            row_dict = {
                "sample_id": sample_id,
                "target": float(target),
                "prediction": float(pred),
            }
            if uncertainty is not None:
                row_dict[uncertainty_field] = float(row[3])
            predictions.append(row_dict)

    if trainer.ema:
        trainer.ema.restore()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["sample_id", "target", "prediction"]
    if predictions and uncertainty_field is not None:
        fieldnames.append(uncertainty_field)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
