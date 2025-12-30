"""Generate CSV predictions for train/val/test splits."""

import argparse
import logging
import os

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_config, setup_imports, setup_logging
from ocpmodels.common.hardness_utils import write_predictions_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hardness prediction CSVs for each split."
    )
    parser.add_argument(
        "--config-yml", required=True, help="Path to hardness config YAML"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--out-dir",
        default="hardness_predictions",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to export (train val test)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    setup_imports()

    config, _, _ = load_config(args.config_yml)
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier="hardness-predict",
        run_dir=os.getcwd(),
        print_every=10,
        seed=0,
        logger=config.get("logger", "tensorboard"),
        local_rank=0,
        amp=False,
        cpu=args.cpu,
        slurm={},
    )

    trainer.load_checkpoint(args.checkpoint)

    split_map = {
        "train": trainer.train_loader,
        "val": trainer.val_loader,
        "test": trainer.test_loader,
    }

    for split in args.splits:
        loader = split_map.get(split)
        if loader is None:
            logging.warning("Split '%s' is not available in config", split)
            continue
        out_path = os.path.join(args.out_dir, f"{split}.csv")
        write_predictions_csv(trainer, loader, out_path)
        logging.info("Wrote %s predictions to %s", split, out_path)


if __name__ == "__main__":
    main()
