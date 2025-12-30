"""Preprocess crystal hardness datasets into LMDB files.

Expected CSV columns:
- cif_path: path to a CIF file (relative to --cif-root if provided)
- hardness: target value (float)
"""

import argparse
import csv
import json
import os
import pickle
from typing import Dict, List, Tuple

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CIF + hardness CSV into an LMDB dataset."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument(
        "--out-path",
        help="Output directory for data.lmdb and metadata.npz",
    )
    parser.add_argument(
        "--out-root",
        help="Root directory for train/val/test outputs when using --split",
    )
    parser.add_argument(
        "--cif-root",
        default=None,
        help="Optional root directory prepended to cif paths",
    )
    parser.add_argument(
        "--cif-column",
        default="cif_path",
        help="CSV column containing CIF paths",
    )
    parser.add_argument(
        "--target-column",
        default="hardness",
        help="CSV column containing hardness values",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional CSV column to store as sample_id",
    )
    parser.add_argument(
        "--max-neigh",
        type=int,
        default=50,
        help="Maximum neighbors per atom",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Neighbor cutoff radius (Angstrom)",
    )
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB instead of OTF graph building",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip rows that fail to parse",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=64,
        help="LMDB map size in GB",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Optional split fractions for train/val/test",
    )
    parser.add_argument(
        "--split-names",
        nargs=3,
        default=["train", "val", "test"],
        help="Split names when using --split",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed for --split",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.split and not args.out_root:
        raise SystemExit("--out-root is required when using --split")
    if not args.split and not args.out_path:
        raise SystemExit("--out-path is required unless --split is provided")
    if args.split:
        split_sum = sum(args.split)
        if split_sum <= 0:
            raise SystemExit("--split values must sum to a positive number")
        if len(args.split_names) != 3:
            raise SystemExit("--split-names must provide three values")


def build_cif_path(cif_root: str, cif_path: str) -> str:
    if cif_root:
        return os.path.join(cif_root, cif_path)
    return cif_path


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_rows_with_header(
    csv_path: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def split_rows(
    rows: List[Dict[str, str]],
    split: List[float],
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)

    split = np.array(split, dtype=np.float64)
    split = split / split.sum()
    n_total = len(rows)
    n_train = int(split[0] * n_total)
    n_val = int(split[1] * n_total)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]
    return train_rows, val_rows, test_rows


def write_csv(
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_row(
    a2g: AtomsToGraphs,
    row: Dict[str, str],
    idx: int,
    cif_root: str,
    cif_column: str,
    target_column: str,
    id_column: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cif_path = build_cif_path(cif_root, row[cif_column])
    atoms = ase.io.read(cif_path)
    data_object = a2g.convert(atoms)
    data_object.y = torch.tensor(
        [float(row[target_column])], dtype=torch.float
    )
    data_object.sid = idx
    if id_column:
        data_object.sample_id = str(row[id_column])
    data_object.cif_path = os.path.abspath(cif_path)

    natoms = torch.tensor([data_object.natoms])
    if hasattr(data_object, "edge_index"):
        neighbors = torch.tensor([data_object.edge_index.shape[1]])
    else:
        neighbors = torch.tensor([0])

    return data_object, natoms, neighbors


def write_lmdb(
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    out_path: str,
) -> None:
    os.makedirs(out_path, exist_ok=True)

    a2g = AtomsToGraphs(
        max_neigh=args.max_neigh,
        radius=args.radius,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=args.get_edges,
        r_fixed=True,
    )

    db_path = os.path.join(out_path, "data.lmdb")
    db = lmdb.open(
        db_path,
        map_size=args.map_size_gb * 1024**3,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    natoms_list = []
    neighbors_list = []
    targets = []

    idx = 0
    with db.begin(write=True) as txn:
        for row in tqdm(rows, desc="Converting CIFs to LMDB"):
            try:
                data_object, natoms, neighbors = convert_row(
                    a2g,
                    row,
                    idx,
                    args.cif_root,
                    args.cif_column,
                    args.target_column,
                    args.id_column,
                )
            except Exception as exc:
                if args.skip_failed:
                    print(
                        f"Skipping row {idx} due to error: {exc}",
                        flush=True,
                    )
                    continue
                raise

            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            natoms_list.append(natoms)
            neighbors_list.append(neighbors)
            targets.append(float(row[args.target_column]))
            idx += 1

        txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))

    db.sync()
    db.close()

    natoms_arr = torch.cat(natoms_list).numpy() if natoms_list else np.array([])
    neighbors_arr = (
        torch.cat(neighbors_list).numpy() if neighbors_list else np.array([])
    )
    np.savez(
        os.path.join(out_path, "metadata.npz"),
        natoms=natoms_arr,
        neighbors=neighbors_arr,
    )

    if targets:
        target_array = np.array(targets, dtype=np.float32)
        stats = {
            "target_mean": float(target_array.mean()),
            "target_std": float(target_array.std()),
            "num_samples": int(target_array.shape[0]),
        }
    else:
        stats = {"target_mean": 0.0, "target_std": 1.0, "num_samples": 0}

    stats_path = os.path.join(out_path, "target_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(
        "Finished writing LMDB with {count} samples. Stats saved to {stats}.".format(
            count=idx, stats=stats_path
        )
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.split:
        rows, fieldnames = load_rows_with_header(args.csv)
        train_rows, val_rows, test_rows = split_rows(
            rows, args.split, args.split_seed
        )
        split_rows_map = {
            args.split_names[0]: train_rows,
            args.split_names[1]: val_rows,
            args.split_names[2]: test_rows,
        }
        for split_name, split_rows_list in split_rows_map.items():
            split_csv_path = os.path.join(args.out_root, f"{split_name}.csv")
            write_csv(split_rows_list, fieldnames, split_csv_path)
            split_out_path = os.path.join(args.out_root, split_name)
            write_lmdb(args, split_rows_list, split_out_path)
        return

    rows = load_rows(args.csv)
    write_lmdb(args, rows, args.out_path)


if __name__ == "__main__":
    main()
