import pickle
from pathlib import Path

import lmdb
import pandas as pd

from tools.graph_lmdb import build_lmdb_from_csv, build_splits, resolve_cif_path


def _write_cif(path: Path, content: str) -> None:
    path.write_text(content)


def test_resolve_cif_path(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    cif_root = tmp_path / "cifs"
    csv_dir.mkdir()
    cif_root.mkdir()
    rel_path = "sample.cif"

    resolved = resolve_cif_path(rel_path, csv_dir, cif_root)
    assert resolved == cif_root / rel_path

    resolved_no_root = resolve_cif_path(rel_path, csv_dir, None)
    assert resolved_no_root == csv_dir / rel_path


def test_split_reproducible() -> None:
    df = pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(10)],
            "cif_path": [f"{i}.cif" for i in range(10)],
            "hardness": list(range(10)),
        }
    )
    first = build_splits(df, split=(0.8, 0.1, 0.1), seed=123)
    second = build_splits(df, split=(0.8, 0.1, 0.1), seed=123)
    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "test"}


def test_build_graph_and_lmdb(tmp_path: Path) -> None:
    cif_content = """
    data_Si
    _symmetry_space_group_name_H-M   'P 1'
    _cell_length_a   3.84
    _cell_length_b   3.84
    _cell_length_c   3.84
    _cell_angle_alpha 90
    _cell_angle_beta 90
    _cell_angle_gamma 90
    _symmetry_Int_Tables_number 1
    loop_
      _atom_site_label
      _atom_site_type_symbol
      _atom_site_fract_x
      _atom_site_fract_y
      _atom_site_fract_z
      Si1 Si 0 0 0
      Si2 Si 0.25 0.25 0.25
    """
    cif_root = tmp_path / "cifs"
    cif_root.mkdir()
    cif_path = cif_root / "si.cif"
    _write_cif(cif_path, cif_content)

    csv_path = tmp_path / "all.csv"
    df = pd.DataFrame(
        {
            "id": ["si"],
            "cif_path": ["si.cif"],
            "hardness": [1.23],
            "split": ["train"],
        }
    )
    df.to_csv(csv_path, index=False)

    out_root = tmp_path / "out"
    build_lmdb_from_csv(
        csv_path=csv_path,
        out_root=out_root,
        cif_root=cif_root,
        get_edges=True,
        cutoff=6.0,
        max_neighbors=12,
        map_size_mb=64,
    )

    lmdb_path = out_root / "lmdb" / "train.lmdb"
    assert lmdb_path.exists()

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        payload = txn.get(b"0")
    env.close()

    record = pickle.loads(payload)
    assert record["id"] == "si"
    assert record["num_nodes"] == 2
    assert record["z"].shape[0] == 2
    assert "edge_index" in record
