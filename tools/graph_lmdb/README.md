# Graph LMDB Builder

This module builds lightweight graph datasets from CIF files and writes them to
LMDB. It is independent of any training or modeling code and exposes both a
Python API and a CLI wrapper.

## Dependencies

Install the following packages in your environment:

- `pymatgen`
- `numpy`
- `pandas`
- `lmdb`
- `tqdm`

## CLI Usage

```bash
python tools/graph_lmdb/graph_lmdb.py \
  --csv data/hardness/all.csv \
  --out-root data/hardness \
  --cif-root data/hardness/cifs \
  --get-edges \
  --split 0.8 0.1 0.1
```

### Path Resolution Rules

- If `cif_path` is absolute, it is used as-is.
- If `cif_path` is relative and `--cif-root` is provided, paths resolve as
  `cif_root / cif_path`.
- If `cif_path` is relative and `--cif-root` is not provided, paths resolve as
  `csv_dir / cif_path`.

## Output Structure

```
<out-root>/
  train.csv
  val.csv
  test.csv
  errors.csv (optional)
  stats.json
  lmdb/
    train.lmdb/
    val.lmdb/
    test.lmdb/
    train_id_map.json
    val_id_map.json
    test_id_map.json
```

## LMDB Record Schema

Each LMDB record is stored as a pickled Python dictionary:

```python
{
  "id": str,
  "num_nodes": int,
  "z": np.int64[num_nodes],
  "pos": np.float32[num_nodes, 3],
  "edge_index": np.int64[2, num_edges],   # optional
  "edge_dist": np.float32[num_edges],     # optional
  "y": float,
  "cif_path": str,
  "meta": {
      "formula": str,
      "spacegroup": str,
      "lattice": (a, b, c),
  },
}
```

Edges are directed as returned by the neighbor list (i.e., `edge_index[0] ->
edge_index[1]`). When `--get-edges` is used, neighbors are selected by a cutoff
radius and then limited to `--max-neighbors` per atom.

## Common Issues

- **CIF parsing failures**: check `errors.csv` for the failing path. Ensure CIFs
  are valid and readable by `pymatgen`.
- **Neighbor generation is slow**: reduce `--cutoff` or `--max-neighbors`.
- **LMDB map size too small**: increase `--map-size-mb`.

## Python API

```python
from tools.graph_lmdb import build_lmdb_from_csv

build_lmdb_from_csv(
    csv_path="data/hardness/all.csv",
    out_root="data/hardness",
    cif_root="data/hardness/cifs",
    split=(0.8, 0.1, 0.1),
    get_edges=True,
    cutoff=8.0,
    max_neighbors=12,
    seed=42,
)
```
