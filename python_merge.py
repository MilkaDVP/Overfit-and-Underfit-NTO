import sys
from pathlib import Path

import pandas as pd


def merge_pairs(root: Path, prefix: str = "ind_", ext: str = ".csv") -> None:
    """Find files like ind_*.csv, merge each with its base *.csv, save *_merged.csv."""
    for ind_path in root.glob(f"{prefix}*{ext}"):
        base_stem = ind_path.stem
        if not base_stem.startswith(prefix):
            continue

        plain_name = base_stem[len(prefix):] + ind_path.suffix
        plain_path = ind_path.with_name(plain_name)
        if not plain_path.exists():
            print(f"No pair for {ind_path.name}")
            continue

        df_ind = pd.read_csv(ind_path)
        df_plain = pd.read_csv(plain_path)
        merged = pd.concat([df_ind, df_plain], ignore_index=True, sort=False)

        out_name = base_stem[len(prefix):] + "_merged" + ind_path.suffix
        out_path = ind_path.with_name(out_name)
        merged.to_csv(out_path, index=False)
        print(f"{ind_path.name} + {plain_path.name} -> {out_path.name}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    merge_pairs(root)
