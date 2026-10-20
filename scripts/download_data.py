"""Download the UCI Default of Credit Card Clients dataset into data/raw/."""

from __future__ import annotations

import argparse
from pathlib import Path

from credit_risk.data import RAW_FILE, download_dataset, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RAW_FILE, help="Destination CSV path.")
    args = parser.parse_args()

    path = download_dataset(args.output)
    df = load_dataset(path)
    print(f"Saved {path} - shape={df.shape}, default rate={df['default'].mean():.3f}")


if __name__ == "__main__":
    main()
