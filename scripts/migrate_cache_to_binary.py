from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data_cache"
    created = 0
    refreshed = 0
    skipped = 0
    failed = 0

    if not root.exists():
        print("cache-sidecar-migration created=0 refreshed=0 skipped=0 failed=0")
        return

    for csv_path in sorted(root.rglob("*.csv")):
        pkl_path = csv_path.with_suffix(".pkl")
        csv_stat = csv_path.stat()
        if pkl_path.exists() and pkl_path.stat().st_mtime_ns >= csv_stat.st_mtime_ns:
            skipped += 1
            continue
        try:
            had_pkl = pkl_path.exists()
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = (
                df.dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(subset=["date"])
                .reset_index(drop=True)
            )
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(pkl_path)
            if had_pkl:
                refreshed += 1
            else:
                created += 1
        except Exception:
            failed += 1

    print(
        "cache-sidecar-migration "
        f"created={created} "
        f"refreshed={refreshed} "
        f"skipped={skipped} "
        f"failed={failed}"
    )


if __name__ == "__main__":
    main()
