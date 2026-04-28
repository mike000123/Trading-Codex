"""Refresh the GLD central-bank proxy from IMF/DBnomics monthly gold reserves."""
from __future__ import annotations

import argparse

from data.official_gold_sources import (
    DEFAULT_DBNOMICS_IMF_GOLD_RESERVE_AREAS,
    cache_dbnomics_central_bank_proxy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a curated IMF/DBnomics monthly gold-reserve basket and cache GLD_CB_PROXY.",
    )
    parser.add_argument(
        "--areas",
        nargs="*",
        default=list(DEFAULT_DBNOMICS_IMF_GOLD_RESERVE_AREAS),
        help="Override the default IMF reference-area basket (for example: US DE IT FR RU CN CH IN JP TR NL PL KZ).",
    )
    parser.add_argument(
        "--cache-root",
        default="data_cache",
        help="Cache root directory. Defaults to ./data_cache",
    )
    args = parser.parse_args()

    daily_path, monthly_path = cache_dbnomics_central_bank_proxy(
        ref_areas=args.areas,
        cache_root=args.cache_root,
    )
    print(f"Daily proxy written to: {daily_path}")
    print(f"Monthly proxy written to: {monthly_path}")


if __name__ == "__main__":
    main()
