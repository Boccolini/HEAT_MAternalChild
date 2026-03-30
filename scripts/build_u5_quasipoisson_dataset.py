from __future__ import annotations

import calendar
import gzip
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "analysis" / "ad_hoc_quasipoisson_u5_20260326"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "u5_municipality_day_model_input.csv.gz"

DEFAULT_YEARS = (2020, 2021, 2022, 2023, 2024)
KEEP_COLUMNS = [
    "MUNICIPALITY_CODE",
    "DATE",
    "YEAR",
    "DEATHS_U5",
    "POP_U5",
    "TMAX_C",
    "RH_MEAN_PCT",
    "HEAT_INDEX_APPROX_C",
    "QC_STATUS",
    "UNDER5_POPULATION_OFFSET_VALID_FLAG",
]

UF_CODE_TO_ABBR = {
    "11": "RO",
    "12": "AC",
    "13": "AM",
    "14": "RR",
    "15": "PA",
    "16": "AP",
    "17": "TO",
    "21": "MA",
    "22": "PI",
    "23": "CE",
    "24": "RN",
    "25": "PB",
    "26": "PE",
    "27": "AL",
    "28": "SE",
    "29": "BA",
    "31": "MG",
    "32": "ES",
    "33": "RJ",
    "35": "SP",
    "41": "PR",
    "42": "SC",
    "43": "RS",
    "50": "MS",
    "51": "MT",
    "52": "GO",
    "53": "DF",
}


def _year_path(year: int) -> Path:
    return (
        ROOT
        / "data"
        / "gold"
        / "gold_mnch_mortality"
        / f"YEAR={year}"
        / f"gold_mnch_mortality_{year}.parquet"
    )


def _prepare_year_frame(year: int) -> pd.DataFrame:
    path = _year_path(year)
    frame = pd.read_parquet(path, columns=KEEP_COLUMNS).copy()
    frame["MUNICIPALITY_CODE"] = frame["MUNICIPALITY_CODE"].astype(str).str.zfill(6)
    frame["UF"] = frame["MUNICIPALITY_CODE"].str[:2].map(UF_CODE_TO_ABBR)
    frame["DAYS_IN_YEAR"] = calendar.isleap(year) and 366 or 365

    frame = frame.loc[
        frame["QC_STATUS"].eq("PASS")
        & frame["UNDER5_POPULATION_OFFSET_VALID_FLAG"].fillna(0).eq(1)
        & frame["POP_U5"].notna()
        & frame["POP_U5"].gt(0)
        & frame["UF"].notna()
        & frame["DEATHS_U5"].notna()
    ].copy()

    frame["DEATHS_U5"] = pd.to_numeric(frame["DEATHS_U5"], errors="coerce").fillna(0).astype(int)
    frame["POP_U5"] = pd.to_numeric(frame["POP_U5"], errors="coerce")
    for column in ("TMAX_C", "RH_MEAN_PCT", "HEAT_INDEX_APPROX_C"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame[
        [
            "MUNICIPALITY_CODE",
            "UF",
            "DATE",
            "YEAR",
            "DAYS_IN_YEAR",
            "DEATHS_U5",
            "POP_U5",
            "TMAX_C",
            "RH_MEAN_PCT",
            "HEAT_INDEX_APPROX_C",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(DEFAULT_YEARS),
        help="Calendar years to extract from gold_mnch_mortality.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Compressed CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_dir = output_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    first_chunk = True
    with gzip.open(output_path, "wt", newline="") as handle:
        for year in args.years:
            frame = _prepare_year_frame(year)
            frame.to_csv(handle, index=False, header=first_chunk)
            first_chunk = False
            print(f"{year}: wrote {len(frame):,} rows")

    print(f"Saved analytical dataset to {output_path}")


if __name__ == "__main__":
    main()
