from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heat_mnch.future_rounds.lrh_americas_round2_validation import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                *sys.argv[1:],
            ]
        )
    )
