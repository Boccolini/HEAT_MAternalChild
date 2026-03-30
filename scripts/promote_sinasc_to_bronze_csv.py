from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heat_mnch.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            ["promote-bronze", "--dataset", "sinasc", *sys.argv[1:]],
            invocation=" ".join(
                [
                    shlex.quote(sys.executable),
                    shlex.quote(str(Path(__file__).resolve())),
                    *(shlex.quote(arg) for arg in sys.argv[1:]),
                ]
            ),
        )
    )
