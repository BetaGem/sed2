# This is a demo script for running SED2
import argparse
import sys
from pathlib import Path

# make the project root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sed2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SED2 pipeline.")
    parser.add_argument("par_file", nargs="?", default=None,
                        help="parameter file (default: demo/demo.par)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="overwrite existing outputs without asking")
    args = parser.parse_args()

    par_file = args.par_file
    if par_file is None:
        par_file = Path(__file__).parent / "demo.par"

    sed2.run(par_file, assume_yes=args.yes)
