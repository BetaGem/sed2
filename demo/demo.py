# This is a demo script for running SED2
import sys
from pathlib import Path

# make the project root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sed2

if __name__ == "__main__":
    par_file = sys.argv[1] if len(sys.argv) > 1 else None
    if par_file is None:
        par_file = Path(__file__).parent / "demo.par"

    sed2.run(par_file)
