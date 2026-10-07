"""Rebuild protocol windows and rerun Table III when 32 subjects are present."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDIR = ROOT / "data" / "ceap" / "5_PhysioData" / "Frame"


def n_physio() -> int:
    return sum(1 for p in PDIR.glob("P*_Physio_FrameData.json") if p.stat().st_size > 1_000_000)


def main() -> int:
    while True:
        n = n_physio()
        print(f"physio={n}/32", flush=True)
        if n >= 32:
            break
        time.sleep(40)
    py = sys.executable
    subprocess.check_call([py, "-u", str(ROOT / "src" / "build_protocol.py")], cwd=str(ROOT))
    subprocess.check_call([py, "-u", str(ROOT / "src" / "run_protocol.py")], cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
