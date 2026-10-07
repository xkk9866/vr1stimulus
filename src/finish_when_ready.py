"""Wait until 32 CEAP subjects are on disk, then rebuild windows and rerun CV."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "data" / "ceap" / "2_QuestionnaireData"
PDIR = ROOT / "data" / "ceap" / "5_PhysioData" / "Frame"


def n_ready() -> int:
    qs = set()
    for p in QDIR.glob("P*_Questionnaire_Data.json"):
        if p.stat().st_size > 1000:
            qs.add(p.stem.split("_")[0])
    ps = set()
    for p in PDIR.glob("P*_Physio_FrameData.json"):
        if p.stat().st_size > 10000:
            ps.add(p.stem.split("_")[0])
    return len(qs & ps)


def main() -> int:
    while True:
        n = n_ready()
        print(f"complete subjects={n}/32", flush=True)
        if n >= 32:
            break
        time.sleep(45)
    py = sys.executable
    subprocess.check_call([py, str(ROOT / "src" / "ceap_prepare.py")], cwd=str(ROOT))
    subprocess.check_call([py, str(ROOT / "src" / "run_experiments.py")], cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
