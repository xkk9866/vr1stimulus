"""Download CEAP-360VR analysis files from GitHub (no Unity project)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE = "https://raw.githubusercontent.com/cwi-dis/CEAP-360VR-Dataset/master/CEAP-360VR"
ROOT = Path(__file__).resolve().parents[1] / "data" / "ceap"


def urls() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    items.append((f"{BASE}/1_Stimuli/VideoInfo.json", ROOT / "1_Stimuli" / "VideoInfo.json"))
    for pid in range(1, 33):
        items.append(
            (
                f"{BASE}/2_QuestionnaireData/P{pid}_Questionnaire_Data.json",
                ROOT / "2_QuestionnaireData" / f"P{pid}_Questionnaire_Data.json",
            )
        )
        items.append(
            (
                f"{BASE}/3_AnnotationData/Frame/P{pid}_Annotation_FrameData.json",
                ROOT / "3_AnnotationData" / "Frame" / f"P{pid}_Annotation_FrameData.json",
            )
        )
        items.append(
            (
                f"{BASE}/4_BehaviorData/Frame/P{pid}_Behavior_FrameData.json",
                ROOT / "4_BehaviorData" / "Frame" / f"P{pid}_Behavior_FrameData.json",
            )
        )
        items.append(
            (
                f"{BASE}/5_PhysioData/Frame/P{pid}_Physio_FrameData.json",
                ROOT / "5_PhysioData" / "Frame" / f"P{pid}_Physio_FrameData.json",
            )
        )
    return items


def fetch(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 2000:
        return f"skip {dest.name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl.exe",
        "-L",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "-A",
        "Mozilla/5.0",
        "-o",
        str(tmp),
        url,
    ]
    for attempt in range(5):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if tmp.exists() and tmp.stat().st_size > 2000:
                tmp.replace(dest)
                return f"ok {dest.name} {dest.stat().st_size}"
        except Exception as exc:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            last = exc
        else:
            last = RuntimeError("empty file")
            time.sleep(2)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return f"FAIL {dest.name}: {last}"


def main() -> int:
    jobs = urls()
    print(f"Downloading {len(jobs)} files into {ROOT}", flush=True)
    ok = fail = 0
    for url, dest in jobs:
        msg = fetch(url, dest)
        print(msg, flush=True)
        if msg.startswith("FAIL"):
            fail += 1
        else:
            ok += 1
        time.sleep(0.4)
    print(f"done ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
