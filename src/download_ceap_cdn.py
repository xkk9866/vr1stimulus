"""Faster CEAP download via jsDelivr, skipping files already on disk."""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MIRRORS = [
    "https://cdn.jsdelivr.net/gh/cwi-dis/CEAP-360VR-Dataset@master/CEAP-360VR",
    "https://raw.githubusercontent.com/cwi-dis/CEAP-360VR-Dataset/master/CEAP-360VR",
]
ROOT = Path(__file__).resolve().parents[1] / "data" / "ceap"


def jobs() -> list[tuple[str, Path]]:
    items = [("1_Stimuli/VideoInfo.json", ROOT / "1_Stimuli" / "VideoInfo.json")]
    for pid in range(1, 33):
        items.append((f"2_QuestionnaireData/P{pid}_Questionnaire_Data.json", ROOT / "2_QuestionnaireData" / f"P{pid}_Questionnaire_Data.json"))
        items.append((f"3_AnnotationData/Frame/P{pid}_Annotation_FrameData.json", ROOT / "3_AnnotationData" / "Frame" / f"P{pid}_Annotation_FrameData.json"))
        items.append((f"4_BehaviorData/Frame/P{pid}_Behavior_FrameData.json", ROOT / "4_BehaviorData" / "Frame" / f"P{pid}_Behavior_FrameData.json"))
        items.append((f"5_PhysioData/Frame/P{pid}_Physio_FrameData.json", ROOT / "5_PhysioData" / "Frame" / f"P{pid}_Physio_FrameData.json"))
    return items


def curl(url: str, dest: Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl.exe", "-L", "--fail", "--retry", "4", "--retry-all-errors",
        "--retry-delay", "1", "--connect-timeout", "20", "-A", "Mozilla/5.0",
        "-o", str(tmp), url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        if tmp.exists() and tmp.stat().st_size > 2000:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(dest)
            return True
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return False


def fetch_one(rel: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 2000:
        return f"skip {dest.name}"
    for base in MIRRORS:
        if curl(f"{base}/{rel}", dest):
            return f"ok {dest.name} {dest.stat().st_size}"
        time.sleep(0.3)
    return f"FAIL {dest.name}"


def main() -> int:
    todo = jobs()
    print(f"queue={len(todo)}", flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(fetch_one, rel, dest) for rel, dest in todo]
        for fut in as_completed(futs):
            msg = fut.result()
            print(msg, flush=True)
            if msg.startswith("FAIL"):
                fail += 1
            else:
                ok += 1
    print(f"done ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
