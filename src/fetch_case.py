"""Download and unpack the CASE dataset (Sharma et al., Sci Data 2019).

CASE is the external replication corpus: 30 participants, 8 video stimuli,
continuous joystick valence/arousal annotation plus peripheral physiology.
Structurally the same fixed-stimulus design as CEAP-360VR but non-immersive,
which is what makes it a test of whether our diagnosis is VR-specific.

figshare article 8869157, DOI 10.6084/m9.figshare.8869157
"""

from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "case"
URL = "https://ndownloader.figshare.com/files/16260497"
ZIP = DATA / "CASE_full.zip"


def _report(done: int, total: int, t0: float) -> None:
    mb = done / 1e6
    pct = 100.0 * done / total if total else 0.0
    rate = mb / max(time.time() - t0, 1e-6)
    sys.stdout.write(f"\rCASE {mb:8.1f} MB / {total/1e6:.0f} MB  {pct:5.1f}%  {rate:5.1f} MB/s")
    sys.stdout.flush()


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and ZIP.stat().st_size > 4.5e9:
        print(f"already downloaded: {ZIP} ({ZIP.stat().st_size/1e6:.0f} MB)")
        return
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(ZIP, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done % (50 << 20) < (1 << 20):
                    _report(done, total, t0)
    print(f"\nDONE download {ZIP.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s")


def unpack() -> None:
    """Extract only the interpolated per-participant data and metadata.

    The archive also carries raw LabVIEW dumps and the authors' own
    preprocessing code, which we do not need and which dominate its size.
    """
    want = ("interpolated", "metadata")
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if any(w in n.lower() for w in want)]
        print(f"extracting {len(names)} of {len(z.namelist())} entries")
        for n in names:
            z.extract(n, DATA)
    print("DONE unpack")


if __name__ == "__main__":
    download()
    unpack()
    print("CASE READY")
