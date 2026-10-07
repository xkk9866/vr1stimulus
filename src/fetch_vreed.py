"""Download the VREED corpus from Kaggle (Tabbaa et al., IMWUT 2021).

VREED is the immersive external replication: 26 analysed participants,
twelve 360-degree environments, clip-level eye / ECG / GSR features.

Requires a Kaggle API token in one of:
  - environment variable KAGGLE_API_TOKEN
  - file ~/.kaggle/access_token  (Windows: %USERPROFILE%\\.kaggle\\access_token)

Do not commit the token. Create it at https://www.kaggle.com/settings
under API -> Create new token (KGAT_...).
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "vreed"
SLUG = "lumaatabbaa/vr-eyes-emotions-dataset-vreed"


def _token_present() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    p = Path.home() / ".kaggle" / "access_token"
    return p.is_file() and p.stat().st_size > 0


def download() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    marker = DEST / "04 Eye Tracking Data"
    if marker.exists():
        print(f"already present: {DEST}")
        return
    if not _token_present():
        raise SystemExit(
            "Kaggle token missing. Set KAGGLE_API_TOKEN or write the token "
            "to ~/.kaggle/access_token (chmod 600 / icacls restricted)."
        )
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"downloading {SLUG} -> {DEST}")
    api.dataset_download_files(SLUG, path=str(DEST), unzip=False, quiet=False)
    zips = list(DEST.glob("*.zip"))
    if not zips:
        raise SystemExit(f"Kaggle download produced no zip in {DEST}")
    zpath = zips[0]
    print(f"unpacking {zpath.name}")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(DEST)
    zpath.unlink(missing_ok=True)
    print("DONE VREED")


if __name__ == "__main__":
    download()
