"""Copy the sample images bundled with Ultralytics into data/samples/.

For video, drop any clip into data/videos/ (git-ignored) — a 20-60 s street or
hallway clip from your phone, or a free stock clip (e.g. Pexels "traffic").
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics.utils import ASSETS

dst = Path(__file__).resolve().parents[1] / "data" / "samples"
dst.mkdir(parents=True, exist_ok=True)
(dst.parent / "videos").mkdir(exist_ok=True)

for img in sorted(Path(ASSETS).glob("*.jpg")):
    shutil.copy2(img, dst / img.name)
    print(f"copied {img.name} -> {dst / img.name}")
