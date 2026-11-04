"""Convert data/MoNuSeg_37 (img/labelcol layout) into tiled npy samples.

MoNuSeg_37 ships as 1000x1000 RGB ROIs with binary labelcol masks under
train/validate/test. This script:

1. Stages that layout under data/MoNuSeg_37/raw/
2. Rasterizes instance maps from official MoNuSeg XML annotations when found
   under data/MoNuSeg/raw (same TCGA stem). Falls back to connected components
   on the binary labelcol mask when no XML is available.
3. Writes 256x256 tiles with image/mask/inst/h_map/v_map.npy under
   data/MoNuSeg_37/{train,val,test}/, matching SegmentationDataset.

Run from src/:  python prepare_monuseg37.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_monuseg import (
    RAW as MONUSEG_RAW,
    _polygons,
    _rasterize,
    _write_tiles,
)

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required to convert MoNuSeg_37") from exc

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "MoNuSeg_37"
RAW37 = SRC / "raw"
# Dataset loader expects "val"; the published folder name is "validate".
SPLIT_MAP = {"train": "train", "validate": "val", "test": "test"}
ROI_SUFFIX = re.compile(r"_\d+_\d+_roi$", re.IGNORECASE)


def _patient_stem(path: Path) -> str:
    return ROI_SUFFIX.sub("", path.stem)


def _stage_raw_layout():
    """Move img/labelcol splits under raw/ once so tiles can own train/val/test."""
    needs_stage = any((SRC / name / "img").is_dir() for name in SPLIT_MAP)
    if not needs_stage:
        return
    RAW37.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_MAP:
        src = SRC / name
        if not (src / "img").is_dir():
            continue
        dest = RAW37 / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        print(f"staged {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


def _xml_index(raw_root: Path) -> dict[str, Path]:
    if not raw_root.exists():
        return {}
    index = {}
    for path in raw_root.rglob("*.xml"):
        if path.stem.startswith("._") or "__MACOSX" in str(path):
            continue
        index[path.stem] = path
    return index


def _instances_from_binary(mask: np.ndarray) -> np.ndarray:
    """Connected components when XML contours are unavailable."""
    _num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def _image_label_pairs(split_dir: Path):
    img_dir = split_dir / "img"
    label_dir = split_dir / "labelcol"
    if not img_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"expected img/ and labelcol/ under {split_dir}")
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        if img_path.name.startswith("._"):
            continue
        label_path = label_dir / f"{img_path.stem}.png"
        if not label_path.exists():
            candidates = list(label_dir.glob(f"{img_path.stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"no labelcol mask for {img_path.name}")
            label_path = candidates[0]
        pairs.append((img_path, label_path))
    return pairs


def convert():
    _stage_raw_layout()
    if not RAW37.exists():
        raise SystemExit(
            f"MoNuSeg_37 raw layout not found at {RAW37}. "
            "Place train/validate/test with img/ and labelcol/ under data/MoNuSeg_37."
        )

    xmls = _xml_index(MONUSEG_RAW)
    if not xmls:
        print(f"warning: no XMLs under {MONUSEG_RAW}; using connected components on labelcol")

    totals = {"train": 0, "val": 0, "test": 0}
    images_used = {"train": 0, "val": 0, "test": 0}

    for src_name, out_name in SPLIT_MAP.items():
        split_src = RAW37 / src_name
        if not split_src.exists():
            print(f"skipping missing split {src_name}")
            continue
        split_out = SRC / out_name
        if split_out.exists():
            shutil.rmtree(split_out)
        split_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in _image_label_pairs(split_src):
            patient = _patient_stem(img_path)
            image = np.asarray(Image.open(img_path).convert("RGB"))
            binary = (np.asarray(Image.open(label_path).convert("L")) > 0).astype(np.uint8)
            if binary.shape != image.shape[:2]:
                raise ValueError(
                    f"mask shape {binary.shape} != image shape {image.shape[:2]} for {label_path}"
                )

            xml_path = xmls.get(patient)
            if xml_path is not None:
                inst = _rasterize(_polygons(xml_path), image.shape[:2])
                source = "xml"
            else:
                inst = _instances_from_binary(binary)
                source = "cc"

            n = _write_tiles(patient, image, inst, split_out)
            totals[out_name] += n
            images_used[out_name] += 1
            print(
                f"{out_name}: {patient} [{source}] -> {n} tiles, "
                f"{int((inst > 0).sum())} inst px, {int(binary.sum())} label px"
            )

    print("tiles", totals, "images", images_used)
    print(f"Wrote {SRC}/{{train,val,test}}")


def main():
    convert()


if __name__ == "__main__":
    main()
