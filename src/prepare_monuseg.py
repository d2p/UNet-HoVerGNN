"""Download official MoNuSeg archives and write 256x256 tiles for this repo.

Source: https://monuseg.grand-challenge.org/Data/ (CC BY-NC-SA 4.0). The Google
Drive ids below are the training and testing buttons on that page.

The 30-image Kumar archive is split the way HoVer-Net splits Kumar: 16 images
from breast, liver, kidney and prostate (4 each) versus 14 test images (2 of
each of those organs, plus bladder, colon and stomach). Four of the 16, one
per training organ, are held out as validation. The separate 14-image MoNuSeg
2018 test set is written to data/MoNuSeg_Test2018 and is not mixed into the
Kumar test split.

Within each organ the ids follow the challenge organ list. The first four are
the HoVer-Net training pool and the last two are the Kumar test images.
"""

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required to rasterize MoNuSeg contours") from exc

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "MoNuSeg" / "raw"
OUT = ROOT / "data" / "MoNuSeg"
TEST2018 = ROOT / "data" / "MoNuSeg_Test2018"
TILE = 256
# Keep a tile that still contains a nucleus. 0.5% of a 256 tile is larger than some nuclei.
MIN_NUCLEAR_PIXELS = 64

# Official grand-challenge download buttons (Google Drive).
TRAIN_FILE_IDS = [
    "1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
    "1JZN9Jq9km0rZNiYNEukE_8f0CsSK3Pe4",
]
TEST_FILE_IDS = [
    "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw",
]

BREAST = [
    "TCGA-A7-A13E-01Z-00-DX1",
    "TCGA-A7-A13F-01Z-00-DX1",
    "TCGA-AR-A1AK-01Z-00-DX1",
    "TCGA-AR-A1AS-01Z-00-DX1",
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
]
KIDNEY = [
    "TCGA-B0-5711-01Z-00-DX1",
    "TCGA-HE-7128-01Z-00-DX1",
    "TCGA-HE-7129-01Z-00-DX1",
    "TCGA-HE-7130-01Z-00-DX1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-B0-5698-01Z-00-DX1",
]
LIVER = [
    "TCGA-18-5592-01Z-00-DX1",
    "TCGA-38-6178-01Z-00-DX1",
    "TCGA-49-4488-01Z-00-DX1",
    "TCGA-50-5931-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-21-5786-01Z-00-DX1",
]
PROSTATE = [
    "TCGA-G9-6336-01Z-00-DX1",
    "TCGA-G9-6348-01Z-00-DX1",
    "TCGA-G9-6356-01Z-00-DX1",
    "TCGA-G9-6363-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",
]
BLADDER = ["TCGA-DK-A2I6-01A-01-TS1", "TCGA-G2-A2EK-01A-02-TSB"]
COLON = ["TCGA-AY-A8YK-01A-01-TS1", "TCGA-NH-A8F7-01A-01-TS1"]
STOMACH = ["TCGA-KB-A93J-01A-01-TS1", "TCGA-RD-A8N9-01A-01-TS1"]

# HoVer-Net training pool is the first four of each seen organ. The fourth is val.
TRAIN_IDS = set()
VAL_IDS = set()
for organ in (BREAST, KIDNEY, LIVER, PROSTATE):
    TRAIN_IDS.update(organ[:3])
    VAL_IDS.add(organ[3])
TEST_IDS = set()
for organ in (BREAST, KIDNEY, LIVER, PROSTATE):
    TEST_IDS.update(organ[4:])
TEST_IDS.update(BLADDER)
TEST_IDS.update(COLON)
TEST_IDS.update(STOMACH)


def _download(file_id, destination: Path):
    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"already downloaded {destination}")
        return
    if destination.exists():
        destination.unlink()
    gdown.download(id=file_id, output=str(destination), quiet=False)
    if not destination.exists() or destination.stat().st_size < 1_000_000:
        raise RuntimeError(f"download failed for {file_id}")


def _extract(archive: Path, folder: Path):
    marker = folder / ".extracted"
    if marker.exists():
        return
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(folder)
    marker.write_text("ok", encoding="utf-8")


def _pairs(root: Path):
    xmls = {p.stem: p for p in root.rglob("*.xml")}
    tifs = {}
    for pattern in ("*.tif", "*.tiff", "*.png"):
        for path in root.rglob(pattern):
            tifs.setdefault(path.stem, path)
    stems = sorted(
        stem for stem in (set(xmls) & set(tifs))
        if not stem.startswith("._") and "__MACOSX" not in str(xmls[stem])
    )
    return [(stem, tifs[stem], xmls[stem]) for stem in stems]


def _polygons(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    polygons = []
    for region in root.findall(".//Region"):
        pts = []
        for vertex in region.findall(".//Vertex"):
            if "X" not in vertex.attrib or "Y" not in vertex.attrib:
                continue
            pts.append((float(vertex.attrib["X"]), float(vertex.attrib["Y"])))
        if len(pts) >= 3:
            polygons.append(np.asarray(pts, dtype=np.float32))
    return polygons


def _rasterize(polygons, shape):
    inst = np.zeros(shape, np.int32)
    # Larger nuclei first so a later, smaller contour stays on top where they overlap.
    order = sorted(polygons, key=lambda p: abs(_polygon_area(p)), reverse=True)
    for idx, poly in enumerate(order, start=1):
        cnt = np.round(poly).astype(np.int32)
        cv2.fillPoly(inst, [cnt], int(idx))
    return inst


def _polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _hv_maps(inst):
    """Signed distance to each nucleus centroid, scaled to [-1, 1] per side."""
    h_map = np.zeros(inst.shape, np.float32)
    v_map = np.zeros(inst.shape, np.float32)
    for inst_id in np.unique(inst):
        if inst_id == 0:
            continue
        ys, xs = np.where(inst == inst_id)
        cy = ys.mean()
        cx = xs.mean()
        dy = ys.astype(np.float32) - cy
        dx = xs.astype(np.float32) - cx
        pos_y = dy > 0
        neg_y = dy < 0
        pos_x = dx > 0
        neg_x = dx < 0
        if pos_y.any():
            dy[pos_y] /= dy[pos_y].max() + 1e-8
        if neg_y.any():
            dy[neg_y] /= -dy[neg_y].min() + 1e-8
        if pos_x.any():
            dx[pos_x] /= dx[pos_x].max() + 1e-8
        if neg_x.any():
            dx[neg_x] /= -dx[neg_x].min() + 1e-8
        v_map[ys, xs] = dy
        h_map[ys, xs] = dx
    return h_map, v_map


def _quantize(values):
    return np.clip(np.rint((values + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _tile_starts(length, tile):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _write_tiles(stem, image, inst, split_root: Path):
    h_map, v_map = _hv_maps(inst)
    binary = (inst > 0).astype(np.uint8)
    height, width = inst.shape
    written = 0
    for y0 in _tile_starts(height, TILE):
        for x0 in _tile_starts(width, TILE):
            inst_crop = inst[y0 : y0 + TILE, x0 : x0 + TILE]
            if int((inst_crop > 0).sum()) < MIN_NUCLEAR_PIXELS:
                continue
            # Contiguous ids inside the tile. Hover maps stay in the full-image frame.
            inst_ids = inst_crop.copy()
            kept = [i for i in np.unique(inst_ids) if i != 0]
            remapped = np.zeros_like(inst_ids)
            for new_id, old_id in enumerate(kept, start=1):
                remapped[inst_ids == old_id] = new_id
            sample = split_root / f"{stem}_y{y0}_x{x0}"
            sample.mkdir(parents=True, exist_ok=True)
            np.save(sample / "image.npy", image[y0 : y0 + TILE, x0 : x0 + TILE])
            np.save(sample / "mask.npy", binary[y0 : y0 + TILE, x0 : x0 + TILE])
            np.save(sample / "inst.npy", remapped.astype(np.int32))
            np.save(sample / "h_map.npy", _quantize(h_map[y0 : y0 + TILE, x0 : x0 + TILE]))
            np.save(sample / "v_map.npy", _quantize(v_map[y0 : y0 + TILE, x0 : x0 + TILE]))
            written += 1
    return written


def _convert_kumar(raw_root: Path):
    counts = {"train": 0, "val": 0, "test": 0}
    images_used = {name: 0 for name in ("train", "val", "test")}
    seen = set()
    for stem, tif_path, xml_path in _pairs(raw_root):
        if stem in TRAIN_IDS:
            split = "train"
        elif stem in VAL_IDS:
            split = "val"
        elif stem in TEST_IDS:
            split = "test"
        else:
            print(f"skipping unlisted training image {stem}")
            continue
        image = np.asarray(Image.open(tif_path).convert("RGB"))
        inst = _rasterize(_polygons(xml_path), image.shape[:2])
        n = _write_tiles(stem, image, inst, OUT / split)
        counts[split] += n
        images_used[split] += 1
        seen.add(stem)
        print(f"{split}: {stem} -> {n} tiles, {int((inst > 0).sum())} nuclear pixels")
    missing = (TRAIN_IDS | VAL_IDS | TEST_IDS) - seen
    if missing:
        raise RuntimeError(f"Kumar archive is missing {sorted(missing)}")
    return counts, images_used


def _convert_test2018(raw_root: Path):
    dest = TEST2018 / "test"
    n_tiles = 0
    n_images = 0
    for stem, tif_path, xml_path in _pairs(raw_root):
        image = np.asarray(Image.open(tif_path).convert("RGB"))
        inst = _rasterize(_polygons(xml_path), image.shape[:2])
        n_tiles += _write_tiles(stem, image, inst, dest)
        n_images += 1
        print(f"test2018: {stem}")
    if n_images == 0:
        raise RuntimeError("MoNuSeg 2018 test archive had no image/xml pairs")
    return n_images, n_tiles


def _fetch_archive(file_ids, archive_path: Path, extract_dir: Path):
    last_error = None
    for file_id in file_ids:
        try:
            _download(file_id, archive_path)
            _extract(archive_path, extract_dir)
            if _pairs(extract_dir):
                return
            print(f"{file_id} extracted but contained no image/xml pairs")
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
        except Exception as exc:
            last_error = exc
            print(f"failed {file_id}: {exc}")
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
    raise RuntimeError(f"could not download a usable archive from {file_ids}") from last_error


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("Fetching MoNuSeg training archive (30 Kumar images)...")
    _fetch_archive(TRAIN_FILE_IDS, RAW / "monuseg_train.zip", RAW / "train")
    print("Fetching MoNuSeg 2018 test archive...")
    _fetch_archive(TEST_FILE_IDS, RAW / "monuseg_test.zip", RAW / "test2018")

    for split in ("train", "val", "test"):
        folder = OUT / split
        if folder.exists():
            shutil.rmtree(folder)
    if (TEST2018 / "test").exists():
        shutil.rmtree(TEST2018 / "test")

    kumar_counts, kumar_images = _convert_kumar(RAW / "train")
    test_images, test_tiles = _convert_test2018(RAW / "test2018")
    print("Kumar tiles", kumar_counts, "images", kumar_images)
    print(f"MoNuSeg 2018 test images {test_images}, tiles {test_tiles}")
    print(f"Wrote {OUT} and {TEST2018}")


if __name__ == "__main__":
    main()
