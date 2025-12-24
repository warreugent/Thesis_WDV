from __future__ import annotations

import json
import shutil
from pathlib import Path
from collections import defaultdict


def coco_manifest_to_yolo(
    coco_json: str | Path,
    manifest_txt: str | Path,
    dataset_root: str | Path,
    images_dst_dir: str | Path,
    labels_dst_dir: str | Path,
    *,
    exclude_iscrowd: bool = True,
    make_empty_label_files: bool = True,
    flatten_images: bool = True,  # kept for API compatibility; currently images are flattened by rel_p.name
) -> dict:
    """
    Convert COCO JSON annotations to YOLO labels using a manifest of image paths.
    """
    coco_json = Path(coco_json).expanduser().resolve()
    manifest_txt = Path(manifest_txt).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    images_dst_dir = Path(images_dst_dir).expanduser().resolve()
    labels_dst_dir = Path(labels_dst_dir).expanduser().resolve()

    images_dst_dir.mkdir(parents=True, exist_ok=True)
    labels_dst_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images or categories is None:
        raise ValueError("COCO JSON missing 'images' or 'categories'.")

    # COCO category_id -> YOLO class index
    cat_ids = sorted(int(c["id"]) for c in categories)
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    images_by_id = {int(im["id"]): im for im in images}
    file_to_imgid = {str(im["file_name"]): int(im["id"]) for im in images}

    anns_by_img = defaultdict(list)
    for ann in annotations:
        if exclude_iscrowd and int(ann.get("iscrowd", 0)) == 1:
            continue
        img_id = int(ann["image_id"])
        if img_id in images_by_id:
            anns_by_img[img_id].append(ann)

    manifest_lines = [
        line.strip()
        for line in manifest_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    n_missing_images_on_disk = 0
    n_missing_in_coco = 0
    n_labels_written = 0
    n_processed = 0

    for rel_path in manifest_lines:
        rel_p = Path(rel_path)

        # Resolve relative to the MANIFEST file, not the dataset root
        src_img = (manifest_txt.parent / rel_p).resolve()

        # Find COCO image_id
        img_id = file_to_imgid.get(rel_path) or file_to_imgid.get(rel_p.as_posix())

        if img_id is None:
            # Fallback: match by filename only
            matches = [k for k in file_to_imgid if Path(k).name == rel_p.name]
            if len(matches) == 1:
                img_id = file_to_imgid[matches[0]]
            else:
                n_missing_in_coco += 1
                if src_img.exists():
                    dst_img = images_dst_dir / rel_p.name
                    shutil.copy2(src_img, dst_img)
                else:
                    n_missing_images_on_disk += 1
                continue

        im = images_by_id[img_id]
        W, H = int(im["width"]), int(im["height"])

        # Copy image (flatten by filename)
        dst_img = images_dst_dir / rel_p.name
        if not src_img.exists():
            print(f"MISSING ON DISK: {src_img}")
            n_missing_images_on_disk += 1
        else:
            shutil.copy2(src_img, dst_img)

        # Write YOLO label
        label_path = labels_dst_dir / f"{rel_p.stem}.txt"
        lines = []

        for ann in anns_by_img.get(img_id, []):
            cid = int(ann["category_id"])
            if cid not in cat_id_to_idx:
                continue

            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue

            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            ww = w / W
            hh = h / H

            lines.append(
                f"{cat_id_to_idx[cid]} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"
            )

        if lines or make_empty_label_files:
            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        n_labels_written += len(lines)
        n_processed += 1

    return {
        "processed_images": n_processed,
        "labels_written": n_labels_written,
        "missing_images_on_disk": n_missing_images_on_disk,
        "missing_manifest_entries_in_coco": n_missing_in_coco,
        "nc": len(cat_ids),
        "class_names": [
            c["name"] for c in sorted(categories, key=lambda x: int(x["id"]))
        ],
    }


def convert_inst_split_to_yolo(
    *,
    inst: int,
    coco_json: str | Path,
    dataset_root: str | Path,
    splits_root: str | Path,
    exclude_iscrowd: bool = True,
    make_empty_label_files: bool = True,
    flatten_images: bool = True,
    manifest_train_name: str = "manifest_train_images.txt",
    manifest_val_name: str = "manifest_val_images.txt",
) -> dict:
    """
    Run conversion for one inst split (e.g. inst=250) for train+val manifests.

    Expected layout:
      {splits_root}/inst{inst}/manifest_train_images.txt
      {splits_root}/inst{inst}/manifest_val_images.txt
      {splits_root}/inst{inst}/images/train, images/val
      {splits_root}/inst{inst}/labels/train, labels/val
    """
    splits_root = Path(splits_root).expanduser().resolve()
    inst_dir = (splits_root / f"inst{inst}").resolve()

    train_manifest = inst_dir / manifest_train_name
    val_manifest = inst_dir / manifest_val_name

    stats_train = coco_manifest_to_yolo(
        coco_json=coco_json,
        manifest_txt=train_manifest,
        dataset_root=dataset_root,
        images_dst_dir=inst_dir / "images" / "train",
        labels_dst_dir=inst_dir / "labels" / "train",
        exclude_iscrowd=exclude_iscrowd,
        make_empty_label_files=make_empty_label_files,
        flatten_images=flatten_images,
    )

    stats_val = coco_manifest_to_yolo(
        coco_json=coco_json,
        manifest_txt=val_manifest,
        dataset_root=dataset_root,
        images_dst_dir=inst_dir / "images" / "val",
        labels_dst_dir=inst_dir / "labels" / "val",
        exclude_iscrowd=exclude_iscrowd,
        make_empty_label_files=make_empty_label_files,
        flatten_images=flatten_images,
    )

    return {"inst": inst, "train": stats_train, "val": stats_val}


def convert_all_inst_splits_to_yolo(
    *,
    inst_values: list[int],
    coco_json: str | Path,
    dataset_root: str | Path,
    splits_root: str | Path,
    exclude_iscrowd: bool = True,
    make_empty_label_files: bool = True,
    flatten_images: bool = True,
) -> list[dict]:
    """
    Run conversion for multiple inst splits and return a list of per-inst stats.
    """
    results: list[dict] = []
    for inst in inst_values:
        res = convert_inst_split_to_yolo(
            inst=inst,
            coco_json=coco_json,
            dataset_root=dataset_root,
            splits_root=splits_root,
            exclude_iscrowd=exclude_iscrowd,
            make_empty_label_files=make_empty_label_files,
            flatten_images=flatten_images,
        )
        results.append(res)
    return results

def write_yolo_dataset_yaml(yolo_dir, stats_json=None, out_yaml=None):
    yolo_dir = Path(yolo_dir).expanduser()
    stats_json = Path(stats_json).expanduser() if stats_json else yolo_dir / "stats.json"
    out_yaml = Path(out_yaml).expanduser() if out_yaml else yolo_dir / "dataset.yaml"

    data = json.loads(stats_json.read_text(encoding="utf-8"))
    mapping = data["class_id_to_name"]

    items = sorted((int(k), v) for k, v in mapping.items())
    names = [name for _, name in items]

    yaml_text = (
        f"path: {yolo_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  - {n}\n" for n in names)
    )

    out_yaml.write_text(yaml_text, encoding="utf-8")
    return out_yaml
