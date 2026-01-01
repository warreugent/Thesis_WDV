from __future__ import annotations

import json
import shutil
from pathlib import Path
from collections import defaultdict
from collections.abc import Sequence
from typing import Union, Dict, List, Set, Tuple
from ultralytics import YOLO
import random
from dataclasses import dataclass
import time
import cv2

from annotation_methods.io_utils import write_coco_output


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


from pathlib import Path

def convert_inst_split_to_yolo(
    *,
    inst: int,
    repeat: int,
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
      {splits_root}/inst{inst}_r{repeat}/manifest_train_images.txt
      {splits_root}/inst{inst}_r{repeat}/manifest_val_images.txt
      {splits_root}/inst{inst}_r{repeat}/images/train, images/val
      {splits_root}/inst{inst}_r{repeat}/labels/train, labels/val
    """
    splits_root = Path(splits_root).expanduser().resolve()
    inst_dir = (splits_root / f"inst{inst}_r{repeat}").resolve()

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

    return {
        "inst": inst,
        "repeat": repeat,
        "train": stats_train,
        "val": stats_val,
    }


def convert_all_inst_splits_to_yolo(
    *,
    inst_values: Sequence[int],
    num_repeats: int,
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
        for repeat in range(num_repeats):
            res = convert_inst_split_to_yolo(
                inst=inst,
                repeat=repeat,
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


def train_yolo_model(
    dataset: str,
    split: str | Path,
    model_weights: str,
    data_root: Path,
    yaml_name: str | Path,
    runs_dir: Path,
    stats_json_name: str = "stats.json",
):
    """
    Train a YOLO model for a given dataset + split and
    save timing stats to training_info.json and store the 
    model in the appropriate runs/dataset folder.
    """

    # ---- Paths derived from inputs ----
    split_path = Path(split)
    data_dir = data_root / dataset / split_path

    data_yaml_path = data_dir / yaml_name
    stats_json_path = data_dir / stats_json_name

    project_dir = runs_dir / dataset

    # derive model base name from weights (e.g. "yolo11s" from "yolo11s.pt")
    model_base = Path(model_weights).stem  # "yolo11s"
    num_instances_repeat = split_path.name.replace("inst", "")  # e.g. "250_r0"
    model_name = f"{model_base}_inst{num_instances_repeat}"

    # ---- Load model ----
    model = YOLO(model_weights)

    # ---- Train and time ----
    start = time.perf_counter()
    train_results = model.train(
        data=str(data_yaml_path),
        epochs=100,
        imgsz=640,
        device="cuda",
        project=str(project_dir),
        name=model_name,
    )
    end = time.perf_counter()
    machine_training_time_s = end - start

    # ---- Read stats.json if available ----
    if stats_json_path.exists():
        with stats_json_path.open("r") as f:
            stats = json.load(f)
        num_initial_bbox = stats.get("budget_instances_actual_pool", 0)
    else:
        num_initial_bbox = 0

    # ---- Save training info ----
    output_path = project_dir / model_name / "training_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_info = {
        "machine_training_time_s": machine_training_time_s,
        "num_initial_bbox": num_initial_bbox,
    }

    with output_path.open("w") as f:
        json.dump(training_info, f, indent=2)

    # ---- Return useful outputs for further processing ----
    return {
        "dataset": dataset,
        "split": str(split_path),
        "model_weights": model_weights,
        "model_name": model_name,
        "project_dir": project_dir,
        "train_results": train_results,
        "machine_training_time_s": machine_training_time_s,
        "num_initial_bbox": num_initial_bbox,
        "training_info_path": output_path,
    }

def run_inference(
    *,
    dataset: str,
    model_name: str,
    data_root: Path,
    runs_dir: Path,
    outdir_base: Path,
    warmup_required: bool = True,
):
    """
    Run a YOLO model on the test split for a given dataset and
    save timing + bbox stats in outdir_base/experiment_name with:
      - YOLO bbox information in 'labels/' as txt files
      - timing information in timing_info.json.

    All directory roots are passed explicitly.
    """

    # ---- Paths and names ----
    experiment_name = f"{dataset}_test_{model_name}"

    model_path = runs_dir / dataset / model_name / "weights" / "best.pt"
    training_info_path = runs_dir / dataset / model_name / "training_info.json"
    test_images_path = data_root / dataset / "images" / "test"

    outdir_base = Path(outdir_base)
    exp_outdir = outdir_base / experiment_name

    # ---- prepare output directory ----
    exp_outdir.mkdir(parents=True, exist_ok=True)

    labels_dir = exp_outdir / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    model = YOLO(str(model_path))

    # ---- run a small warmup ----
    if warmup_required:
        warmup_images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            warmup_images.extend(test_images_path.glob(ext))
        warmup_images = warmup_images[:4] #take the first 4

        model.predict(
            source=[str(p) for p in warmup_images],
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
        )

    # ---- run inference ----
    start = time.perf_counter()
    model.predict(
        source=str(test_images_path),
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(outdir_base),
        name=experiment_name,
        exist_ok=True,
    )
    total_inference_time_s = time.perf_counter() - start

    # ---- read training info ----
    with training_info_path.open("r") as f:
        training_info = json.load(f)

    timing_info = {
        "total_inference_time_s": total_inference_time_s,
        "machine_training_time_s": training_info["machine_training_time_s"],
        "num_initial_bbox": training_info["num_initial_bbox"],
    }

    # ---- write timing info ----
    timing_info_path = exp_outdir / "timing_info.json"
    with timing_info_path.open("w") as f:
        json.dump(timing_info, f, indent=2)

    # ---- return metadata ----
    return {
        "dataset": dataset,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "experiment_dir": exp_outdir,
        "model_path": model_path,
        "test_images_path": test_images_path,
        "total_inference_time_s": total_inference_time_s,
        "timing_info_path": timing_info_path,
    }

def yolo_pred_txt_to_coco_results(
    *,
    dataset: str,
    model_name: str,
    categories_list,
    data_root: Path,
    outdir_base: Path,
    results_path: Path,
    is_xywh_normalized: bool = True,
    has_conf: bool = True,
):
    """
    Convert YOLO prediction .txt files into a COCO-format JSON file written via
    `write_coco_output()`.

    YOLO txt format assumed: cls xc yc w h [conf]
    """

    experiment_name = f"{dataset}_test_{model_name}"

    test_images_path = data_root / dataset / "images" / "test"
    yolo_experiment_outputs = Path(outdir_base) / experiment_name
    yolo_labels_path = yolo_experiment_outputs / "labels"
    yolo_timing_info_path = yolo_experiment_outputs / "timing_info.json"

    # Index images by stem (filename without extension)
    img_index = {}
    for p in test_images_path.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            img_index[p.stem] = p

    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    num_pred_boxes = 0

    for txt in sorted(yolo_labels_path.glob("*.txt")):
        stem = txt.stem
        img_path = img_index.get(stem)
        if img_path is None:
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        images.append(
            {
                "id": img_id,
                "file_name": f"images/test/{img_path.name}",
                "width": int(w),
                "height": int(h),
            }
        )

        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                cls = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                conf = float(parts[5]) if (has_conf and len(parts) >= 6) else 1.0

                if is_xywh_normalized:
                    xc *= w
                    yc *= h
                    bw *= w
                    bh *= h

                x = xc - bw / 2.0
                y = yc - bh / 2.0

                coco_cat = cls  # assuming 0-based indexing

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(coco_cat),
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "score": float(conf),
                    }
                )
                ann_id += 1
                num_pred_boxes += 1

        img_id += 1

    # Read timing info
    with yolo_timing_info_path.open("r") as f:
        timing_info = json.load(f)

    machine_training_time_s = float(timing_info.get("machine_training_time_s", 0.0))
    total_inference_time_s = float(timing_info.get("total_inference_time_s", 0.0))
    num_initial_bbox = int(timing_info.get("num_initial_bbox", 0))

    return write_coco_output(
        images_folder=str(test_images_path),
        model_name=model_name,
        categories_list=categories_list,
        images=images,
        annotations=annotations,
        num_images=len(images),
        num_initial_bbox=num_initial_bbox,
        num_pred_boxes=num_pred_boxes,
        machine_training_time_s=machine_training_time_s,
        total_inference_time_s=total_inference_time_s,
        output_path=str(results_path),
    )