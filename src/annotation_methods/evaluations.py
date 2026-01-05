from __future__ import annotations
import numpy as np
import os, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Dict, Any
import math

from collections import Counter

# Define per-dataset annotation/removal times (seconds)
ANNOTATION_TIMES = {
    "tomatoes": {
        "add":    {"mean": 4.816719160104988, "se": 0.19224942608703616},
        "remove": {"mean": 2.0005069124423964, "se": 0.09765567385259763},
    },
    "apples": {
        "add":    {"mean": 5.452281021897811, "se": 0.10369701538214723},
        "remove": {"mean": 2.0005069124423964, "se": 0.09765567385259763},
    },
}
AVERAGE_BOXES_PER_IMAGE = {
    "tomatoes": 6.79,     
    "apples": 13.61,    
}
DATASETS = ["tomatoes", "apples"]

# Default removal time (seconds)
DEFAULT_ADD_TIME = 5.15  # seconds
DEFAULT_REMOVE_TIME = 2.00  # seconds   

# Define review time per bounding box (seconds)
REVIEW_TIME_PER_BOX = 1/3  # seconds

# Define hourly compute cost ($/hour)
HOURLY_COMPUTE_COST = 2.03081  # $/hour

# Define IoU thresholds for evaluation
EVAL_IOU_THRESHOLDS = [0.5,0.7,0.9]

def _xywh_to_xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)

def _iou(b1, b2):
    x1,y1,x2,y2 = _xywh_to_xyxy(b1)
    x1g, y1g, x2g, y2g = _xywh_to_xyxy(b2)
    ix1, iy1 = max(x1, x1g), max(y1, y1g)
    ix2, iy2 = min(x2, x2g), min(y2, y2g)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def compute_pr_manual(
    gt_annotations,         # list of dicts: {"image_id", "category_id", "bbox"}
    pred_annotations,       # list of dicts: {"image_id", "category_id", "bbox", "score"}
    iou_thr=0.50,
    image_ids_eval=None     # optional: restrict to these GT image_ids
):
    """
    Returns: dict with TP, FP, FN, precision, recall
    Notes:
      - Greedy one-to-one matching per (image_id, category_id)
      - All predictions considered (no maxDets)
      - All gt considered (no area/crowd filtering)
    """

    # group GT and predictions by (image_id, category_id)
    gts = defaultdict(list)
    dts = defaultdict(list)

    if image_ids_eval is not None:
        image_ids_eval = set(image_ids_eval)

    for g in gt_annotations:
        if image_ids_eval is not None and g["image_id"] not in image_ids_eval:
            continue
        gts[(g["image_id"], g["category_id"])].append({"bbox": g["bbox"], "matched": False})

    for d in pred_annotations:
        if image_ids_eval is not None and d["image_id"] not in image_ids_eval:
            continue
        dts[(d["image_id"], d["category_id"])].append({"bbox": d["bbox"], "score": float(d.get("score", 1.0))})

    tp = 0
    fp = 0
    fn = 0

    # iterate over all keys present in either GT or preds
    keys = set(gts.keys()) | set(dts.keys())
    for key in keys:
        gt_list = gts.get(key, [])
        dt_list = dts.get(key, [])

        # sort detections by score desc
        dt_list.sort(key=lambda x: x["score"], reverse=True)

        gt_matched = [False] * len(gt_list)

        # greedy matching
        for det in dt_list:
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_list):
                if gt_matched[j]:
                    continue
                iou = _iou(det["bbox"], gt["bbox"])
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                gt_matched[best_j] = True
                tp += 1
            else:
                fp += 1

        # any unmatched GT are FN
        fn += sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }

def parse_filename(stem: str) -> tuple[str, str, str] | None:
    """
    Parse dataset, subset, model from a filename stem.

    Expected: dataset_subset_model_predictions
    Falls back to splitting at 'predictions' token.
    """
    parts = stem.split("_")

    # Standard pattern: at least 3 parts and last token is 'predictions'
    if len(parts) >= 3 and parts[-1] == "predictions":
        dataset = parts[0]
        subset = parts[1]
        model = "_".join(parts[2:-1])  # between subset and 'predictions'
        return dataset, subset, model

    # Fallback: try to find explicit 'predictions' token
    try:
        pred_idx = parts.index("predictions")
        core = parts[:pred_idx]
    except ValueError:
        core = parts

    if len(core) < 3:
        return None

    dataset = core[0]
    subset = core[1]
    model = "_".join(core[2:])
    return dataset, subset, model

def make_empty_eval(
    dataset_name: str,
    subset_name: str,
    model_name: str,
    iou_type: str,
    iou_thresholds: List[float],
) -> Dict[str, Any]:
    metric_keys = [
        "AP@[.50:.95]", "AP@0.50", "AP@0.75",
        "AP_small", "AP_medium", "AP_large",
        "AR@1", "AR@10", "AR@100",
        "AR_small", "AR_medium", "AR_large",
    ]
    coco_metrics = {k: 0.0 for k in metric_keys}

    # Build conditional efficiency metric dicts
    efficiency_metrics = {
        f"efficiency_metrics@iou_{t:.2f}": {
            "bbox_additions": 0,
            "bbox_removals": 0,
            "precision": 0.0,
            "recall": 0.0,
            # "total_annotation_time_s": 0.0,
            # "annotation_time_per_bbox_s": 0.0,
            # "annotation_time_manual_pct": 0.0,
            # "annotation_time_machine_pct": 0.0,
            # "total_cost_usd": 0.0,
            # "cost_per_bbox_usd": 0.0,
        }
        for t in iou_thresholds
    }

    return {
        "info": {
            "timestamp": "",
            "dataset_name": dataset_name,
            "subset_name": subset_name,
            "model_name": model_name,
            "iou_type": iou_type,
            "num_initial_bbox": 0,
            "machine_training_time_s": 0.0,
            "inference_time_s": {
                "total": 0.0,
                "image_mean": 0.0,
                "image_se": 0.0,
            },
            "num_predicted_bbox": {
                "total": 0,
                "image_mean": 0.0,
                "image_se": 0.0,
            },
            "num_gt_bbox": 0,
            "num_eval_images": 0,
            "num_eval_categories": 0,
        },
        "coco_metrics": coco_metrics,
        **efficiency_metrics,
    }

def mean_and_se_num_pred_per_image(
    per_image_counts,
):
    """
    Mean and standard error of predictions/image.

    Parameters
    ----------
    per_image_counts : array-like
        Vector P_i = number of predictions for each evaluated image.

    Returns
    -------
    dict with:
        mean   -> sample mean
        se     -> standard error of the mean
    """
    x = np.asarray(per_image_counts, dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "se": np.nan}

    mean = float(x.mean())
    sd = float(x.std(ddof=1))   # sample SD
    se = sd / np.sqrt(n)

    return {"mean": mean, "se": se}

def mean_and_se_tinf_per_image(
    batch_size,
    num_eval_images,
    batch_times_s,
    total_inference_time_s=None,
):
    """
    Mean and SE of inference time per image.
    Falls back to mean-only if batch timing info is unavailable.
    """
    # --- fallback path: no batch timing info ---
    if (
        batch_size is None
        or batch_times_s is None
        or len(batch_times_s) == 0
    ):
        if total_inference_time_s is None or num_eval_images in (None, 0):
            return {"mean": None, "se": None}

        mean = float(total_inference_time_s / num_eval_images)
        return {"mean": mean, "se": None}

    # --- full path: batch timing info available ---
    x = np.asarray(batch_times_s, dtype=float)
    n_batches = len(x)
    if n_batches == 0:
        return {"mean": None, "se": None}

    # infer batch sizes (handle smaller last batch)
    batch_sizes = np.full(n_batches, batch_size, dtype=float)
    if num_eval_images % batch_size != 0:
        last = num_eval_images - batch_size * (n_batches - 1)
        if last > 0:
            batch_sizes[-1] = last

    # per-batch per-image rates
    r = x / batch_sizes

    # weighted mean = total time / total images
    mean = float(x.sum() / batch_sizes.sum())

    # weighted SE across batches
    w = batch_sizes
    k = n_batches
    se = float(np.sqrt(np.sum(w * (r - mean) ** 2) / ((k - 1) * np.sum(w))))

    return {"mean": mean, "se": se}

def evaluate_predictions(
    predictions_dir: str | Path,
    iou_thresholds: List[float],
    iou_type: str = "bbox",
) -> List[Dict[str, Any]]:
    """
    Evaluate COCO-style prediction files in a directory.

    Expects prediction files named:
        {dataset}_{subset}_{model}_predictions.json

    Writes evaluations to:
        ../Evaluations/{PREDICTIONS_DIR_NAME}/{dataset}_{subset}_{model}_evaluation.json

    Requirements:
      - Ground truth at: ../Data/{dataset}/annotations/instances_{subset}.json

    Returns:
      - a list of evaluation summaries (one per prediction file)
    """

    # Prepare output directory
    predictions_dir = Path(predictions_dir)
    out_root = Path("..") / "Evaluations" / predictions_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    # Find prediction JSON files
    prediction_jsons = sorted(predictions_dir.glob("*_predictions.json"))
    if not prediction_jsons:
        print(f"No prediction files found in {predictions_dir}")
        return []

    results: List[Dict[str, Any]] = []

    for pred_json in prediction_jsons:
        # Infer dataset, subset, model from filename
        parsed = parse_filename(pred_json.stem)
        if parsed is None:
            print(f"Skip unrecognized filename pattern: {pred_json.name}")
            continue
        dataset_name, subset_name, model_name = parsed

        # Locate GT file
        gt_path = Path("..") / "Data" / dataset_name / "annotations" / f"instances_{subset_name}.json"
        if not gt_path.exists():
            print(f"Missing GT: {gt_path}")
            continue

        # Load COCO GT
        cocoGt = COCO(str(gt_path))

        # Ensure every GT annotation has an `iscrowd` field (default 0)
        for ann in cocoGt.dataset.get("annotations", []):
            ann.setdefault("iscrowd", 0)
        # Rebuild indices after modifying the dataset
        cocoGt.createIndex()

        # Load predictions
        with open(pred_json, "r", encoding="utf-8") as f:
            preds = json.load(f)
        
        # Pass info from the predictions to the eval
        info = preds.get("info", {})

        num_initial_bbox = info.get("num_initial_bbox")
        machine_training_time_s = info.get("machine_training_time_s")
        total_inference_time_s = info.get("total_inference_time_s")

        batch_info = preds.get("batch_info", {}) # Is not existing yet for yolo

        batch_size = batch_info.get("batch_size")
        batch_times_s = batch_info.get("batch_times_s")

        # Map GT and prediction images by filename
        gt_name_to_id = {
            os.path.basename(im["file_name"]): im["id"]
            for im in cocoGt.dataset["images"]
        }

        pred_images = preds.get("images", [])
        pred_id_to_name = {
            im["id"]: os.path.basename(im["file_name"])
            for im in pred_images
        }

        # Image set: all images listed in preds["images"] that exist in GT
        img_ids_eval = sorted({
            gt_name_to_id[os.path.basename(im["file_name"])]
            for im in pred_images
            if os.path.basename(im["file_name"]) in gt_name_to_id
        })

        # If there is no overlap between GT and predictions, write zeros and continue
        if not img_ids_eval:
            eval_summary = make_empty_eval(dataset_name, subset_name, model_name, iou_type, iou_thresholds)
            eval_json = out_root / f"{dataset_name}_{subset_name}_{model_name}_evaluation.json"
            with open(eval_json, "w", encoding="utf-8") as f:
                json.dump(eval_summary, f, indent=2)
            results.append(eval_summary)
            continue

        # ------------------------------------------------------------------
        # Categories: use GT category IDs directly; ignore names completely
        # ------------------------------------------------------------------
        cat_ids_eval = sorted(c["id"] for c in cocoGt.dataset["categories"])

        # ------------------------------------------------------------------
        # Remap detections: only image IDs are remapped; category_ids used as-is
        # ------------------------------------------------------------------
        remapped: List[Dict[str, Any]] = []
        valid_cat_ids = set(cat_ids_eval)

        for ann in preds.get("annotations", []):
            pred_name = pred_id_to_name.get(ann["image_id"])
            if not pred_name:
                continue

            gt_img_id = gt_name_to_id.get(pred_name)
            if gt_img_id is None:
                continue

            cat_id = ann["category_id"]  # must already match GT category_id

            # Optionally drop detections with invalid category_id
            if cat_id not in valid_cat_ids:
                continue

            x, y, w, h = ann["bbox"]

            remapped.append({
                "image_id": gt_img_id,
                "category_id": cat_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(ann.get("score", 0.0)),  # default score 0.0 if missing
            })

        # ACTUAL EVALUATION
        if len(remapped) > 0:
            cocoDt = cocoGt.loadRes(remapped)
            cocoEval = COCOeval(cocoGt, cocoDt, iouType=iou_type)
            cocoEval.params.imgIds = img_ids_eval
            cocoEval.params.catIds = cat_ids_eval

            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            coco_metrics = {
                "AP@[.50:.95]": cocoEval.stats[0],
                "AP@0.50":      cocoEval.stats[1],
                "AP@0.75":      cocoEval.stats[2],
                "AP_small":     cocoEval.stats[3],
                "AP_medium":    cocoEval.stats[4],
                "AP_large":     cocoEval.stats[5],
                "AR@1":         cocoEval.stats[6],
                "AR@10":        cocoEval.stats[7],
                "AR@100":       cocoEval.stats[8],
                "AR_small":     cocoEval.stats[9],
                "AR_medium":    cocoEval.stats[10],
                "AR_large":     cocoEval.stats[11],
            }
        else:
            # no predictions for these images → coco_metrics are all zero
            coco_metrics = {
                "AP@[.50:.95]": 0.0,
                "AP@0.50":      0.0,
                "AP@0.75":      0.0,
                "AP_small":     0.0,
                "AP_medium":    0.0,
                "AP_large":     0.0,
                "AR@1":         0.0,
                "AR@10":        0.0,
                "AR@100":       0.0,
                "AR_small":     0.0,
                "AR_medium":    0.0,
                "AR_large":     0.0,
            }

        # Additional metrics
        num_gt_bbox = sum(
            1 for a in cocoGt.dataset["annotations"]
            if a["image_id"] in img_ids_eval
        )
        num_predicted_bbox = len(remapped)

        # Calculate the mean and SE predictions per image
        ## Count predictions per image among remapped detections
        preds_per_image_counter = Counter(a["image_id"] for a in remapped)
        ## Build vector aligned to img_ids_eval (images with 0 preds get 0)
        per_image_counts = [
            preds_per_image_counter.get(img_id, 0)
            for img_id in img_ids_eval
        ]
        pred_per_image_stats = mean_and_se_num_pred_per_image(per_image_counts)

        # calculate mean and SE of inference time per image
        num_eval_images = len(img_ids_eval)
        inference_stats = mean_and_se_tinf_per_image(
            batch_size=batch_size,
            num_eval_images=num_eval_images,
            batch_times_s=batch_times_s,
            total_inference_time_s=total_inference_time_s,
        )
        # Initialize eval_summary (with empty efficiency_metrics per IoU)
        eval_summary = make_empty_eval(
            dataset_name=dataset_name,
            subset_name=subset_name,
            model_name=model_name,
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
        )

        # Fill info and coco_metrics
        eval_summary["info"].update({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_gt_bbox": num_gt_bbox,
            "num_initial_bbox": num_initial_bbox,
            "machine_training_time_s": machine_training_time_s,
            "num_eval_images": len(img_ids_eval),
            "num_eval_categories": len(cat_ids_eval),
        })
        eval_summary["info"]["num_predicted_bbox"].update({
            "total": num_predicted_bbox,
            "image_mean": pred_per_image_stats["mean"],
            "image_se":   pred_per_image_stats["se"],
        })
        eval_summary["info"]["inference_time_s"].update({
            "total": total_inference_time_s,
            "image_mean": inference_stats["mean"],
            "image_se":   inference_stats["se"],
        })
        eval_summary["coco_metrics"] = coco_metrics

        # Efficiency metrics per IoU threshold
        for iou_thr in iou_thresholds:
            pr_res = compute_pr_manual(
                cocoGt.dataset["annotations"],
                remapped,
                iou_thr=iou_thr,
                image_ids_eval=img_ids_eval,
            )

            bbox_additions = pr_res["fn"]
            bbox_removals  = pr_res["fp"]
            precision_iou  = pr_res["precision"]
            recall_iou     = pr_res["recall"]

            eff_key = f"efficiency_metrics@iou_{iou_thr:.2f}"
            eff_metrics = eval_summary[eff_key]  # already created by make_empty_eval

            eff_metrics.update({
                "bbox_additions": bbox_additions,
                "bbox_removals": bbox_removals,
                "precision": precision_iou,
                "recall": recall_iou,
            })

        # Write eval_summary
        eval_json = out_root / f"{dataset_name}_{subset_name}_{model_name}_evaluation.json"
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        results.append(eval_summary)

    return results

def retrieve_evaluations_info(quality_threshold=0.5, data=None):
    """
    Extract evaluation- and model-related quantities that do not depend on
    dataset_size_images. This can be called once per evaluation JSON.
    """

    # --- Load info from `data` ---
    info = data["info"]

    dataset = info.get("dataset_name")
    num_initial_bbox = info.get("num_initial_bbox", 0)

    t_train = info.get("machine_training_time_s", 0.0)
    image_t_inf = info["inference_time_s"].get("image_mean", 0.0)
    se_image_t_inf = info["inference_time_s"].get("image_se", 0.0)
    if se_image_t_inf is None or (isinstance(se_image_t_inf, float) and math.isnan(se_image_t_inf)):
        se_image_t_inf = 0.0
    image_num_pred = info["num_predicted_bbox"].get("image_mean", 0)
    se_image_num_pred = info["num_predicted_bbox"].get("image_se", 0)

    # Retrieve precision/recall at different IoUs
    prc_05 = data["efficiency_metrics@iou_0.50"].get("precision", 0.0)
    prc_07 = data["efficiency_metrics@iou_0.70"].get("precision", 0.0)
    prc_09 = data["efficiency_metrics@iou_0.90"].get("precision", 0.0)

    rec_05 = data["efficiency_metrics@iou_0.50"].get("recall", 0.0)
    rec_07 = data["efficiency_metrics@iou_0.70"].get("recall", 0.0)
    rec_09 = data["efficiency_metrics@iou_0.90"].get("recall", 0.0)

    # --- Variable assignment depending on dataset and quality threshold ---
    t_add = ANNOTATION_TIMES[dataset]["add"]["mean"]
    se_t_add = ANNOTATION_TIMES[dataset]["add"]["se"]
    t_remove = ANNOTATION_TIMES[dataset]["remove"]["mean"]
    se_t_remove = ANNOTATION_TIMES[dataset]["remove"]["se"]
    t_review = REVIEW_TIME_PER_BOX

    if quality_threshold == 0.5:
        precision = prc_05
        recall = rec_05
    elif quality_threshold == 0.7:
        precision = prc_07
        recall = rec_07
    elif quality_threshold == 0.9:
        precision = prc_09
        recall = rec_09
    else:
        precision = prc_05
        recall = rec_05

    model_name = info.get("model_name", "model")

    return {
        "num_initial_bbox": num_initial_bbox,
        "t_add": t_add,
        "se_t_add": se_t_add,
        "t_remove": t_remove,
        "se_t_remove": se_t_remove,
        "t_review": t_review,
        "t_train": t_train,
        "image_t_inf": image_t_inf,
        "se_image_t_inf": se_image_t_inf,
        "image_num_pred": image_num_pred,
        "se_image_num_pred": se_image_num_pred,
        "precision": precision,
        "recall": recall,
        "dataset": dataset,
        "model_name": model_name,
    }

def perform_efficiency_calculations(
    dataset_size_images,
    *,
    num_initial_bbox,
    t_add,
    t_remove,
    t_review,
    t_train,
    image_t_inf,
    image_num_pred,
    precision,
    recall,
    dataset,
    **_,   # ignore extra keys like model_name
):
    avg_boxes_per_image = AVERAGE_BOXES_PER_IMAGE[dataset]

    # Minimum dataset size needed to cover the initially annotated images
    required_size = num_initial_bbox / avg_boxes_per_image

    # If dataset size is too small → return NaNs for all numeric values
    if dataset_size_images < required_size:
        return {
            "t_init": math.nan,
            "t_corr": math.nan,
            "t_manual": math.nan,
            "t_manual_avg_bbox": t_add,
            "pct_manual_workload_red": 0,
            "t_machine": math.nan,
            "t_total": math.nan,
            "t_avg": math.nan,
            "t_avg_bbox": t_add,
            "c_total": math.nan,
            "c_avg_bbox": math.nan,
            "dataset": dataset,
        }

    # --- Normal path ---
    num_annotated_images = required_size
    images_to_process = max(0, dataset_size_images - num_annotated_images)
    estimated_num_pred_boxes = image_num_pred * images_to_process
    estimated_gt_boxes = avg_boxes_per_image * images_to_process

    t_init = num_initial_bbox * t_add
    t_corr = (
        t_review * estimated_num_pred_boxes
        + (1 - precision) * estimated_num_pred_boxes * t_remove
        + (1 - recall) * estimated_gt_boxes * t_add
    )

    t_manual = t_init + t_corr
    t_machine = t_train + image_t_inf * images_to_process
    t_total = t_manual + t_machine

    t_avg = t_total / dataset_size_images if dataset_size_images > 0 else math.nan
    denom_boxes = estimated_gt_boxes + num_initial_bbox
    t_avg_bbox = t_total / denom_boxes if denom_boxes > 0 else math.nan
    t_manual_avg_bbox = t_manual / denom_boxes if denom_boxes > 0 else math.nan
    pct_manual_workload_red = (
        (1 - (t_manual_avg_bbox / t_add)) * 100
        if t_add > 0
        else math.nan
)

    c_total = (t_machine / 3600) * HOURLY_COMPUTE_COST
    c_avg_bbox = c_total / denom_boxes if denom_boxes > 0 else math.nan


    return {
        "t_init": t_init,
        "t_corr": t_corr,
        "t_manual": t_manual,
        "t_manual_avg_bbox": t_manual_avg_bbox,
        "pct_manual_workload_red": pct_manual_workload_red,
        "t_machine": t_machine,
        "t_total": t_total,
        "t_avg": t_avg,
        "t_avg_bbox": t_avg_bbox,
        "c_total": c_total,
        "c_avg_bbox": c_avg_bbox,
        "dataset": dataset,
    }
