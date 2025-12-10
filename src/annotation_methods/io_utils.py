from pathlib import Path
from PIL import Image
import random
import os
from datetime import datetime
import json



# the print takes a long time so inspect to use a native package or another solution to do this faster
def retrieve_image_batches(images_folder, batch_size, sample_size=None, random_state=None):
    """
    Yield batches of images directly from a given folder.

    Args:
        images_folder (str | Path): Path to folder containing images (searched recursively).
        batch_size (int): Number of images per batch.
        sample_size (int, optional): Limit total number of images to sample.
        random_state (int, optional): Seed for reproducible shuffling if needed.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    paths = []
    with os.scandir(images_folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(exts):
                paths.append(Path(entry.path))
    print(f"Found {len(paths)} image files")

    if random_state is not None:
        random.seed(random_state)
        random.shuffle(paths)
    if sample_size:
        paths = paths[:sample_size]

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        yield [Image.open(p).convert("RGB").copy() for p in batch], [p.name for p in batch]




def build_results_output_path(experiment_name, images_folder, model_name):
    """
    Construct a standardized output file path for prediction results.

    Steps:
    1. Extract the name of the directory two levels above `images_folder`
       to include as part of the output filename.
    2. Extract the base name of `images_folder` (without extension) 
       to further identify the dataset or batch.
    3. Create a results directory at ../Results/<experiment_name>.
    4. Build a JSON filename combining parent name, base name, and model name.
    5. Return the full path to the output JSON file.
    """
    parent = os.path.basename(os.path.dirname(os.path.dirname(images_folder)))
    base = os.path.splitext(os.path.basename(images_folder))[0]

    out_dir = os.path.join("..", "Results", experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{parent}_{base}_{model_name}_predictions.json")
    return out_file


def write_coco_output(
    images_folder,
    model_name,
    categories_list,
    images,
    annotations,
    num_images,
    num_boxes,
    total_time,
    experiment_name, 
    gpu_hourly_price=2.5,
    gpu_tdp_watts=250.0,
    gpu_utilization_factor=0.9,
    power_utilization_factor=0.7,
    electricity_price_per_kwh=0.30,
):
    """
    Build and write a COCO-style JSON file with predictions and metadata.

    This function:
    1. Computes basic timing statistics:
       - Average inference time per image.
       - Average inference time per bounding box.
    2. Estimates infrastructure and energy costs for the inference run:
       - GPU hours used (scaled by `gpu_utilization_factor`).
       - Infrastructure cost in EUR using `gpu_hourly_price`.
       - Energy consumption in kWh (using GPU TDP and `power_utilization_factor`).
       - Energy cost in EUR using `electricity_price_per_kwh`.
       - Total cost and cost per predicted bounding box.
    3. Assembles a COCO-style dictionary with:
       - `info`: run description, date, timing, and cost statistics.
       - `images`: list of image entries (already in COCO-compatible format).
       - `annotations`: list of prediction annotations.
       - `categories`: list of category definitions derived from `categories_list`.
    4. Builds an output path via `build_output_path` and writes the JSON file there.

    Parameters
    ----------
    images_folder : str
        Path to the folder containing the input images.
    model_name : str
        Name or identifier of the model used for predictions.
    categories_list : list[str]
        List of category names; each is converted into a COCO category with an ID.
    images : list[dict]
        COCO-style image entries.
    annotations : list[dict]
        COCO-style annotation entries for predicted bounding boxes.
    num_images : int
        Number of images processed.
    num_boxes : int
        Number of predicted bounding boxes.
    total_time : float
        Total inference time in seconds over all images.
    gpu_hourly_price : float, optional
        Cost of GPU usage per hour in EUR.
    gpu_tdp_watts : float, optional
        Thermal Design Power (TDP) of the GPU in watts.
    gpu_utilization_factor : float, optional
        Factor (0-1) to account for average GPU utilization over the run.
    power_utilization_factor : float, optional
        Factor (0-1) to account for effective power draw vs. TDP.
    electricity_price_per_kwh : float, optional
        Electricity price per kWh in EUR.

    Returns
    -------
    str
        Path to the written COCO JSON output file.
    """
    # ---- timing statistics ----
    avg_time_image = total_time / num_images if num_images else 0.0
    avg_time_bbox = total_time / num_boxes if num_boxes else 0.0

    # ---- cost / energy estimates ----
    gpu_hours = (total_time / 3600.0) * gpu_utilization_factor
    infra_cost_eur = gpu_hours * gpu_hourly_price
    energy_kwh = (gpu_tdp_watts / 1000.0) * (total_time / 3600.0) * power_utilization_factor
    energy_cost_eur = energy_kwh * electricity_price_per_kwh
    total_cost_eur = infra_cost_eur + energy_cost_eur

    # cost_per_image_eur = total_cost_eur / num_images if num_images else 0.0
    cost_per_bbox_eur = total_cost_eur / num_boxes if num_boxes else 0.0

    info = {
        "description": f"Predictions on {images_folder} with {model_name}",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_images": num_images,
        "num_predicted_bbox": num_boxes,
        "avg_inference_time_s_image": avg_time_image,
        "avg_inference_time_s_bbox": avg_time_bbox,
        "total_inference_time_s": total_time,
        "gpu_hours_estimate": gpu_hours,
        "total_cost_eur": total_cost_eur,
        "cost_per_bbox_eur": cost_per_bbox_eur,
    }

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(categories_list)]

    coco_output = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_file = build_results_output_path(experiment_name, images_folder, model_name)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)

    return out_file
