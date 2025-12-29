from pathlib import Path
from PIL import Image
import random
import os
from datetime import datetime
import json

def retrieve_image_batches(images_folder, batch_size, sample_size=None, random_state=None):
    """
    Yield batches of images directly from a given folder.

    Args:
        images_folder (str | Path): Path to folder containing images.
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


def build_results_output_path(output_path, images_folder, model_name):
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

    out_dir = output_path
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
    num_initial_bbox,
    num_pred_boxes,
    machine_training_time_s,
    total_inference_time_s,
    output_path, 
):
    """
    Write prediction results and metadata to a COCO-format JSON file.

    The function assembles COCO-style images, annotations, categories, and
    basic experiment statistics, then writes the result to disk.
    """
    # Build COCO-style output

    info = {
        "description": f"Predictions on {images_folder} with {model_name}",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_images": num_images,
        "num_predicted_bbox": num_pred_boxes,
        "num_initial_bbox": num_initial_bbox,
        "machine_training_time_s": machine_training_time_s,
        "total_inference_time_s": total_inference_time_s,

    }

    categories = [{"id": i, "name": name} for i, name in enumerate(categories_list)]

    coco_output = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_file = build_results_output_path(output_path, images_folder, model_name)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)

    return out_file
