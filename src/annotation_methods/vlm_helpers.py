import os
from pathlib import Path
import random
import statistics
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, infer_device
import torch
from datetime import datetime
import time
import json
import hashlib
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

# project specific imports
from annotation_methods.io_utils import retrieve_image_batches, write_coco_output, build_results_output_path
from annotation_methods.coco_utils import process_prediction_boxes

def select_model(model_name):
    """
    Map a short model key to its Hugging Face model ID, load the processor
    and zero-shot object detection model, move the model to the inferred
    device, and return both objects.

    The function:
    1. Selects the correct model ID based on `model_name`.
    2. Infers the compute device via `infer_device()`.
    3. Loads the corresponding AutoProcessor and AutoModelForZeroShotObjectDetection.
    4. Returns (processor, model).

    """
    if model_name == "gd_t":
        model_id = "IDEA-Research/grounding-dino-tiny"

    if model_name == "gd_b":
        model_id = "IDEA-Research/grounding-dino-base"

    if model_name == "owlvit_b_16":
        model_id = "google/owlvit-base-patch16"

    if model_name == "owlvit_b_32":
        model_id = "google/owlvit-base-patch32"

    if model_name == "owlv2_16_ensemble": 
        model_id = "owlv2-base-patch16-ensemble"

    if model_name == "owlvit_l_14":
        model_id = "google/owlvit-large-patch14"

    if model_name == "mmgd_t":
        model_id = "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"

    if model_name == "mmgd_b_all": 
        model_id = "rziga/mm_grounding_dino_base_all"

    if model_name == "mmgd_l_all": 
        model_id = "rziga/mm_grounding_dino_large_all"

    device = infer_device()
    processor = AutoProcessor.from_pretrained(model_id)#, token=os.environ["HF_TOKEN"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)#, token=os.environ["HF_TOKEN"])
    model = model.to(device)

    return processor, model

class ZeroShotBackend(ABC):
    @abstractmethod
    def build_text_inputs_cache(self, processor, model, categories_list):
        """
        Return a dict of text tensors on the correct device, ready to be
        repeated per batch. Can be None if the model doesn't use text.
        """
        ...

    @abstractmethod
    def build_batch_inputs(self, model, batch_size_actual, text_inputs_cache, inputs):
        """
        Return a dict of model inputs for this batch, moved to the correct device.
        """
        ...

    @abstractmethod
    def postprocess(
        self,
        processor,
        outputs,
        inputs,
        batch_images,
        text_labels,
        threshold,
        text_threshold,
    ):
        """
        Return a list of per-image results in a common format:
        [
            {
                "boxes": Tensor[N,4],
                "scores": Tensor[N],
                "labels": list[str] or Tensor[N]
            },
            ...
        ]
        """
        ...

class GroundedDetBackend(ZeroShotBackend):
    def build_text_inputs_cache(self, processor, model, categories_list):
        # GroundingDINO expects: "cat. dog. person."
        text_labels = ". ".join(categories_list) + "."

        # Build mapping from category -> category
        label_mapping = {c: c for c in categories_list} # redundant but for consistency

        text_inputs = processor(
            text=text_labels,
            return_tensors="pt",
            padding=True,
        )

        text_inputs = {
            k: v.to(model.device, non_blocking=True)
            for k, v in text_inputs.items()
        }
        return text_inputs, text_labels, label_mapping # pass text_labels for consistency with other methods (grounding dino takes the tokens in the postprocessing step and does not need text_labels)

    def build_batch_inputs(self, model, batch_size_actual, text_inputs_cache, inputs):

        # add cached text encoding (2D -> [B, L])
        if text_inputs_cache is not None:
            for k, v in text_inputs_cache.items():
                inputs[k] = v.repeat(batch_size_actual, 1)

        # move everything to device
        inputs = {
            k: v.to(model.device, non_blocking=True)
            for k, v in inputs.items()
        }
        return inputs

    def postprocess(
        self,
        processor,
        outputs,
        inputs,
        batch_images,
        text_labels,  # unused but kept for consistency
        threshold=None,
        text_threshold=None,
    ):
        target_sizes = [(im.height, im.width) for im in batch_images]

        kwargs = {
            "target_sizes": target_sizes,
        }
        if threshold is not None:
            kwargs["threshold"] = threshold
        if text_threshold is not None:
            kwargs["text_threshold"] = text_threshold

        return processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            **kwargs,
        )
    
class OwlViTBackend(ZeroShotBackend):
    def build_text_inputs_cache(self, processor, model, categories_list):
        # Encode each category as a separate sentence
        text_labels = [f"a photo of a {c}" for c in categories_list]

        # Build mapping from text_label -> original category
        label_mapping = {tl: c for tl, c in zip(text_labels, categories_list)}

        text_inputs = processor(
            text=text_labels,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {
            k: v.to(model.device, non_blocking=True)
            for k, v in text_inputs.items()
        }

        return text_inputs, text_labels, label_mapping


    def build_batch_inputs(self, model, batch_size_actual, text_inputs_cache, inputs):
        # pixel_values are already in `inputs["pixel_values"]` with shape [B, 3, H, W]

        if text_inputs_cache is not None:
            for k, v in text_inputs_cache.items():
                # v: [Q, L]
                if v.dim() != 2:
                    raise ValueError(f"Expected 2D text tensor for {k}, got {v.shape}")

                # Repeat text queries for each image:
                # [Q, L] -> [B*Q, L]
                v = v.repeat(batch_size_actual, 1)
                inputs[k] = v.to(model.device, non_blocking=True)

        # Make sure everything else is on device (including pixel_values)
        inputs = {
            k: v.to(model.device, non_blocking=True)
            for k, v in inputs.items()
        }
        return inputs

    def postprocess(
        self,
        processor,
        outputs,
        inputs,  # unused but kept for interface consistency
        batch_images,
        text_labels,
        threshold=None,
        text_threshold=None,  # unused but kept for interface consistency
    ):
        target_sizes = [(im.height, im.width) for im in batch_images]

        kwargs = {
            "target_sizes": target_sizes,
        }
        if threshold is not None:
            kwargs["threshold"] = threshold

        return processor.post_process_grounded_object_detection(
            outputs,
            text_labels=[text_labels] * len(batch_images),
            **kwargs,
        )

BACKENDS = {
    "grounded_model": GroundedDetBackend(),
    "owl_model": OwlViTBackend(),
}

# Map all model names from select_model() to a backend identifier.
MODEL_BACKEND_MAP = {
    "gd_t": "grounded_model",
    "gd_b": "grounded_model",
    "owlvit_b_16": "owl_model",
    "owlvit_b_32": "owl_model",
    "owlvit_l_14": "owl_model",
    "owlv2_16_ensemble": "owl_model",
    "mmgd_t": "grounded_model",
    "mmgd_b_all": "grounded_model",
    "mmgd_l_all": "grounded_model",
}

def get_backend(model_name: str) -> ZeroShotBackend:
    try:
        backend_key = MODEL_BACKEND_MAP[model_name]
    except KeyError:
        raise ValueError(f"No backend registered for model_name={model_name!r}")

    try:
        return BACKENDS[backend_key]
    except KeyError:
        raise ValueError(f"No backend instance found for backend_key={backend_key!r}")
    
def make_zero_shot_predictions(
    images_folder,
    categories_list,
    model_name,
    batch_size,
    output_path,
    sample_size = None,
    threshold = None,
    text_threshold = None,
    random_state=None,
    warmup_steps = 1,
):
    """
    Run zero-shot object detection and export results in COCO format.
    """

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    processor, model = select_model(model_name)
    model.eval()  # set the model in inference mode
    use_cuda = (model.device.type == "cuda")
    backend = get_backend(model_name)

    # Map category name -> category id (0-based)
    categories_dict = {name: i for i, name in enumerate(categories_list)}

    images = []
    annotations = []

    def image_id_from_name(name: str) -> int:
        """Stable integer id derived from the image filename."""
        return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

    # Labels used for the text encoder
    text_inputs_cache  = None  # filled the first time we see a batch
    label_mapping = None       # some backends need a specific mapping of labels that differs from categories_list

    # Timing and counting variables
    total_inference_time_s = 0.0
    batch_times_s: list[float] = []  # per-batch times (minus warmup)
    num_images = 0
    num_pred_boxes = 0
    first_batch = True

    # -------------------------------------------------------------------------
    # Main loop: iterate over image batches
    # -------------------------------------------------------------------------
    with torch.inference_mode():
        for batch_images, names in retrieve_image_batches(
            images_folder=images_folder,
            batch_size=batch_size,
            sample_size=sample_size,
            random_state=random_state,
        ):
            batch_start = time.perf_counter()
            warmup_time = 0.0

            # 1) model-specific text encoding (once)
            if text_inputs_cache is None:
                text_inputs_cache, text_labels, label_mapping = backend.build_text_inputs_cache(
                    processor=processor,
                    model=model,
                    categories_list=categories_list,
                )

            # 2) common image preprocessing
            inputs = processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )

            batch_size_actual = len(batch_images)

            # 3) model-specific addition of cached text encoding
            inputs = backend.build_batch_inputs(
                model=model,
                batch_size_actual=batch_size_actual,
                text_inputs_cache=text_inputs_cache,
                inputs=inputs,
            )

            # 4) common warmup & inference
            if first_batch:
                warmup_start = time.perf_counter()
                for _ in range(warmup_steps):
                    _ = model(**inputs)
                    if use_cuda:
                        torch.cuda.synchronize()
                warmup_end = time.perf_counter()
                warmup_time = warmup_end - warmup_start
                first_batch = False

            # forward
            outputs = model(**inputs)

            # 5) model-specific postprocess
            results = backend.postprocess(
                processor=processor,
                outputs=outputs,
                inputs=inputs,
                batch_images=batch_images,
                text_labels=text_labels,
                threshold=threshold,
                text_threshold=text_threshold,
            )

            if use_cuda:
                torch.cuda.synchronize()
            batch_end = time.perf_counter()

            # Per-batch inference time (same definition as total_inference_time_s)
            batch_inference_time = batch_end - batch_start - warmup_time
            total_inference_time_s += batch_inference_time
            batch_times_s.append(batch_inference_time)
            num_images += len(batch_images)

            # 6) common COCO conversion
            for name, res, im in zip(names, results, batch_images):
                img_id = image_id_from_name(name)
                H, W = im.height, im.width

                images.append({
                    "id": img_id,
                    "file_name": f"images/test/{name}",
                })

                boxes = res["boxes"].tolist()
                scores = res["scores"].tolist()
                labels = res.get("text_labels", [])

                # Remap labels
                remapped_labels = [label_mapping.get(l, l) for l in labels]

                num_pred_boxes += process_prediction_boxes(
                    boxes=boxes,
                    scores=scores,
                    labels=remapped_labels,
                    img_id=img_id,
                    img_h=H,
                    img_w=W,
                    categories_dict=categories_dict,
                    annotations=annotations,
                )

    # -------------------------------------------------------------------------
    # Write COCO JSON and log path
    # -------------------------------------------------------------------------
    out_file = write_coco_output(
        images_folder=images_folder,
        model_name=model_name,
        categories_list=categories_list,
        images=images,
        annotations=annotations,
        num_images=num_images,
        output_path=output_path,
        num_initial_bbox=0,  # zero-shot has no initial boxes
        num_pred_boxes=num_pred_boxes,
        total_inference_time_s=total_inference_time_s,
        machine_training_time_s=0.0,  # zero-shot requires no training time
        batch_times_s=batch_times_s,
        batch_size=batch_size,
    )

    print(f"Wrote COCO-format JSON to {out_file}")
    return out_file


def benchmark_batch_size(
    images_folder,
    categories_list,
    model_name,
    batch_size,
    sample_size,
    random_state=None,
    threshold=None,
    text_threshold=None,
    warmup_steps=1,
):
    """
    Benchmark a given batch size:
      - prints CUDA / device info
      - processes `sample_size` images
      - reports throughput and peak GPU memory
      - returns a flag if CUDA OOM occurred
    """

    # -------------------------------------------------------------------------
    # Load model + backend
    # -------------------------------------------------------------------------
    processor, model = select_model(model_name)
    model.eval()
    backend = get_backend(model_name)

    # -------------------------------------------------------------------------
    # Device checks
    # -------------------------------------------------------------------------
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No CUDA available, running on CPU")

    model_device = getattr(model, "device", torch.device("cpu"))
    use_cuda = (model_device.type == "cuda")

    # -------------------------------------------------------------------------
    # Text encoder cache etc.
    # -------------------------------------------------------------------------
    text_inputs_cache = None
    text_labels = None
    label_mapping = None

    # -------------------------------------------------------------------------
    # Reset CUDA memory stats for whole run (nr3)
    # -------------------------------------------------------------------------
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(model_device)
        torch.cuda.synchronize(model_device)

    num_images = 0
    total_inference_time_s = 0.0
    first_batch = True
    oom = False

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    with torch.inference_mode():
        for batch_images, names in retrieve_image_batches(
            images_folder=images_folder,
            batch_size=batch_size,
            sample_size=sample_size,
            random_state=random_state,
        ):
            batch_size_actual = len(batch_images)
            if batch_size_actual == 0:
                continue

            try:
                # 1) Text cache (once)
                if text_inputs_cache is None:
                    text_inputs_cache, text_labels, label_mapping = backend.build_text_inputs_cache(
                        processor=processor,
                        model=model,
                        categories_list=categories_list,
                    )

                # 2) Preprocess images
                inputs = processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                )

                # Let backend attach text inputs, move to device, etc.
                inputs = backend.build_batch_inputs(
                    model=model,
                    batch_size_actual=batch_size_actual,
                    text_inputs_cache=text_inputs_cache,
                    inputs=inputs,
                )

                # Optional warmup on the first batch (included in peak memory)
                if first_batch and warmup_steps > 0:
                    for _ in range(warmup_steps):
                        _ = model(**inputs)
                        if use_cuda:
                            torch.cuda.synchronize(model_device)
                    first_batch = False

                # 3) Timed forward + postprocess
                start = time.perf_counter()
                outputs = model(**inputs)

                results = backend.postprocess(
                    processor=processor,
                    outputs=outputs,
                    inputs=inputs,
                    batch_images=batch_images,
                    text_labels=text_labels,
                    threshold=threshold,
                    text_threshold=text_threshold,
                )

                if use_cuda:
                    torch.cuda.synchronize(model_device)
                end = time.perf_counter()

                total_inference_time_s += (end - start)
                num_images += batch_size_actual

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(
                        f"CUDA OOM at batch_size={batch_size} "
                        f"(processed {num_images} images so far)"
                    )
                    oom = True
                    if use_cuda:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(model_device)
                    break
                else:
                    # Re-raise non-OOM errors
                    raise

    # -------------------------------------------------------------------------
    # Throughput and peak memory
    # -------------------------------------------------------------------------
    if total_inference_time_s == 0:
        images_per_second = 0.0
    else:
        images_per_second = num_images / total_inference_time_s

    if use_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(model_device)
        peak_mb = peak_bytes / 1e6
    else:
        peak_mb = 0.0

    print(
        f"[batch={batch_size}] "
        f"images={num_images} | "
        f"time={total_inference_time_s:.2f}s | "
        f"throughput={images_per_second:.2f} img/s | "
        f"peak_mem={peak_mb:.1f} MB | "
        f"oom={oom}"
    )

    metrics = {
        "model": model_name,
        "batch_size": batch_size,
        "num_images": num_images,
        "total_time_s": total_inference_time_s,
        "images_per_second": images_per_second,
        "peak_mem_mb": peak_mb,
        "oom": oom,
}

    df = pd.DataFrame([metrics])
    return metrics, df

from inspect import signature

def make_zero_shot_predictions_minimal(
    images_folder,
    categories_list,
    model_name,
    batch_size,
    sample_size=None,
    random_state=None,
):
    processor, model = select_model(model_name)
    model.eval()
    use_cuda = (model.device.type == "cuda")
    backend = get_backend(model_name)

    text_inputs_cache = None

    # --- Inspect and print true postprocess defaults ---
    fn = processor.post_process_grounded_object_detection
    sig = signature(fn)
    print("post_process_grounded_object_detection defaults:")
    for name, param in sig.parameters.items():
        if param.default is not param.empty:
            print(f"  {name} = {param.default}")
        else:
            print(f"  {name} = <required>")

    with torch.inference_mode():
        for batch_images, names in retrieve_image_batches(
            images_folder=images_folder,
            batch_size=batch_size,
            sample_size=sample_size,
            random_state=random_state,
        ):
            # Build text inputs once
            if text_inputs_cache is None:
                text_inputs_cache, text_labels, label_mapping = backend.build_text_inputs_cache(
                    processor=processor,
                    model=model,
                    categories_list=categories_list,
                )

            # --- Print original image sizes ---
            orig_sizes = [(im.width, im.height) for im in batch_images]
            print("Original sizes:", orig_sizes)

            # Preprocess (resizing + normalization)
            inputs = processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )

            # --- Print processed tensor sizes ---
            tensor_shape = tuple(inputs["pixel_values"].shape)  # (B, C, H, W)
            print("Processed tensor shape:", tensor_shape)

            # Add cached text embeddings
            inputs = backend.build_batch_inputs(
                model=model,
                batch_size_actual=len(batch_images),
                text_inputs_cache=text_inputs_cache,
                inputs=inputs,
            )

            # Forward pass only
            outputs = model(**inputs)
            if use_cuda:
                torch.cuda.synchronize()

            # Stop after first batch (probing only)
            break

