import os
from pathlib import Path
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, infer_device
import torch
from datetime import datetime
import time
import json
import hashlib
import numpy as np

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

    if model_name == "owlv2_16_ensemble": # crashes RAM
        model_id = "owlv2-base-patch16-ensemble"

    if model_name == "mmgd_t":
        model_id = "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"

    if model_name == "mmgd_b_all": # too big for T4 GPU
        model_id = "rziga/mm_grounding_dino_base_all"

    if model_name == "mmgd_l_all": # too big for T4 GPU
        model_id = "rziga/mm_grounding_dino_large_all"

    device = infer_device()
    processor = AutoProcessor.from_pretrained(model_id)#, token=os.environ["HF_TOKEN"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)#, token=os.environ["HF_TOKEN"]).to(device)

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
    # ...
}

# Map all model names from select_model() to a backend identifier.
MODEL_BACKEND_MAP = {
    "gd_t": "grounded_model",
    "gd_b": "grounded_model",
    "owlvit_b_16": "owl_model",
    "owlvit_b_32": "owl_model",
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
    sample_size,
    output_path,
    threshold = None,
    text_threshold = None,
    random_state=None,
    warmup_steps = 1,

):
    """
    Run zero-shot object detection and export results in COCO format.

    This function:
      1. Loads a model and processor via `select_model(model_name)`.
      2. Iterates over images in `images_folder` in batches.
      3. Runs zero-shot detection.
      4. Collects predictions into COCO-style structures.
      5. Calls `write_coco_output` to write a JSON file to the output_path.

    Parameters
    ----------
    images_folder : str
        Folder containing images.
    categories_list : list[str]
        Class names (e.g. ["cat", "dog", "person"]).
    model_name : str
        Model identifier for select_model().
    batch_size : int
        Number of images per batch.
    sample_size : int
        Total number of images to sample from the folder.
    random_state : int or None, optional
        Random seed used in `retrieve_image_batches`.
    threshold : float, optional
        Detection score threshold.
    text_threshold : float, optional
        Text matching threshold for the grounded detection head.
    warmup_steps : int, optional
        Pass how many batches should be processed to warmup the kernel (default = 1)
    """

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    processor, model = select_model(model_name)
    model.eval() # set the model in inference mode

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


    # Simple timing and counting variables
    total_inference_time_s = 0.0
    num_images = 0
    num_pred_boxes = 0

    warmup_steps = warmup_steps  # number of warmup steps
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
                for _ in range(warmup_steps):
                    _ = model(**inputs)
                    if use_cuda:
                        torch.cuda.synchronize()
                first_batch = False

            # forward

            print("len(batch_images):", len(batch_images))
            print("pixel_values shape:", inputs["pixel_values"].shape)


            start = time.perf_counter()
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
            end = time.perf_counter()

            total_inference_time_s += (end - start)
            num_images += len(batch_images)

            print("OLD make_zero_shot_predictions model.device:", model.device)

            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            reserved_mb = torch.cuda.memory_reserved() / 1e6
            print(f"batch={len(batch_images)}  peak={peak_mb:.1f} MB  reserved={reserved_mb:.1f} MB  time={end-start:.4f}s")


            # After model call:
            print("outputs type:", type(outputs))


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
        num_initial_bbox = 0, # zero-shot has no initial boxes
        num_pred_boxes=num_pred_boxes,
        total_inference_time_s=total_inference_time_s,
        machine_training_time_s=0.0,  # zero-shot requires no training time
    )

    print(f"Wrote COCO-format JSON to {out_file}")



def benchmark_batch_size(
    images_folder,
    categories_list,
    model_name,
    batch_size,
    sample_size,
    random_state=0,
    threshold=None,
    text_threshold=None,
    warmup_steps=1,
):
    """
    Benchmark a given batch size:
      - prints CUDA / device info
      - processes `sample_size` images
      - reports throughput and peak GPU memory
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
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("current CUDA device index:", torch.cuda.current_device())
    else:
        print("No CUDA available, running on CPU")

    # Ensure model is on CUDA if available
    if torch.cuda.is_available() and getattr(model, "device", None) is not None:
        if model.device.type != "cuda":
            model = model.to("cuda")
    elif torch.cuda.is_available() and not hasattr(model, "device"):
        # If model has no .device attribute, force to cuda:0
        model = model.to("cuda:0")

    # Fallback: if still no 'device' attribute, assume CPU
    model_device = getattr(model, "device", torch.device("cpu"))
    print("model.device:", model_device)

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
        f"peak_mem={peak_mb:.1f} MB"
    )

    return {
        "batch_size": batch_size,
        "num_images": num_images,
        "total_time_s": total_inference_time_s,
        "images_per_second": images_per_second,
        "peak_mem_mb": peak_mb,
    }


# def make_zero_shot_predictions(
#     images_folder,
#     categories_list,
#     model_name,
#     batch_size,
#     sample_size,
#     output_path,
#     threshold = None,
#     text_threshold = None,
#     random_state=None,
#     warmup_steps = 1,
# ):
#     """
#     Run zero-shot object detection and export results in COCO format.
#     """

#     # -------------------------------------------------------------------------
#     # Setup
#     # -------------------------------------------------------------------------
#     processor, model = select_model(model_name)
#     model.eval()  # set the model in inference mode

#     # Device / CUDA checks
#     print("torch.cuda.is_available():", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("CUDA device count:", torch.cuda.device_count())
#         print("current CUDA device index:", torch.cuda.current_device())
#     else:
#         print("No CUDA available, running on CPU")

#     # If your select_model already moves the model to CUDA, you might not need this.
#     # Keeping it defensive:
#     if torch.cuda.is_available():
#         if getattr(model, "device", None) is not None:
#             if model.device.type != "cuda":
#                 model = model.to("cuda")
#         else:
#             model = model.to("cuda:0")

#     model_device = getattr(model, "device", torch.device("cpu"))
#     print("model.device:", model_device)

#     use_cuda = (model_device.type == "cuda")

#     backend = get_backend(model_name)

#     # Map category name -> category id (0-based)
#     categories_dict = {name: i for i, name in enumerate(categories_list)}

#     images = []
#     annotations = []

#     def image_id_from_name(name: str) -> int:
#         """Stable integer id derived from the image filename."""
#         return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

#     # Labels used for the text encoder
#     text_inputs_cache  = None  # filled the first time we see a batch
#     label_mapping = None       # some backends need a specific mapping of labels that differs from categories_list

#     # Simple timing and counting variables
#     total_inference_time_s = 0.0
#     num_images = 0
#     num_pred_boxes = 0

#     warmup_steps = warmup_steps  # number of warmup steps
#     first_batch = True

#     # -------------------------------------------------------------------------
#     # Reset CUDA peak memory stats for the WHOLE run (nr3)
#     # -------------------------------------------------------------------------
#     if use_cuda:
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats(model_device)
#         torch.cuda.synchronize(model_device)

#     # ------------------------------------------------------------------------- 
#     # Main loop: iterate over image batches 
#     # ------------------------------------------------------------------------- 
#     with torch.inference_mode():
#         for batch_images, names in retrieve_image_batches(
#             images_folder=images_folder,
#             batch_size=batch_size,
#             sample_size=sample_size,
#             random_state=random_state,
#         ):

#             # 1) model-specific text encoding (once)
#             if text_inputs_cache is None:
#                 text_inputs_cache, text_labels, label_mapping = backend.build_text_inputs_cache(
#                     processor=processor,
#                     model=model,
#                     categories_list=categories_list,
#                 )

#             # 2) common image preprocessing
#             inputs = processor(
#                 images=batch_images,
#                 return_tensors="pt",
#                 padding=True,
#             )

#             batch_size_actual = len(batch_images)

#             # 3) model-specific addition of cached text encoding
#             inputs = backend.build_batch_inputs(
#                 model=model,
#                 batch_size_actual=batch_size_actual,
#                 text_inputs_cache=text_inputs_cache,
#                 inputs=inputs,
#             )

#             # 4) common warmup & inference
#             if first_batch:
#                 for _ in range(warmup_steps):
#                     _ = model(**inputs)
#                     if use_cuda:
#                         torch.cuda.synchronize(model_device)
#                 first_batch = False

#             # forward + postprocess are what you time (unchanged)
#             start = time.perf_counter()
#             outputs = model(**inputs)

#             # 5) model-specific postprocess
#             results = backend.postprocess(
#                 processor=processor,
#                 outputs=outputs,
#                 inputs=inputs,
#                 batch_images=batch_images,
#                 text_labels=text_labels,
#                 threshold=threshold,
#                 text_threshold=text_threshold,
#             )

#             if use_cuda:
#                 torch.cuda.synchronize(model_device)
#             end = time.perf_counter()

#             total_inference_time_s += (end - start)
#             num_images += len(batch_images)

#             # 6) common COCO conversion
#             for name, res, im in zip(names, results, batch_images):
#                 img_id = image_id_from_name(name)
#                 H, W = im.height, im.width

#                 images.append({
#                     "id": img_id,
#                     "file_name": f"images/test/{name}",
#                 })

#                 boxes = res["boxes"].tolist()
#                 scores = res["scores"].tolist()
#                 labels = res.get("text_labels", [])

#                 # Remap labels
#                 remapped_labels = [label_mapping.get(l, l) for l in labels]

#                 num_pred_boxes += process_prediction_boxes(
#                     boxes=boxes,
#                     scores=scores,
#                     labels=remapped_labels,
#                     img_id=img_id,
#                     img_h=H,
#                     img_w=W,
#                     categories_dict=categories_dict,
#                     annotations=annotations,
#                 )

#     # -------------------------------------------------------------------------
#     # Peak GPU memory over the whole run
#     # -------------------------------------------------------------------------
#     if use_cuda:
#         peak_bytes = torch.cuda.max_memory_allocated(model_device)
#         peak_mb = peak_bytes / 1e6
#         print(f"Peak GPU memory over run: {peak_mb:.1f} MB")
#     else:
#         print("Peak GPU memory: 0.0 MB (CPU run)")

#     # Optional: throughput log
#     if total_inference_time_s > 0:
#         images_per_second = num_images / total_inference_time_s
#         print(
#             f"Processed {num_images} images in {total_inference_time_s:.2f}s "
#             f"({images_per_second:.2f} images/s)"
#         )

#     # -------------------------------------------------------------------------
#     # Write COCO JSON and log path (unchanged)
#     # -------------------------------------------------------------------------
#     out_file = write_coco_output(
#         images_folder=images_folder,
#         model_name=model_name,
#         categories_list=categories_list,
#         images=images,
#         annotations=annotations,
#         num_images=num_images,
#         output_path=output_path,
#         num_initial_bbox = 0,  # zero-shot has no initial boxes
#         num_pred_boxes=num_pred_boxes,
#         total_inference_time_s=total_inference_time_s,
#         machine_training_time_s=0.0,  # zero-shot requires no training time
#     )

#     print(f"Wrote COCO-format JSON to {out_file}")
