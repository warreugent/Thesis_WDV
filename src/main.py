# empty but not for long
import random
from pathlib import Path
from PIL import Image


def retrieve_image_batches(dataset_folder, batch_size, sample_size=None, random_state=None):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    paths = [p for p in Path(dataset_folder, "images").rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(paths)} image files")

    random.seed(random_state)
    random.shuffle(paths)
    if sample_size:
        paths = paths[:sample_size]

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        yield [Image.open(p).convert("RGB").copy() for p in batch], [p.name for p in batch]