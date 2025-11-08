import random
from pathlib import Path
from PIL import Image


def retrieve_image_batches(images_folder, batch_size, sample_size=None, random_state=None):
    """
    Yield batches of images directly from a given folder.

    Args:
        images_folder (str | Path): Path to folder containing images (searched recursively).
        batch_size (int): Number of images per batch.
        sample_size (int, optional): Limit total number of images to sample.
        random_state (int, optional): Seed for reproducible shuffling.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    paths = [p for p in Path(images_folder).glob("*") if p.suffix.lower() in exts]
    print(f"Found {len(paths)} image files")

    random.seed(random_state)
    random.shuffle(paths)
    if sample_size:
        paths = paths[:sample_size]

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        yield [Image.open(p).convert("RGB").copy() for p in batch], [p.name for p in batch]



if __name__ == "__main__":
    test_root = "../Data/tomatoes/images/val"   # adjust to your dataset path
    for imgs, names in retrieve_image_batches(test_root, batch_size=2, sample_size=4, random_state=0):
        print("Batch filenames:", names)
        print("Batch size:", len(imgs))
        break  # only first batch for quick test
