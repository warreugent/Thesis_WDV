from pathlib import Path
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass(frozen=True)
class Params:
    budgets: Tuple[int, ...] = (250, 500, 1000)
    val_frac: float = 0.20
    seed: int = 42
    exclude_iscrowd: bool = True


def load_coco(path: Path) -> dict:
    coco = json.loads(path.read_text(encoding="utf-8"))
    for k in ("images", "annotations", "categories"):
        if k not in coco:
            raise ValueError(f"COCO JSON missing '{k}'")
    return coco


def build_per_image_stats(coco: dict, exclude_iscrowd: bool):
    images_by_id = {int(im["id"]): im for im in coco["images"]}

    # NEW: category id -> name mapping (COCO-style: categories have "id" and "name")
    cat_id_to_name: Dict[int, str] = {}
    for cat in coco["categories"]:
        cid = int(cat["id"])
        cat_id_to_name[cid] = str(cat.get("name", cid))

    anns_by_img = defaultdict(list)
    class_ids: Set[int] = set()

    for ann in coco["annotations"]:
        if exclude_iscrowd and int(ann.get("iscrowd", 0)) == 1:
            continue
        img_id = int(ann["image_id"])
        if img_id not in images_by_id:
            continue
        cid = int(ann["category_id"])
        class_ids.add(cid)
        anns_by_img[img_id].append(ann)

    class_ids = sorted(class_ids)

    # per-image counts and presence
    img_counts: Dict[int, Dict[int, int]] = {}
    img_total: Dict[int, int] = {}
    img_present: Dict[int, Set[int]] = {}

    global_counts = {cid: 0 for cid in class_ids}

    for img_id in images_by_id.keys():
        counts = {cid: 0 for cid in class_ids}
        for ann in anns_by_img.get(img_id, []):
            cid = int(ann["category_id"])
            counts[cid] = counts.get(cid, 0) + 1
        total = sum(counts.values())
        present = {cid for cid, v in counts.items() if v > 0}

        img_counts[img_id] = counts
        img_total[img_id] = total
        img_present[img_id] = present

        for cid, v in counts.items():
            global_counts[cid] = global_counts.get(cid, 0) + v

    if sum(global_counts.values()) == 0:
        raise ValueError("No instances found (after filtering).")

    # NEW: return cat_id_to_name too
    return images_by_id, class_ids, img_counts, img_total, img_present, global_counts, cat_id_to_name


def proportions(global_counts: Dict[int, int]) -> Dict[int, float]:
    tot = sum(global_counts.values())
    return {cid: global_counts[cid] / tot for cid in global_counts.keys()}


def targets_for_budget(budget: int, props: Dict[int, float]) -> Dict[int, int]:
    # rounded targets summing exactly to budget
    cids = list(props.keys())
    raw = {cid: int(round(props[cid] * budget)) for cid in cids}
    drift = budget - sum(raw.values())
    if drift != 0:
        order = sorted(cids, key=lambda c: props[c], reverse=True)
        step = 1 if drift > 0 else -1
        for i in range(abs(drift)):
            raw[order[i % len(order)]] += step
    return raw


def choose_next_image(
    rng: random.Random,
    remaining: Set[int],
    img_present: Dict[int, Set[int]],
    deficits: Dict[int, int],
) -> int:
    # pick a class with remaining deficit (weighted by deficit), then pick a random image containing it
    classes = [c for c, d in deficits.items() if d > 0]
    if not classes:
        # no deficits left: pick any remaining image uniformly
        return rng.choice(tuple(remaining))

    weights = [deficits[c] for c in classes]
    chosen_class = rng.choices(classes, weights=weights, k=1)[0]

    candidates = [i for i in remaining if chosen_class in img_present[i]]
    if candidates:
        return rng.choice(candidates)

    return rng.choice(tuple(remaining))


def build_nested_pools(
    img_ids: List[int],
    class_ids: List[int],
    img_counts: Dict[int, Dict[int, int]],
    img_total: Dict[int, int],
    img_present: Dict[int, Set[int]],
    global_counts: Dict[int, int],
    params: Params,
) -> Dict[int, Set[int]]:
    rng = random.Random(params.seed)
    props = proportions(global_counts)

    selected: Set[int] = set()
    pools: Dict[int, Set[int]] = {}

    all_ids = [i for i in img_ids if img_total.get(i, 0) > 0]
    remaining: Set[int] = set(all_ids)

    current_counts = {cid: 0 for cid in class_ids}
    current_total = 0

    for B in params.budgets:
        target = targets_for_budget(B, props)

        while current_total < B and remaining:
            deficits = {cid: max(0, target[cid] - current_counts.get(cid, 0)) for cid in class_ids}
            img_id = choose_next_image(rng, remaining, img_present, deficits)

            remaining.remove(img_id)
            selected.add(img_id)

            current_total += img_total[img_id]
            for cid, v in img_counts[img_id].items():
                current_counts[cid] = current_counts.get(cid, 0) + v

        pools[B] = set(selected)

    return pools


def split_pool_train_val(
    pool: Set[int],
    class_ids: List[int],
    img_counts: Dict[int, Dict[int, int]],
    img_total: Dict[int, int],
    img_present: Dict[int, Set[int]],
    pool_props: Dict[int, float],
    params: Params,
) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    rng = random.Random(params.seed + 1)

    pool_list = [i for i in pool if img_total.get(i, 0) > 0]
    rng.shuffle(pool_list)

    pool_total = sum(img_total[i] for i in pool_list)
    val_budget = int(round(params.val_frac * pool_total))
    val_targets = targets_for_budget(val_budget, pool_props)

    remaining = set(pool_list)
    val: Set[int] = set()
    val_counts = {cid: 0 for cid in class_ids}
    val_total = 0

    while val_total < val_budget and remaining:
        deficits = {cid: max(0, val_targets[cid] - val_counts.get(cid, 0)) for cid in class_ids}
        img_id = choose_next_image(rng, remaining, img_present, deficits)

        remaining.remove(img_id)
        val.add(img_id)

        val_total += img_total[img_id]
        for cid, v in img_counts[img_id].items():
            val_counts[cid] = val_counts.get(cid, 0) + v

    train = set(pool_list) - val

    train_counts = {cid: 0 for cid in class_ids}
    for i in train:
        for cid, v in img_counts[i].items():
            train_counts[cid] += v

    return train, val, train_counts, val_counts


def write_split(out_dir: Path, images_by_id: Dict[int, dict], train: Set[int], val: Set[int], stats: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_names = sorted(f"../{images_by_id[i]['file_name']}" for i in train)
    val_names = sorted(f"../{images_by_id[i]['file_name']}" for i in val)

    (out_dir / "manifest_train_images.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (out_dir / "manifest_val_images.txt").write_text("\n".join(val_names) + "\n", encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def make_nested_splits(train_json: str | Path, out_root: str | Path, params: Params = Params()):
    train_json = Path(train_json)
    out_root = Path(out_root)

    coco = load_coco(train_json)
    (
        images_by_id,
        class_ids,
        img_counts,
        img_total,
        img_present,
        global_counts,
        cat_id_to_name,   # NEW
    ) = build_per_image_stats(coco, params.exclude_iscrowd)

    img_ids = sorted(images_by_id.keys())
    pools = build_nested_pools(img_ids, class_ids, img_counts, img_total, img_present, global_counts, params)

    for B, pool in pools.items():
        pool_class_counts = {cid: 0 for cid in class_ids}
        pool_total = 0
        for i in pool:
            pool_total += img_total[i]
            for cid, v in img_counts[i].items():
                pool_class_counts[cid] += v
        pool_props = {cid: (pool_class_counts[cid] / pool_total) if pool_total else 0.0 for cid in class_ids}

        train, val, train_counts, val_counts = split_pool_train_val(
            pool, class_ids, img_counts, img_total, img_present, pool_props, params
        )

        # keep mapping for convenience
        class_id_to_name_strkey = {str(cid): cat_id_to_name.get(cid, str(cid)) for cid in class_ids}

        stats = {
            "budget_instances_target": B,
            "budget_instances_actual_pool": int(sum(img_total[i] for i in pool)),
            "seed": params.seed,
            "val_frac": params.val_frac,
            "n_pool_images": len(pool),
            "n_train_images": len(train),
            "n_val_images": len(val),
            "train_instances_total": int(sum(train_counts.values())),
            "val_instances_total": int(sum(val_counts.values())),
            "train_instances_per_class": {str(k): int(v) for k, v in train_counts.items()},
            "val_instances_per_class": {str(k): int(v) for k, v in val_counts.items()},
            "class_id_to_name": class_id_to_name_strkey,
        }

        write_split(out_root / f"inst{B}", images_by_id, train, val, stats)

    print(f"Wrote splits to: {out_root.resolve()}")
