

def process_prediction_boxes(
    boxes, scores, labels, img_id, img_h, img_w, categories_dict, annotations
):
    """
    Apply clamping, ordering, xyxy→xywh conversion, label lookup,
    and append valid COCO annotations to `annotations`.

    Parameters
    ----------
    boxes : list[list[float]]
        List of [x1, y1, x2, y2] boxes.
    scores : list[float]
    labels : list[str]
    img_id : int
    img_h, img_w : int
        Image height and width.
    categories_dict : dict[str, int]
        Mapping from label string to category_id.
    annotations : list[dict]
        The list that will be appended to.

    Returns
    -------
    int
        Number of new boxes added.
    """
    added = 0

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(float, box)

        # Clamp to image bounds
        x1 = max(0.0, min(x1, img_w))
        y1 = max(0.0, min(y1, img_h))
        x2 = max(0.0, min(x2, img_w))
        y2 = max(0.0, min(y2, img_h))

        # Enforce ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w == 0.0 or h == 0.0:
            continue

        cid = categories_dict.get(label)
        if cid is None:
            continue

        ann_id = len(annotations) + 1
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cid,
            "bbox": [x1, y1, w, h],
            "score": float(score),
        })

        added += 1

    return added
