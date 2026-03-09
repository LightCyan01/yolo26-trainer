import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def mask_to_yolo_bbox(
    mask_path: Path,
    image_path: Path,
    class_id: int,
    output_path: Path,
    min_area: int = 25,
    threshold: int = 127
) -> int:
    """
    Convert a pixel mask to YOLO bbox labels.
    Returns number of annotations written.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path))

    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize mask to match image if sizes differ
    img_h, img_w = image.shape[:2]
    if mask.shape != (img_h, img_w):
        logger.warning("Mask/image size mismatch for %s — resizing mask", mask_path.name)
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Binarize soft masks
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            logger.debug("Skipping contour with area %d (below min %d)", area, min_area)
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / img_w
        y_center = (y + bh / 2) / img_h
        norm_w = bw / img_w
        norm_h = bh / img_h
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    output_path.write_text("\n".join(lines))
    logger.info("Wrote %d annotations to %s", len(lines), output_path)
    return len(lines)

