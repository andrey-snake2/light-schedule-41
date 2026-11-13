import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from PIL import Image


TEMPLATES_FILE = "digit_templates.json"
DATE_X1 = 230
DATE_X2 = 480
GLYPH_SIZE = (28, 36)  # width, height
MIN_DIGIT_SCORE = 0.85
MIN_DIGIT_MARGIN = 0.05
MAX_SHIFT = 2  # pixels in the normalized image
SHIFT_PENALTY = 0.01  # per pixel of translation
DIGITS = "0123456789"


def normalize(region):
    rows, cols = np.nonzero(region)
    if not rows.size:
        raise ValueError("Empty character")
    cropped = region[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    image = Image.fromarray(cropped.astype(np.uint8) * 255)
    return np.asarray(image.resize(GLYPH_SIZE, Image.Resampling.NEAREST)) > 127


def _load_templates():
    try:
        with open(TEMPLATES_FILE, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or set(data) - set(DIGITS):
            raise ValueError("Expected template classes 0 through 9")
        missing = sorted(set(DIGITS) - set(data))
        if missing:
            raise ValueError(f"Missing templates for digits: {', '.join(missing)}")

        references = []
        labels = []
        for digit in DIGITS:
            examples = data[digit]
            if not isinstance(examples, list) or not examples:
                raise ValueError(f"No templates for digit {digit}")
            for example in examples:
                mask = np.asarray(example)
                if (
                    mask.shape != GLYPH_SIZE[::-1]
                    or mask.dtype.kind not in "buif"
                    or not np.all((mask == 0) | (mask == 1))
                    or not mask.any()
                    or mask.all()
                ):
                    raise ValueError(f"Invalid binary template for digit {digit}")
                references.append(mask.astype(np.float32))
                labels.append(digit)
        return np.stack(references), np.asarray(labels)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot load digit templates: {error}. Publication blocked."
        ) from error


def _segment(date_mask):
    # Include one-pixel columns so that small separators survive segmentation.
    occupied = date_mask.any(axis=0)
    edges = np.flatnonzero(np.diff(np.pad(occupied, 1)))
    if len(edges) != 20:
        raise RuntimeError(
            f"Expected 10 date characters, found {len(edges) // 2}. "
            "Publication blocked."
        )
    regions = [date_mask[:, left:right] for left, right in edges.reshape(-1, 2)]
    row_bounds = [
        np.flatnonzero(region.any(axis=1))[[0, -1]] for region in regions
    ]
    digit_indices = [0, 1, 3, 4, 6, 7, 8, 9]
    tops, bottoms = np.asarray(row_bounds)[digit_indices].T
    height = float(np.median(bottoms - tops + 1))
    baseline = float(np.median(bottoms))
    if np.any(bottoms - tops + 1 < 0.75 * height) or np.any(
        np.abs(bottoms - baseline) > 0.20 * height
    ):
        raise RuntimeError("Inconsistent date digit geometry. Publication blocked.")

    for index in (2, 5):
        top, bottom = row_bounds[index]
        if (
            bottom - top + 1 > 0.30 * height
            or regions[index].shape[1] > 0.35 * height
            or abs(bottom - baseline) > 0.20 * height
        ):
            raise RuntimeError(
                f"Invalid date separator at position {index}. Publication blocked."
            )
    return regions, digit_indices


def _classify(character, references, labels, reference_areas, index):
    # Pad both sides of the comparison so shifted ink is never discarded.
    width, height = GLYPH_SIZE
    padded = np.pad(character, 2 * MAX_SHIFT)
    shifts = range(-MAX_SHIFT, MAX_SHIFT + 1)
    offsets = [(dy, dx) for dy in shifts for dx in shifts]
    variants = np.stack([
        padded[
            MAX_SHIFT + dy:height + 3 * MAX_SHIFT + dy,
            MAX_SHIFT + dx:width + 3 * MAX_SHIFT + dx,
        ].ravel()
        for dy, dx in offsets
    ]).astype(np.float32)
    penalties = np.array([SHIFT_PENALTY * (abs(dx) + abs(dy)) for dy, dx in offsets])

    # Dice overlap counts foreground ink and ignores shared background.
    overlap = variants @ references.T
    scores = 2 * overlap / (character.sum() + reference_areas)
    template_scores = (scores - penalties[:, None]).max(axis=0)
    best, runner_up = sorted(
        ((float(template_scores[labels == digit].max()), digit) for digit in DIGITS),
        reverse=True,
    )[:2]
    score, digit = best
    margin = score - runner_up[0]
    if score < MIN_DIGIT_SCORE or margin < MIN_DIGIT_MARGIN:
        raise RuntimeError(
            f"Low recognition confidence at position {index}: {score:.3f} "
            f"(best {digit}, runner-up {runner_up[1]} at {runner_up[0]:.3f}, "
            f"margin {margin:.3f}). Publication blocked."
        )
    return digit, score


def recognize_date(header_mask):
    """Return today's Kyiv date as ISO text, or raise before publication."""
    mask = np.asarray(header_mask)
    if (
        mask.ndim != 2 or not mask.shape[0] or mask.shape[1] < DATE_X2
        or mask.dtype.kind not in "buif"
    ):
        raise RuntimeError("Invalid header mask. Publication blocked.")
    crop = mask[:, DATE_X1:DATE_X2]
    if not np.all((crop == 0) | (crop == 1) | (crop == 255)):
        raise RuntimeError("Expected a binary header mask. Publication blocked.")
    regions, digit_indices = _segment(crop != 0)
    references, labels = _load_templates()
    references = np.pad(
        references,
        ((0, 0), (MAX_SHIFT, MAX_SHIFT), (MAX_SHIFT, MAX_SHIFT)),
    ).reshape(len(labels), -1)
    reference_areas = references.sum(axis=1)

    result = ["."] * 10
    scores = []
    for index in digit_indices:
        result[index], score = _classify(
            normalize(regions[index]), references, labels, reference_areas, index
        )
        scores.append(score)

    date_text = "".join(result)
    try:
        recognized = datetime.strptime(date_text, "%d.%m.%Y").date()
    except ValueError as error:
        raise RuntimeError(
            f"Invalid recognized date {date_text}. Publication blocked."
        ) from error

    try:
        kyiv = ZoneInfo("Europe/Kyiv")
    except ZoneInfoNotFoundError as error:
        raise RuntimeError(
            "Europe/Kyiv time zone is unavailable. Install tzdata or update the "
            "system time zone database. Publication blocked."
        ) from error
    expected = datetime.now(kyiv).date()
    print("Recognized date:", recognized)
    print("Expected date:", expected)
    print("Average digit score:", round(float(np.mean(scores)), 3))
    if recognized != expected:
        raise RuntimeError("DATE NOT VERIFIED. Publication blocked.")

    print("DATE VERIFIED")
    return recognized.isoformat()
