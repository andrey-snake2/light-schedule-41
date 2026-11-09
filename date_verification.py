
import json
from datetime import datetime, timezone, timedelta

import numpy as np
from PIL import Image


TEMPLATES_FILE = "digit_templates.json"

# Coordinates within the cropped header
DATE_X1 = 230
DATE_X2 = 480

# Kyiv timezone for November 2026.
# Temporary fixed offset; we'll use zoneinfo
# for automatic daylight saving time handling.
KYIV_TZ = timezone(timedelta(hours=2))

MIN_DIGIT_SCORE = 0.85


def normalize(region):
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Empty character")

    cropped = region[
        rows[0]:rows[-1] + 1,
        cols[0]:cols[-1] + 1
    ]

    image = Image.fromarray(
        cropped.astype(np.uint8) * 255
    )

    image = image.resize(
        (28, 36),
        Image.Resampling.NEAREST
    )

    return (
        np.asarray(image) > 127
    ).astype(np.uint8)


def recognize_date(header_mask):
    with open(
        TEMPLATES_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        templates = json.load(f)

    date_mask = header_mask[
        :, DATE_X1:DATE_X2
    ]

    columns = date_mask.sum(axis=0)

    groups = []
    start = None

    for x, count in enumerate(columns):
        if count >= 2 and start is None:
            start = x
        elif count < 2 and start is not None:
            if x - start >= 3:
                groups.append((start, x))
            start = None

    if start is not None:
        groups.append((start, len(columns)))

    if len(groups) != 10:
        raise RuntimeError(
            f"Expected 10 date characters, found {len(groups)}"
        )

    result = ""
    scores = []

    for index, (x1, x2) in enumerate(groups):
        if index in (2, 5):
            result += "."
            continue

        character = normalize(
            date_mask[:, x1:x2]
        )

        candidates = []

        for digit, examples in templates.items():
            for example in examples:
                reference = np.asarray(
                    example,
                    dtype=np.uint8
                )

                similarity = np.mean(
                    character == reference
                )

                candidates.append(
                    (similarity, digit)
                )

        best_score, best_digit = max(candidates)

        if best_score < MIN_DIGIT_SCORE:
            raise RuntimeError(
                f"Low recognition confidence at "
                f"position {index}: {best_score:.3f}"
            )

        result += best_digit
        scores.append(best_score)

    recognized = datetime.strptime(
        result, "%d.%m.%Y"
    ).date()

    expected = datetime.now(
        KYIV_TZ
    ).date()

    print("Recognized date:", recognized)
    print("Expected date:", expected)
    print(
        "Average digit score:",
        round(float(np.mean(scores)), 3)
    )

    if recognized != expected:
        raise RuntimeError(
            "DATE NOT VERIFIED. Publication blocked."
        )

    print("DATE VERIFIED")

    return recognized.isoformat()
