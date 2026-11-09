
import asyncio
import io
import json
import os

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession


# ==========================================
# 1. TELEGRAM CONFIGURATION
# ==========================================

load_dotenv()

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

BOT = "pat_voe_bot"
QUEUE = "4.1"

# Official image: 540 x 714 pixels
X_EDGES = [30, 150, 270, 390, 510]
Y_EDGES = [140, 210, 280, 350, 420, 490, 560]


# ==========================================
# 2. COLOR DETECTION
# ==========================================

def detect_color(region):
    rgb = region.astype(np.int16)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    blue_mask = (
        (red < 110) &
        (green > 100) &
        (blue > 160) &
        (blue > red * 1.5)
    )

    yellow_mask = (
        (red > 180) &
        (green > 120) &
        (blue < 130)
    )

    blue_ratio = float(blue_mask.mean())
    yellow_ratio = float(yellow_mask.mean())

    if blue_ratio > 0.20:
        return 1

    if yellow_ratio > 0.20:
        return 2

    return 0


# ==========================================
# 3. IMAGE ANALYSIS
# ==========================================

def analyze_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    print("Image size:", image.size)

    if image.size != (540, 714):
        raise ValueError(
            "Unexpected image size. Analysis stopped."
        )

    pixels = np.asarray(image)
    half_hours = []

    print("\n30-minute schedule:")
    print("-" * 50)

    for hour in range(24):
        row = hour // 4
        col = hour % 4

        x1 = X_EDGES[col] + 5
        x2 = X_EDGES[col + 1] - 5
        y1 = Y_EDGES[row] + 5
        y2 = Y_EDGES[row + 1] - 5

        middle = (x1 + x2) // 2

        def sample_half(left, right):
            top = pixels[y1:y1 + 12, left:right]
            bottom = pixels[y2 - 12:y2, left:right]

            combined = np.concatenate(
                [top, bottom],
                axis=0
            )

            return detect_color(combined)

        first_half = sample_half(x1, middle)
        second_half = sample_half(middle, x2)

        half_hours.extend([first_half, second_half])

        names = {
            0: "ON",
            1: "OUTAGE",
            2: "PLANNED"
        }

        print(
            f"{hour:02d}:00-{hour:02d}:30 "
            f"{names[first_half]:8s} | "
            f"{hour:02d}:30-{hour + 1:02d}:00 "
            f"{names[second_half]}"
        )

    print("-" * 50)

    # ======================================
    # 4. BUILD OUTAGE INTERVALS
    # ======================================

    intervals = []
    start = None
    current_status = 0

    for index in range(49):
        status = (
            half_hours[index]
            if index < 48
            else 0
        )

        if status != current_status:
            if current_status != 0:
                intervals.append(
                    (start, index, current_status)
                )

            start = index if status != 0 else None
            current_status = status

    def format_time(index):
        minutes = index * 30
        return (
            f"{minutes // 60:02d}:"
            f"{minutes % 60:02d}"
        )

    print("\nDetected outage intervals:")

    for begin, end, status in intervals:
        print(
            format_time(begin),
            "-",
            format_time(end),
            "|",
            "OUTAGE" if status == 1 else "PLANNED"
        )

    total_minutes = sum(
        30 for status in half_hours
        if status != 0
    )

    print(
        "\nTotal scheduled outage:",
        total_minutes / 60,
        "hours"
    )

    # ======================================
    # 5. GENERATE JSON FOR WIDGET
    # ======================================

    schedule = {
        "queue": QUEUE,
        "resolutionMinutes": 30,
        "halfHours": half_hours,
        "intervals": [
            {
                "start": format_time(begin),
                "end": format_time(end),
                "status": (
                    "outage" if status == 1
                    else "planned"
                )
            }
            for begin, end, status in intervals
        ],
        "totalOutageMinutes": total_minutes
    }

    print("\nJSON for widget:")
    print(
        json.dumps(
            schedule,
            ensure_ascii=False,
            indent=2
        )
    )

    return schedule


# ==========================================
# 6. GET IMAGE FROM TELEGRAM
# ==========================================

async def main():

    with open(
        "telegram_session.txt",
        "r",
        encoding="utf-8"
    ) as file:
        session = file.read().strip()

    client = TelegramClient(
        StringSession(session),
        API_ID,
        API_HASH
    )

    async with client:
        bot = await client.get_entity(BOT)

        messages = await client.get_messages(
            bot,
            limit=30
        )

        image_message = next(
            (
                msg for msg in messages
                if not msg.out and msg.photo
            ),
            None
        )

        if image_message is None:
            print("No schedule image found.")
            return

        print("Telegram connected.")
        print("Message ID:", image_message.id)

        image_bytes = await client.download_media(
            image_message,
            file=bytes
        )

        analyze_image(image_bytes)


# ==========================================
# 7. START
# ==========================================

if __name__ == "__main__":
    asyncio.run(main())
