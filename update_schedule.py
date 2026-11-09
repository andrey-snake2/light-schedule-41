
import asyncio
import io
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession

from analyze_schedule import analyze_image
from date_verification import recognize_date


# ==========================================
# 1. SETTINGS
# ==========================================

load_dotenv()

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

BOT = "pat_voe_bot"
ACCOUNT_TEXT = "Немирівське шосе"

EXPECTED_QUEUE = "4.1"
IMAGE_SIZE = (540, 714)

HEADER_BOX = (20, 35, 520, 110)
QUEUE_BOX = (0, 0, 210, 75)

REFERENCE_FILE = "queue_reference.png"
OUTPUT_FILE = "schedule_test.json"

MIN_QUEUE_SIMILARITY = 0.98


# ==========================================
# 2. TELEGRAM SESSION
# ==========================================

def get_telegram_session():
    """
    GitHub Actions:
        TELEGRAM_SESSION environment variable

    Local Windows:
        telegram_session.txt
    """

    session = os.environ.get(
        "TELEGRAM_SESSION", ""
    ).strip()

    if session:
        print("Using Telegram session from environment.")
        return session

    session_path = Path("telegram_session.txt")

    if not session_path.is_file():
        raise RuntimeError(
            "Telegram session is not configured. "
            "Set TELEGRAM_SESSION or create "
            "telegram_session.txt."
        )

    session = session_path.read_text(
        encoding="utf-8"
    ).strip()

    if not session:
        raise RuntimeError(
            "Telegram session is empty."
        )

    print("Using local Telegram session file.")

    return session


# ==========================================
# 3. HEADER MASK
# ==========================================

def extract_header_mask(image):

    header = image.crop(HEADER_BOX)

    pixels = np.asarray(header).astype(np.int16)

    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    blue_background = (
        (blue > red + 20)
        & (blue > green + 5)
        & (red < 190)
    )

    white = (
        (red > 220)
        & (green > 220)
        & (blue > 220)
    )

    valid_rows = (
        blue_background.sum(axis=1) > 100
    )

    valid_columns = (
        blue_background.sum(axis=0) > 30
    )

    mask = (
        white
        & valid_rows[:, np.newaxis]
        & valid_columns[np.newaxis, :]
    )

    return mask


# ==========================================
# 4. VERIFY QUEUE 4.1
# ==========================================

def foreground_score(expected, actual):

    intersection = np.logical_and(
        expected, actual
    ).sum()

    total = expected.sum() + actual.sum()

    if total == 0:
        return 0.0

    return float(
        2 * intersection / total
    )


def verify_queue(header_mask):

    reference_path = Path(REFERENCE_FILE)

    if not reference_path.is_file():
        raise RuntimeError(
            "Queue reference file is missing."
        )

    reference = (
        np.asarray(
            Image.open(reference_path).convert("L")
        ) > 127
    )

    x1, y1, x2, y2 = QUEUE_BOX

    current = header_mask[
        y1:y2,
        x1:x2
    ]

    if reference.shape != current.shape:
        raise RuntimeError(
            "Queue reference dimensions do not match."
        )

    score = foreground_score(
        reference, current
    )

    print(
        "Queue similarity:",
        round(score, 4)
    )

    if score < MIN_QUEUE_SIMILARITY:
        raise RuntimeError(
            "QUEUE NOT VERIFIED. Publication blocked."
        )

    print("QUEUE 4.1 VERIFIED")


# ==========================================
# 5. VERIFY IMAGE
# ==========================================

def verify_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    print("Image size:", image.size)

    if image.size != IMAGE_SIZE:
        raise RuntimeError(
            f"Unexpected image size: {image.size}"
        )

    header_mask = extract_header_mask(image)

    # Verify queue 4.1
    verify_queue(header_mask)

    # Verify date
    verified_date = recognize_date(
        header_mask
    )

    print(
        "Verified schedule date:",
        verified_date
    )

    return verified_date


# ==========================================
# 6. VALIDATE SCHEDULE
# ==========================================

def validate_schedule(schedule):

    if not isinstance(schedule, dict):
        raise RuntimeError(
            "Invalid schedule format."
        )

    half_hours = schedule.get("halfHours")

    if not isinstance(half_hours, list):
        raise RuntimeError(
            "halfHours is missing."
        )

    if len(half_hours) != 48:
        raise RuntimeError(
            "Expected exactly 48 half-hour slots."
        )

    for value in half_hours:
        if (
            type(value) is not int
            or value not in (0, 1, 2)
        ):
            raise RuntimeError(
                "Invalid half-hour status."
            )

    calculated_minutes = (
        sum(value != 0 for value in half_hours) * 30
    )

    if (
        schedule.get("totalOutageMinutes")
        != calculated_minutes
    ):
        raise RuntimeError(
            "Outage total does not match halfHours."
        )

    print(
        "Schedule structure verified:",
        calculated_minutes,
        "minutes"
    )


# ==========================================
# 7. FIND TELEGRAM MENU
# ==========================================

async def get_latest_menu(client, bot):

    messages = await client.get_messages(
        bot, limit=20
    )

    for msg in messages:
        if not msg.out and msg.buttons:
            return msg

    raise RuntimeError(
        "Bot menu not found."
    )


# ==========================================
# 8. REQUEST FRESH IMAGE
# ==========================================

async def get_fresh_image(client, bot):

    menu = await get_latest_menu(
        client, bot
    )

    buttons = [
        button.text
        for row in menu.buttons
        for button in row
    ]

    # Return to account selection if necessary
    if any("Гаразд" in text for text in buttons):

        print(
            "Returning to account menu...",
            flush=True
        )

        ok = next(
            text for text in buttons
            if "Гаразд" in text
        )

        try:
            await asyncio.wait_for(
                menu.click(text=ok),
                timeout=15
            )
        except asyncio.TimeoutError:
            print(
                "Return button timeout. Checking menu...",
                flush=True
            )

        await asyncio.sleep(3)

        menu = await get_latest_menu(
            client, bot
        )

    # Find correct account
    account = next(
        (
            button.text
            for row in menu.buttons
            for button in row
            if ACCOUNT_TEXT in button.text
        ),
        None
    )

    if account is None:
        raise RuntimeError(
            "Correct account not found."
        )

    previous = await client.get_messages(
        bot, limit=1
    )

    previous_id = (
        previous[0].id if previous else 0
    )

    print(
        "Requesting fresh schedule...",
        flush=True
    )

    # Prevent indefinite waiting for bot response
    try:
        await asyncio.wait_for(
            menu.click(text=account),
            timeout=15
        )

        print(
            "Account button clicked.",
            flush=True
        )

    except asyncio.TimeoutError:
        print(
            "Bot response timeout. Checking messages...",
            flush=True
        )

    await asyncio.sleep(3)

    messages = await client.get_messages(
        bot, limit=20
    )

    images = [
        msg
        for msg in messages
        if (
            not msg.out
            and msg.photo
            and msg.id > previous_id
        )
    ]

    if not images:
        raise RuntimeError(
            "No new schedule image received."
        )

    newest = max(
        images,
        key=lambda msg: msg.id
    )

    print(
        "New image received:",
        newest.id
    )

    image_bytes = await client.download_media(
        newest,
        file=bytes
    )

    if not image_bytes:
        raise RuntimeError(
            "Image download failed."
        )

    return image_bytes


# ==========================================
# 9. MAIN
# ==========================================

async def main():

    session = get_telegram_session()

    client = TelegramClient(
        StringSession(session),
        API_ID,
        API_HASH
    )

    async with client:

        bot = await client.get_entity(BOT)

        print("Telegram connected.")

        # Step 1: obtain fresh official image
        image_bytes = await get_fresh_image(
            client, bot
        )

        # Step 2: verify queue and date
        verified_date = verify_image(
            image_bytes
        )

        # Step 3: analyze 30-minute intervals
        schedule = analyze_image(
            image_bytes
        )

        # Step 4: validate result
        validate_schedule(schedule)

        # Step 5: add verified metadata
        schedule["date"] = verified_date
        schedule["queue"] = EXPECTED_QUEUE
        schedule["source"] = (
            "official-telegram-bot"
        )

        schedule["updatedAt"] = (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        )

        # Step 6: save verified test JSON
        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                schedule,
                file,
                ensure_ascii=False,
                indent=2
            )

        print()
        print("SUCCESS!")
        print("Queue:", schedule["queue"])
        print("Date:", schedule["date"])
        print(
            "Total outage:",
            schedule["totalOutageMinutes"],
            "minutes"
        )
        print(
            "Verified JSON saved:",
            OUTPUT_FILE
        )
        print(
            "Working schedule.json "
            "has not been modified."
        )


# ==========================================
# 10. START
# ==========================================

if __name__ == "__main__":
    asyncio.run(main())
