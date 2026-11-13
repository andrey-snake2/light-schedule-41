import asyncio
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from telethon import TelegramClient
from telethon.sessions import StringSession

from analyze_schedule import analyze_image
from date_verification import recognize_date


BOT = "pat_voe_bot"
ACCOUNT_TEXT = "Немирівське шосе"
IMAGE_SIZE = (540, 714)
HEADER_BOX = (20, 35, 520, 110)
QUEUE_BOX = (0, 0, 210, 75)
REFERENCE_FILE = "queue_reference.png"
MIN_QUEUE_SIMILARITY = 0.98
OUTPUT_FILE = Path("schedule.json")


def load_telegram_session():
    session = os.getenv("TELEGRAM_SESSION", "").strip()
    if session:
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


async def get_latest_menu(client, bot):
    messages = await client.get_messages(bot, limit=20)

    for msg in messages:
        if not msg.out and msg.buttons:
            return msg

    raise RuntimeError("Bot menu not found")


async def main():
    load_dotenv()
    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    session = load_telegram_session()

    client = TelegramClient(
        StringSession(session),
        api_id,
        api_hash
    )

    async with client:
        bot = await client.get_entity(BOT)
        menu = await get_latest_menu(client, bot)

        # Return to account selection if necessary
        buttons = [
            button.text
            for row in menu.buttons
            for button in row
        ]

        if any("Гаразд" in text for text in buttons):
            print("Returning to account menu...")

            ok = next(
                text for text in buttons
                if "Гаразд" in text
            )

            await menu.click(text=ok)
            await asyncio.sleep(3)
            menu = await get_latest_menu(client, bot)

        # Find the correct account
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
            raise RuntimeError("Correct account not found")

        previous = await client.get_messages(bot, limit=1)
        previous_id = previous[0].id

        print("Requesting fresh schedule...")

        await menu.click(text=account)
        await asyncio.sleep(5)

        messages = await client.get_messages(bot, limit=15)

        images = [
            msg for msg in messages
            if (
                not msg.out
                and msg.photo
                and msg.id > previous_id
            )
        ]

        if not images:
            raise RuntimeError("No new schedule image received")

        newest = max(images, key=lambda msg: msg.id)

        print("New image received:", newest.id)

        image_bytes = await client.download_media(
            newest,
            file=bytes
        )

        if not image_bytes:
            raise RuntimeError("Image download failed.")

        save_verified_schedule(image_bytes)


def save_verified_schedule(image_bytes):
    verified_date = verify_image(image_bytes)
    schedule = analyze_image(image_bytes)
    schedule["date"] = verified_date

    temp_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=OUTPUT_FILE.parent,
            prefix=".schedule-", suffix=".tmp", delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(schedule, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp_path.replace(OUTPUT_FILE)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print("Schedule saved:", OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
