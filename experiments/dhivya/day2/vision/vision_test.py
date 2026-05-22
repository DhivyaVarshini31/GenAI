from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import Client, types
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

IMAGE_PROMPTS = {
    "Prompt 1": "Describe what you see in this image.",
    "Prompt 2": "Extract any text visible in this image.",
    "Prompt 3": "What are the 3 most important elements in this image?",
}

OUTPUT_FILE = Path(__file__).resolve().parent / "vision_output.txt"


def get_image_path() -> Path:
    folder = Path(__file__).resolve().parent
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    candidates: list[Path] = []

    # Prefer standard names first.
    for name in ("test_image.jpg", "test_image.png", "test_image.jpeg"):
        path = folder / name
        if path.exists():
            candidates.append(path)
            if path.stat().st_size > 0:
                return path

    # Then accept any test_image.* file extension.
    for path in folder.glob("test_image.*"):
        if path.exists() and path.suffix.lower() in valid_extensions:
            candidates.append(path)
            if path.stat().st_size > 0:
                return path

    if not candidates:
        raise FileNotFoundError(
            "No local image found. Place a valid test_image.jpg or test_image.png in the same folder."
        )

    bad_files = ", ".join(str(path.name) for path in candidates)
    raise ValueError(
        f"Found image file(s) {bad_files}, but none contain valid image data. "
        "Replace them with a non-empty image file."
    )


def get_image_bytes(image_path: Path) -> bytes:
    data = image_path.read_bytes()
    if not data:
        raise ValueError(
            f"Image file '{image_path.name}' is empty. Replace it with a valid image file."
        )
    return data


def generate_response(client: Client, prompt: str, image_bytes: bytes, mime_type: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return response.text or ""


def main() -> None:
    client = Client(api_key=GEMINI_API_KEY)

    image_path = get_image_path()
    image_bytes = get_image_bytes(image_path)
    mime_type = "image/jpeg" if image_path.suffix.lower() == ".jpg" else "image/png"

    results: list[str] = []

    for label, prompt in IMAGE_PROMPTS.items():
        response_text = generate_response(client, prompt, image_bytes, mime_type)
        output = f"{label}: {prompt}\n{response_text.strip()}\n"
        print(output)
        results.append(output)

    OUTPUT_FILE.write_text("\n".join(results), encoding="utf-8")


if __name__ == "__main__":
    main()
