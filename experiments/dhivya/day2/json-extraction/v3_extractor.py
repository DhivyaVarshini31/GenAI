"""Extract structured article data using Gemini and Pydantic.

This script reads the article text from article.txt, sends a focused extraction
prompt to Gemini using response_schema=Article, and writes the parsed JSON
result to v3_output.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

# Read API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=api_key)


# -----------------------------
# Pydantic Models
# -----------------------------

class NamedEntity(BaseModel):
    name: str
    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        allowed_types = {"PERSON", "ORG", "LOCATION", "EVENT"}

        if value not in allowed_types:
            raise ValueError(
                f"NamedEntity.type must be one of {sorted(allowed_types)}, got {value!r}"
            )

        return value


class Article(BaseModel):
    title: str
    author: str
    publication_date: str
    key_facts: list[str] = Field(..., max_length=7)
    named_entities: list[NamedEntity]


# -----------------------------
# Helper Functions
# -----------------------------

def load_article_text(article_path: Path) -> str:
    """Read article text from article.txt"""
    return article_path.read_text(encoding="utf-8")


def strip_code_fence(text: str) -> str:
    """Remove markdown code fences like ```json ... ```"""
    return re.sub(r"(?s)```(?:json)?\s*(.*?)\s*```", r"\1", text).strip()


def save_json(data: dict, output_path: Path) -> None:
    """Save JSON output"""
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# -----------------------------
# Main Program
# -----------------------------

def main() -> None:

    current_dir = Path(__file__).resolve().parent

    article_path = current_dir / "article.txt"
    output_path = current_dir / "v3_output.json"

    # -----------------------------
    # Load article
    # -----------------------------

    article_text = load_article_text(article_path)

    # Edge case: empty or tiny article
    if len(article_text.strip()) < 200:
        print(
            "Error: article text is empty or too short for reliable extraction "
            "(must be at least 200 characters)."
        )
        sys.exit(1)

    # -----------------------------
    # Improved Prompt
    # -----------------------------

    prompt_text = f"""
You are an information extraction system.

Extract structured information from the news article below.

IMPORTANT RULES:
- Return ONLY valid JSON
- Do not include markdown
- Do not wrap output in ```json fences
- Normalize dates to YYYY-MM-DD format whenever possible
- key_facts must be specific, detailed, and information-dense
- Avoid vague summaries
- Include meaningful names, numbers, and events
- Deduplicate repeated entities with different spellings
- Use only PERSON, ORG, LOCATION, or EVENT for entity types
- If information is missing, use "Unknown"

Return JSON matching the schema exactly.

ARTICLE:
{article_text}
"""

    # -----------------------------
    # Gemini API Call
    # -----------------------------

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text,
            config={
                "response_mime_type": "application/json",
                "response_schema": Article,
            },
        )

    except Exception as exc:

        status_code = None

        if hasattr(exc, "status_code"):
            status_code = getattr(exc, "status_code")

        elif hasattr(exc, "response"):
            status_code = getattr(exc.response, "status_code", None)

        # 429 Rate Limit
        if status_code == 429 or "429" in str(exc):
            print(
                "Rate limited — switch to gemini-2.5-flash-lite "
                "or wait until midnight Pacific Time."
            )
            return

        # 503 Server Overload
        if (
            status_code == 503
            or "503" in str(exc)
            or "UNAVAILABLE" in str(exc)
        ):
            print(
                "Gemini servers are overloaded right now. "
                "Try again in a few minutes."
            )
            return

        print("Unexpected API error:")
        print(exc)
        return

    # -----------------------------
    # Clean JSON Text
    # -----------------------------

    raw_text = response.text

    cleaned_text = strip_code_fence(raw_text)

    # -----------------------------
    # Parse JSON Safely
    # -----------------------------

    try:
        parsed = json.loads(cleaned_text)

    except json.JSONDecodeError:

        print("Failed to decode JSON from Gemini response.")
        print("\nRaw response text:\n")
        print(raw_text)

        return

    # -----------------------------
    # Save Output
    # -----------------------------

    pretty_output = json.dumps(
        parsed,
        indent=2,
        ensure_ascii=False,
    )

    print(pretty_output)

    save_json(parsed, output_path)

    print(f"\nSaved output to: {output_path}")


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    
    main()