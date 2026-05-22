"""Extract structured article data using Gemini and Pydantic.

This script reads the article text from article.txt, sends a simple extraction
prompt to Gemini using response_schema=Article, and writes the parsed JSON
result to v1_output.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, validator

# Load environment variables from .env in the current directory.
load_dotenv()

# Do not hardcode the API key; read it from .env instead.
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create a Gemini client using the free-tier compatible model.
client = genai.Client(api_key=api_key)


class NamedEntity(BaseModel):
    name: str
    type: str

    @validator("type")
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


def load_article_text(article_path: Path) -> str:
    """Read the article text from article.txt in the current folder."""
    return article_path.read_text(encoding="utf-8")


def save_json(data: dict, output_path: Path) -> None:
    """Save parsed JSON to a UTF-8 file with indentation."""
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    article_path = Path(r"C:\Users\Divya Vijayaprakash\OneDrive\Documents\New folder\modex-genai-training\experiments\dhivya\day2\json-extraction\article.txt")
    output_path = current_dir / "v1_output.json"

    article_text = load_article_text(article_path)

    prompt_text = "Extract structured information from this news article."
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{prompt_text}\n\n{article_text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": Article,
        },
    )

    parsed = json.loads(response.text)

    # Print to stdout and save the same structured JSON.
    pretty_output = json.dumps(parsed, indent=2, ensure_ascii=False)
    print(pretty_output)
    save_json(parsed, output_path)


if __name__ == "__main__":
    main()
