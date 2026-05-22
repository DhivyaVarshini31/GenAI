"""Compare two Gemini prompts using the same input text."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Read API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Model name
MODEL_NAME = "gemini-2.5-flash-lite"

# Max response length
MAX_RESPONSE_LENGTH = 1500


def _format_prompt(prompt_template: str, text: str) -> str:
    """Replace {text} placeholder in prompt template."""
    return prompt_template.replace("{text}", text)


def _truncate_response(response: str) -> str:
    """Truncate long responses."""
    if len(response) <= MAX_RESPONSE_LENGTH:
        return response

    return response[:MAX_RESPONSE_LENGTH] + "... (truncated at 1500 chars)"


def _gemini_response(prompt: str) -> tuple[str, float]:
    """Send prompt to Gemini and return response + elapsed time."""

    start = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    elapsed = time.perf_counter() - start

    return response.text, elapsed


def compare(
    text: str,
    prompt_a: str,
    prompt_b: str,
) -> dict[str, dict[str, object]]:
    """
    Compare two prompts using the same input text.
    """

    formatted_a = _format_prompt(prompt_a, text)
    formatted_b = _format_prompt(prompt_b, text)

    response_a, elapsed_a = _gemini_response(formatted_a)
    response_b, elapsed_b = _gemini_response(formatted_b)

    return {
        "prompt_a": {
            "prompt": formatted_a,
            "response": response_a,
            "elapsed": elapsed_a,
        },
        "prompt_b": {
            "prompt": formatted_b,
            "response": response_b,
            "elapsed": elapsed_b,
        },
    }


def _read_text_file(path: Path) -> str:
    """Read text file content."""
    return path.read_text(encoding="utf-8")


def _render_markdown(
    results: dict[str, dict[str, object]],
) -> str:
    """Render comparison results as markdown."""

    response_a = _truncate_response(
        results["prompt_a"]["response"]
    )

    response_b = _truncate_response(
        results["prompt_b"]["response"]
    )

    elapsed_a = results["prompt_a"]["elapsed"]
    elapsed_b = results["prompt_b"]["elapsed"]

    return (
        "# Prompt Comparison Results\n\n"
        f"## Prompt A ({elapsed_a:.3f}s)\n\n"
        f"{response_a}\n\n"
        "---\n\n"
        f"## Prompt B ({elapsed_b:.3f}s)\n\n"
        f"{response_b}\n"
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Compare two Gemini prompts using the same input text."
    )

    parser.add_argument(
        "--text-file",
        required=True,
        type=Path,
        help="Path to the input text file.",
    )

    parser.add_argument(
        "--prompt-a",
        required=True,
        type=Path,
        help="Path to prompt A template file.",
    )

    parser.add_argument(
        "--prompt-b",
        required=True,
        type=Path,
        help="Path to prompt B template file.",
    )

    parser.add_argument(
        "--output",
        default="comparison_result.md",
        type=Path,
        help="Path to save markdown output.",
    )

    args = parser.parse_args()

    # Read files
    text = _read_text_file(args.text_file)
    prompt_a = _read_text_file(args.prompt_a)
    prompt_b = _read_text_file(args.prompt_b)

    # Compare prompts
    results = compare(text, prompt_a, prompt_b)

    # Generate markdown
    output_markdown = _render_markdown(results)

    # Save markdown file
    args.output.write_text(
        output_markdown,
        encoding="utf-8",
    )

    print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
