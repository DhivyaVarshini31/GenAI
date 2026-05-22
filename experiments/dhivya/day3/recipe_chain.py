# python recipe_chain.py --input "your recipe text here"
from __future__ import annotations

import argparse
import os
import time
from dotenv import load_dotenv
from google.genai import Client
from google.genai.errors import ServerError

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = Client(api_key=GEMINI_API_KEY)


def call_gemini(prompt: str) -> str:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            return response.text or ""
        except ServerError as error:
            if attempt == max_attempts:
                raise
            print(
                f"Server unavailable (attempt {attempt}/{max_attempts}). Retrying..."
            )
            time.sleep(2 ** attempt)
    return ""


# STEP 1: Extract ingredients from the recipe as JSON
step1_prompt = (
    "Extract the ingredients from this recipe as a JSON list with name and quantity. "
    "Do not add any extra text."
)

# STEP 2: Convert the ingredient list into a grocery shopping list by aisle
step2_prompt = (
    "Given these ingredients, group them by grocery aisle (produce, dairy, pantry, spices, other) "
    "and format the result as a shopping list."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a two-step Gemini recipe chain from the command line."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Recipe text to process",
    )
    args = parser.parse_args()

    recipe_text = args.input

    step1_input = f"Recipe:\n{recipe_text}\n\n{step1_prompt}"
    step1_result = call_gemini(step1_input)
    print("[Step 1 result]", step1_result)

    step2_input = f"Ingredients:\n{step1_result}\n\n{step2_prompt}"
    final_result = call_gemini(step2_input)
    print("[Final result]", final_result)


if __name__ == "__main__":
    main()
