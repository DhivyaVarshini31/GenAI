import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PREFIX = "Explain your reasoning step-by-step. Be thorough. "

PROMPTS = [
    {
        "label": "Prompt 1",
        "base": "INSERT EXACT PROMPT TEXT FROM BLOCK 2 HERE (factual prompt)",
    },
    {
        "label": "Prompt 2",
        "base": "INSERT EXACT PROMPT TEXT FROM BLOCK 2 HERE (creative prompt)",
    },
    {
        "label": "Prompt 3",
        "base": "INSERT EXACT PROMPT TEXT FROM BLOCK 2 HERE (summarization prompt)",
    },
]


def generate_response(prompt_text: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text,
    )
    return response.text


def main() -> None:
    for prompt_data in PROMPTS:
        base_prompt = prompt_data["base"]
        prompt_label = prompt_data["label"]
        for variation_label, prompt_text in [("A", base_prompt), ("B", PREFIX + base_prompt)]:
            label = f"[{prompt_label} - Variation {variation_label}]"
            print(label)
            print("---")
            print(generate_response(prompt_text))
            print()


if __name__ == "__main__":
    main()
