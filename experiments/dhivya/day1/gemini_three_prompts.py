import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPTS = [
    {
        "label": "Prompt 1 (Factual)",
        "text": "What's the capital of Australia? Answer in one sentence."
    },
    {
        "label": "Prompt 2 (Creative)",
        "text": "Write a 4-line poem about debugging code."
    },
    {
        "label": "Prompt 3 (Summarization)",
        "text": (
            "Summarize this in exactly 2 sentences:\n\n"
            "The Internet has changed nearly every aspect of modern life, from communication and entertainment to business, education, and healthcare. "
            "Originally a research project in the 1960s, it became commercially available in the 1990s and now connects over five billion people. "
            "Today it underpins cloud computing, social media, e-commerce, and AI."
        )
    }
]


def get_response_text(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def main() -> None:
    for prompt_data in PROMPTS:
        print("---")
        print(prompt_data["label"])
        print("Prompt:")
        print(prompt_data["text"])
        print()
        response_text = get_response_text(prompt_data["text"])
        print("Response:")
        print(response_text)
        print()


if __name__ == "__main__":
    main()
