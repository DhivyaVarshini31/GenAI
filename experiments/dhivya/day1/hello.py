"""Simple script that reads GEMINI_API_KEY from a .env file,
configures the google-generativeai SDK, and prints a Gemini response.
"""

from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Configure Gemini
genai.configure(api_key=api_key)

# Create model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Generate response
response = model.generate_content("Say hello in one sentence.")

# Print output
print("Gemini says:", response.text)