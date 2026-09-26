import os
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def generate_text(prompt: str) -> str:
    """Generate normal text using Gemini."""

    if client is None:
        return "Gemini API key is not configured."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text or ""

    except Exception as e:
        return f"AI Error: {e}"


def generate(prompt: str) -> str:
    """Compatibility function."""
    return generate_text(prompt)


def generate_json(prompt: str):
    """Generate JSON-compatible output using Gemini."""

    if client is None:
        return {"error": "Gemini API key is not configured."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text or ""

    except Exception as e:
        return {"error": str(e)}