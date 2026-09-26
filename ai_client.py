import json
import os
from typing import Any

from google import genai
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv(override=True)


API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ---------------------------------------------------------
# Demo fallback responses
# ---------------------------------------------------------

def _demo_response(
    task: str,
    user_input: str = ""
) -> str:

    text = user_input.strip().lower()

    if task == "qa":

        if "largest ocean" in text:
            return (
                "The Pacific Ocean is the largest ocean on Earth. "
                "It covers more area than any other ocean."
            )

        if "moon" in text:
            return (
                "The Moon is Earth's natural satellite. "
                "It revolves around Earth and reflects sunlight."
            )

        return (
            "Gemini is temporarily unavailable because of an API "
            "quota or rate limit. EduGenie is currently using its "
            "demo fallback response."
        )

    if task == "explain":

        if "photosynthesis" in text:
            return (
                "Photosynthesis is the process by which green plants "
                "make their own food using sunlight, carbon dioxide, "
                "and water.\n\n"
                "Simple idea:\n"
                "Sunlight + water + carbon dioxide → glucose + oxygen.\n\n"
                "The process mainly takes place in the chloroplasts "
                "of plant cells."
            )

        return (
            "Gemini is temporarily unavailable. "
            "EduGenie can continue running using its demo mode."
        )

    if task == "summary":

        return (
            "• The text contains important educational information.\n"
            "• The main concepts should be identified and reviewed.\n"
            "• Repeated or unnecessary information can be removed.\n"
            "• Review the key points again for better understanding.\n\n"
            "(Demo fallback response: Gemini quota is temporarily unavailable.)"
        )

    if task == "learning_path":

        return (
            "Learning Path\n\n"
            "1. Beginner fundamentals\n"
            "• Learn the basic definitions and concepts.\n"
            "• Study the essential terminology.\n\n"
            "2. Intermediate concepts\n"
            "• Connect the basic concepts.\n"
            "• Solve practice questions.\n\n"
            "3. Advanced concepts\n"
            "• Study advanced applications.\n"
            "• Work on challenging problems.\n\n"
            "4. Practice\n"
            "• Complete exercises and small projects.\n\n"
            "5. Revision\n"
            "• Review important concepts.\n"
            "• Take a final test.\n\n"
            "(Demo fallback response: Gemini quota is temporarily unavailable.)"
        )

    return (
        "Gemini is temporarily unavailable because of an API "
        "quota or rate limit. EduGenie is using its demo fallback mode."
    )


def _demo_quiz() -> dict[str, Any]:

    return {
        "questions": [
            {
                "question": "What is the main source of energy for photosynthesis?",
                "options": [
                    "Sunlight",
                    "Moonlight",
                    "Sound",
                    "Wind"
                ],
                "correct_answer": "Sunlight",
                "explanation": (
                    "Plants use energy from sunlight to carry out "
                    "photosynthesis."
                )
            },
            {
                "question": "Which organ pumps blood through the human body?",
                "options": [
                    "Heart",
                    "Lung",
                    "Kidney",
                    "Stomach"
                ],
                "correct_answer": "Heart",
                "explanation": (
                    "The heart pumps blood throughout the body."
                )
            },
            {
                "question": "What is H₂O commonly known as?",
                "options": [
                    "Water",
                    "Oxygen",
                    "Hydrogen",
                    "Carbon dioxide"
                ],
                "correct_answer": "Water",
                "explanation": (
                    "H₂O is the chemical formula for water."
                )
            }
        ]
    }


# ---------------------------------------------------------
# Gemini request
# ---------------------------------------------------------

def _call_gemini(
    prompt: str,
    system_instruction: str | None = None,
    schema: dict[str, Any] | None = None
):
    """Send a request to Gemini."""

    if client is None:
        raise RuntimeError("Gemini API key is not configured.")

    config_kwargs: dict[str, Any] = {}

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    if schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config_kwargs if config_kwargs else None
    )

    return response


# ---------------------------------------------------------
# Error detection
# ---------------------------------------------------------

def _is_quota_error(error: Exception) -> bool:

    message = str(error).lower()

    quota_keywords = [
        "quota",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "requests per minute",
        "requests per day",
        "tokens per minute",
        "limit exceeded"
    ]

    return any(keyword in message for keyword in quota_keywords)


# ---------------------------------------------------------
# Normal text generation
# ---------------------------------------------------------

def generate_text(
    prompt: str,
    system_instruction: str | None = None
) -> str:
    """Generate normal text using Gemini."""

    try:

        response = _call_gemini(
            prompt=prompt,
            system_instruction=system_instruction
        )

        return response.text or ""

    except Exception as exc:

        if _is_quota_error(exc):
            return _demo_response(
                task="general",
                user_input=prompt
            )

        return f"AI Error: {exc}"


# ---------------------------------------------------------
# Compatibility generate function
# ---------------------------------------------------------

def generate(
    prompt: str,
    task: str = "general",
    user_input: str | None = None,
    system_instruction: str | None = None
) -> str:
    """
    Compatibility function used by the different EduGenie modules.
    """

    try:

        response = _call_gemini(
            prompt=prompt,
            system_instruction=system_instruction
        )

        return response.text or ""

    except Exception as exc:

        if _is_quota_error(exc):

            return _demo_response(
                task=task,
                user_input=user_input or prompt
            )

        return f"AI Error: {exc}"


# ---------------------------------------------------------
# JSON generation
# ---------------------------------------------------------

def generate_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    system_instruction: str | None = None
) -> dict[str, Any]:
    """
    Generate structured JSON-compatible output using Gemini.
    """

    try:

        response = _call_gemini(
            prompt=prompt,
            system_instruction=system_instruction,
            schema=schema
        )

        raw_text = response.text or ""

        if not raw_text:
            raise RuntimeError("Gemini returned an empty response.")

        try:
            result = json.loads(raw_text)

        except json.JSONDecodeError:

            cleaned = raw_text.strip()

            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]

            if cleaned.startswith("```"):
                cleaned = cleaned[3:]

            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            result = json.loads(cleaned.strip())

        if not isinstance(result, dict):
            raise RuntimeError(
                "Gemini returned JSON in an unexpected format."
            )

        return result

    except Exception as exc:

        if _is_quota_error(exc):
            return _demo_quiz()

        return {
            "error": str(exc)
        }