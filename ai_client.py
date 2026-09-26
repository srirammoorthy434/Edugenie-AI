import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def _demo_response(task: str, user_input: str = "") -> str:
    text = user_input.strip()
    lower_text = text.lower()

    if task == "qa":
        if "largest ocean" in lower_text:
            return (
                "The Pacific Ocean is the largest ocean on Earth. "
                "It covers more area than any other ocean."
            )

        if "moon" in lower_text:
            return (
                "The Moon is Earth's natural satellite. "
                "It revolves around Earth and reflects sunlight."
            )

        if "photosynthesis" in lower_text:
            return (
                "Photosynthesis is the process by which green plants "
                "make food using sunlight, carbon dioxide, and water. "
                "It produces glucose and oxygen."
            )

        return (
            "I’m unable to generate an AI answer right now. "
            "Please try your question again."
        )

    if task == "explain":
        if "photosynthesis" in lower_text:
            return (
                "### Photosynthesis — Simple Explanation\n\n"
                "Photosynthesis is the process by which green plants "
                "make their own food.\n\n"
                "**How it happens:**\n"
                "1. The plant absorbs water through its roots.\n"
                "2. Carbon dioxide enters through the leaves.\n"
                "3. Chlorophyll captures energy from sunlight.\n"
                "4. The plant uses this energy to make glucose.\n"
                "5. Oxygen is released into the air.\n\n"
                "**Simple idea:**\n"
                "Sunlight + water + carbon dioxide → glucose + oxygen."
            )

        return (
            "I’m unable to generate an AI explanation right now. "
            "Please try again."
        )

    if task == "summary":
        if not text:
            return "Please provide some text to summarize."

        clean_text = text
        marker = "Text:"

        if marker in clean_text:
            clean_text = clean_text.split(marker, 1)[1].strip()

        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        selected = sentences[:4]

        if selected:
            summary_lines = ["### Summary", ""]

            for sentence in selected:
                summary_lines.append(f"• {sentence}")

            return "\n".join(summary_lines)

        return (
            "### Summary\n\n"
            f"• {clean_text[:500]}"
        )

    if task == "learning_path":
        topic = text or "the selected topic"

        return (
            f"### Learning Path: {topic}\n\n"
            "**1. Beginner Fundamentals**\n"
            "• Learn the basic definitions and terminology.\n"
            "• Understand the main concepts.\n"
            "• Study simple examples.\n\n"
            "**2. Intermediate Concepts**\n"
            "• Connect the basic concepts together.\n"
            "• Solve practice questions.\n"
            "• Work through small exercises.\n\n"
            "**3. Advanced Concepts**\n"
            "• Study challenging applications.\n"
            "• Solve higher-level problems.\n"
            "• Explore practical use cases.\n\n"
            "**4. Practice Plan**\n"
            "• Practice for 30–60 minutes each day.\n"
            "• Review mistakes after practice.\n"
            "• Take a short test after each topic.\n\n"
            "**5. Final Revision**\n"
            "• Review important concepts.\n"
            "• Revisit difficult areas.\n"
            "• Complete a final practice test."
        )

    return (
        "I’m unable to generate an AI response right now. "
        "Please try again."
    )


def _demo_quiz() -> dict[str, Any]:
    return {
        "questions": [
            {
                "question": (
                    "What is the main source of energy for photosynthesis?"
                ),
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
                "question": (
                    "Which organ pumps blood through the human body?"
                ),
                "options": [
                    "Heart",
                    "Lung",
                    "Kidney",
                    "Stomach"
                ],
                "correct_answer": "Heart",
                "explanation": (
                    "The heart pumps blood throughout the human body."
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


def _call_gemini(
    prompt: str,
    system_instruction: str | None = None,
    schema: dict[str, Any] | None = None
):
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


def _is_temporary_error(error: Exception) -> bool:
    message = str(error).lower()

    temporary_keywords = [
        "quota",
        "resource exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "requests per minute",
        "requests per day",
        "tokens per minute",
        "limit exceeded",
        "503",
        "service unavailable",
        "temporarily unavailable",
        "high demand",
        "unavailable"
    ]

    return any(
        keyword in message
        for keyword in temporary_keywords
    )


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    task: str = "general",
    user_input: str = ""
) -> str:
    try:
        response = _call_gemini(
            prompt=prompt,
            system_instruction=system_instruction
        )

        return response.text or ""

    except Exception as exc:
        if _is_temporary_error(exc):
            return _demo_response(
                task=task,
                user_input=user_input or prompt
            )

        return f"AI Error: {exc}"


def generate(
    prompt: str,
    task: str = "general",
    user_input: str | None = None,
    system_instruction: str | None = None
) -> str:
    try:
        response = _call_gemini(
            prompt=prompt,
            system_instruction=system_instruction
        )

        return response.text or ""

    except Exception as exc:
        if _is_temporary_error(exc):
            return _demo_response(
                task=task,
                user_input=user_input or prompt
            )

        return f"AI Error: {exc}"


def generate_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    system_instruction: str | None = None
) -> dict[str, Any]:
    try:
        response = _call_gemini(
            prompt=prompt,
            system_instruction=system_instruction,
            schema=schema
        )

        raw_text = response.text or ""

        if not raw_text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

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
        if _is_temporary_error(exc):
            return _demo_quiz()

        return {"error": str(exc)}