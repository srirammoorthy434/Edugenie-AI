from ai_client import generate_text
from config import MAX_INPUT_CHARS


def summarize_text(text: str) -> str:
    """
    Summarize educational text using Gemini.

    If Gemini is temporarily unavailable because of quota,
    rate limits, or service availability, ai_client.py
    automatically provides the summary fallback.
    """

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Text is too long. Maximum allowed characters: "
            f"{MAX_INPUT_CHARS}"
        )

    prompt = f"""
Summarize the following educational text.

Requirements:

- Keep the important concepts.
- Remove unnecessary repetition.
- Use simple student-friendly language.
- Use bullet points where helpful.
- Do not introduce facts that are not present in the text.

Text:

{text}
"""

    return generate_text(
        prompt=prompt,
        system_instruction=(
            "You are EduGenie, an educational summarization assistant."
        ),
        task="summary",
        user_input=text
    )