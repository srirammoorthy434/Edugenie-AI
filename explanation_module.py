from ai_client import generate_text
from config import (
    LOCAL_EXPLAINER_ENABLED,
    LOCAL_EXPLAINER_MODEL,
    MAX_INPUT_CHARS
)


LOCAL_PIPELINE = None


def _load_local_model():
    global LOCAL_PIPELINE

    if LOCAL_PIPELINE is not None:
        return LOCAL_PIPELINE

    try:
        from transformers import pipeline

        LOCAL_PIPELINE = pipeline(
            "text2text-generation",
            model=LOCAL_EXPLAINER_MODEL
        )

        return LOCAL_PIPELINE

    except Exception as exc:
        raise RuntimeError(
            "Could not load the local explanation model."
        ) from exc


def explain_topic(topic: str) -> str:

    topic = topic.strip()

    if not topic:
        raise ValueError("Topic cannot be empty.")

    if len(topic) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Topic is too long. Maximum allowed characters: "
            f"{MAX_INPUT_CHARS}"
        )

    prompt = f"""
Explain the following educational topic to a student:

{topic}

Use this structure:

1. Simple definition
2. Main idea
3. Step-by-step explanation
4. Simple example
5. Important points to remember

Use simple language.
"""

    # Optional local model.
    if LOCAL_EXPLAINER_ENABLED:

        try:
            model = _load_local_model()

            result = model(
                prompt,
                max_new_tokens=400,
                do_sample=False
            )

            if result:
                return result[0]["generated_text"]

        except Exception:
            # If local model fails, use Gemini instead.
            pass

    return generate_text(
        prompt=prompt,
        system_instruction=(
            "You are EduGenie, an expert educational tutor. "
            "Explain concepts simply and accurately."
        )
    )