from ai_client import generate_text
from config import MAX_INPUT_CHARS


def generate_learning_path(topic: str) -> str:

    topic = topic.strip()

    if not topic:
        raise ValueError("Learning topic cannot be empty.")

    if len(topic) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Topic is too long. Maximum allowed characters: "
            f"{MAX_INPUT_CHARS}"
        )

    prompt = f"""
Create a personalized learning path for:

{topic}

Organize it from beginner to advanced.

Include:

1. Beginner fundamentals
2. Intermediate concepts
3. Advanced concepts
4. Suggested timeline
5. Practice activities
6. Project ideas
7. Recommended resource types
8. A final revision stage

Make the plan realistic for a student.

Use headings and bullet points.
"""

    return generate_text(
        prompt=prompt,
        system_instruction=(
            "You are EduGenie, a personalized educational "
            "learning-path advisor."
        )
    )