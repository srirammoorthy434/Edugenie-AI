from typing import Any

from ai_client import generate_json
from config import MAX_INPUT_CHARS


QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "options": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "string"
                        }
                    },
                    "correct_answer": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    }
                },
                "required": [
                    "question",
                    "options",
                    "correct_answer",
                    "explanation"
                ]
            }
        }
    },
    "required": ["questions"]
}


def generate_quiz(topic: str) -> dict[str, Any]:

    topic = topic.strip()

    if not topic:
        raise ValueError("Quiz topic cannot be empty.")

    if len(topic) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Quiz topic is too long. Maximum allowed characters: "
            f"{MAX_INPUT_CHARS}"
        )

    prompt = f"""
Create exactly 3 multiple-choice questions about:

{topic}

Requirements:

- Exactly 3 questions.
- Exactly 4 options per question.
- Only one correct answer.
- The correct answer must exactly match one of the options.
- Questions should test understanding, not just random trivia.
- Include a short explanation for every correct answer.
- Return only the requested JSON structure.
"""

    result = generate_json(
        prompt=prompt,
        schema=QUIZ_SCHEMA,
        system_instruction=(
            "You are EduGenie's quiz generator. "
            "Generate accurate educational MCQs."
        )
    )

    questions = result.get("questions")

    if not isinstance(questions, list):
        raise RuntimeError(
            "Invalid quiz response."
        )

    if len(questions) != 3:
        raise RuntimeError(
            "Quiz generator did not return exactly 3 questions."
        )

    for question in questions:

        options = question.get("options", [])

        if len(options) != 4:
            raise RuntimeError(
                "Every quiz question must have exactly 4 options."
            )

        correct_answer = question.get(
            "correct_answer",
            ""
        )

        if correct_answer not in options:
            raise RuntimeError(
                "Correct answer must match one of the options."
            )

    return result