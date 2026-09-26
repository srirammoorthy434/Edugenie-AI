from ai_client import generate


def answer_question(question: str) -> str:
    """
    Answers a user's question using Gemini.
    If Gemini quota is unavailable, ai_client automatically
    provides a demo fallback response.
    """

    question = question.strip()

    if not question:
        return "Please enter a question."

    prompt = f"""
You are EduGenie, a friendly educational AI assistant.

Answer the student's question clearly and concisely.

Question:
{question}

Give a simple, accurate answer suitable for a student.
"""

    return generate(
        prompt,
        task="qa",
        user_input=question,
    )