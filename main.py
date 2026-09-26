from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from explanation_module import explain_topic
from learning_path import generate_learning_path
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_text


BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=1)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "application": "EduGenie"
    }


@app.post("/qa")
async def qa(request: QARequest):

    try:

        answer = answer_question(
            request.question
        )

        return {
            "success": True,
            "answer": answer
        }

    except ValueError as exc:

        return {
            "success": False,
            "error": str(exc)
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }


@app.post("/explain")
async def explain(request: TopicRequest):

    try:

        explanation = explain_topic(
            request.topic
        )

        return {
            "success": True,
            "explanation": explanation
        }

    except ValueError as exc:

        return {
            "success": False,
            "error": str(exc)
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }


@app.post("/quiz")
async def quiz(request: TopicRequest):

    try:

        quiz = generate_quiz(
            request.topic
        )

        return {
            "success": True,
            "quiz": quiz
        }

    except ValueError as exc:

        return {
            "success": False,
            "error": str(exc)
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }


@app.post("/summarize")
async def summarize(request: TextRequest):

    try:

        summary = summarize_text(
            request.text
        )

        return {
            "success": True,
            "summary": summary
        }

    except ValueError as exc:

        return {
            "success": False,
            "error": str(exc)
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }


@app.post("/learn/recommendations")
async def learning_recommendations(
    request: TopicRequest
):

    try:

        learning_path = generate_learning_path(
            request.topic
        )

        return {
            "success": True,
            "recommendations": learning_path
        }

    except ValueError as exc:

        return {
            "success": False,
            "error": str(exc)
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc)
        }