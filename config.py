import os

from dotenv import load_dotenv


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Change this if your Google AI Studio account provides another model.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

# Optional local explanation model.
# Keep this false initially because it requires a large download.
LOCAL_EXPLAINER_ENABLED = (
    os.getenv("LOCAL_EXPLAINER_ENABLED", "false").lower() == "true"
)

LOCAL_EXPLAINER_MODEL = os.getenv(
    "LOCAL_EXPLAINER_MODEL",
    "MBZUAI/LaMini-Flan-T5-783M"
)

MAX_INPUT_CHARS = int(
    os.getenv("MAX_INPUT_CHARS", "12000")
)