import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

load_dotenv()

def pickllm(level: str):
    level = level.lower()
    if level not in ("low", "medium", "high"):
        raise ValueError(f"Unsupported level: {level}")

    provider = os.getenv("LLM_PROVIDER", "")

    # 1. Google Gemini
    if provider == "Google Gemini" or (not provider and os.getenv("GOOGLE_API_KEY")):
        model_map = {
            "low": "gemini-3.8-flash",
            "medium": "gemini-3.8-flash",
            "high": "gemini-3.8-flash",
        }
        return ChatGoogleGenerativeAI(
            model=model_map[level],
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # 2. Mistral
    if provider == "Mistral" or (not provider and os.getenv("MISTRAL_API_KEY")):
        model_map = {
            "low": "mistral-small-latest",
            "medium": "mistral-small-latest",
            "high": "mistral-large-latest",
        }
        return ChatMistralAI(
            model=model_map[level],
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )

    # 3. Default: Groq
    model_map = {
        "low": "openai/gpt-oss-20b",
        "medium": "qwen/qwen3.8-27b",
        "high": "openai/gpt-oss-120b",
    }
    return ChatGroq(
        model=model_map[level],
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
