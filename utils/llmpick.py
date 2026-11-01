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
    if provider == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0,
            max_retries=5,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # 2. Mistral
    if provider == "Mistral":
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

    # 3. Groq (Fast & free)
    model_map = {
        "low": "openai/gpt-oss-20b",
        "medium": "openai/gpt-oss-20b",
        "high": "openai/gpt-oss-120b",
    }
    return ChatGroq(
        model=model_map[level],
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
