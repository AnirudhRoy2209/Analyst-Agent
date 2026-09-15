from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def pickllm(level:str):
    """
    pick the appropriate LLM based on the level of the question.
    Args: level(str): level of the question can be "low", "medium", "high".
    returns: google gemini LLM instance to be used.
    """

    if level.lower()=="low":
        llm= ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    elif level.lower()=="medium":
        llm= ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    elif level.lower()=="high":
        llm= ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    else:
        raise ValueError(f"unsupported level {level}")
    return llm
