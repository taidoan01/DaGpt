from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
def load_llm(model_name):
    """
    Load Large Language Model
    """
    if model_name == "gpt-3.5-turbo":
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
        )
    elif model_name == "gpt-4":
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
        )
    elif model_name=="gemini-1.5-pro":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
        )  
    else:
        raise ValueError(
            "Unkown model.\
                Please choose from ['gemini-1.5-pro',gpt-3.5-turbo','gpt-4',...]"
        )