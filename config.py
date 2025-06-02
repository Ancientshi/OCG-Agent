import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            "Export it in your shell or place it in a .env file."
        )
    return value

transformer_cache     = _get_env("TRANSFORMERS_CACHE")
proj_path             = _get_env("PROJ_PATH")
openai_api_key        = _get_env("OPENAI_API_KEY")
silicon_api_key      = _get_env("SILICONFLOW_API_KEY")
serper_search_key     = _get_env("SERPER_SEARCH_KEY")
tmdb_key              = _get_env("TMDB_KEY")
perplexity_api_key    = _get_env("PERPLEXITY_API_KEY")
tiangong_app_key      = _get_env("TIANGONG_APP_KEY")
tiangong_app_secret   = _get_env("TIANGONG_APP_SECRET")
gemini_api_key        = _get_env("GEMINI_API_KEY")
