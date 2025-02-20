import os
from typing import Optional
from openai import OpenAI
from swarm import Swarm  # type: ignore[import]


def llm_client() -> Swarm:
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    return Swarm(client=client)


def handle_base_model_arg(model: Optional[str]) -> str:
    if model is None:
        return os.getenv("OPENAI_BASE_MODEL", "gpt-4o")
    return model
