"""
SOYUT TABAN AJAN SINIFI (agents/base_agent.py)
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from logger import logger

load_dotenv(override=True)


class BaseAgent(ABC):
    def __init__(
        self,
        model_env_var: Optional[str] = None,
        default_model: str = "gemma4:31b-cloud",
        temperature: float = 0.0,
        format_schema: Optional[dict] = None,
        agent_role: str = "agent",
    ):
        if model_env_var and os.getenv(model_env_var):
            model_name = os.getenv(model_env_var)
        else:
            model_name = os.getenv("OLLAMA_MODEL", default_model)

        raw_host = os.getenv("OLLAMA_HOST") or "https://ollama.com"
        raw_host = raw_host.replace("0.0.0.0","127.0.0.1").rstrip("/")

        if not raw_host.startswith(("http://", "https://")):
            raw_host = f"http://{raw_host}"

        ollama_host = raw_host
        ollama_key = os.getenv("OLLAMA_API_KEY")

        client_kwargs = {"trust_env": False}
        if ollama_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_key}"}

        self.model_name = model_name
        self.ollama_host = ollama_host

        logger.info(f"Ollama -> {ollama_host} | {agent_role} model={model_name}")

        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "base_url": ollama_host,
            "client_kwargs": client_kwargs,
        }
        if format_schema is not None:
            llm_kwargs["format"] = format_schema

        self.llm = ChatOllama(**llm_kwargs)

    def invoke_with_retry(
        self,
        messages: List[Tuple[str, str]],
        response_processor: Optional[Callable[[str], Any]] = None,
        max_retries: int = 3,
    ) -> Any:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_content = self.llm.invoke(messages).content
                if response_processor is not None:
                    return response_processor(raw_content)
                return raw_content
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = attempt * 3
                    logger.warning(
                        f"[{self.__class__.__name__}] {type(e).__name__}: {str(e)[:200]} "
                        f"-> {wait} sn sonra tekrar ({attempt}/{max_retries})"
                    )
                    time.sleep(wait)
        raise RuntimeError(f"{max_retries} deneme sonunda başarısız: {str(last_err)[:300]}")

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass