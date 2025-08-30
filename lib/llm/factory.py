from typing import Any, Dict, List
from .base import LLMClient
from .ollama_client import OllamaClient


def create_llm_client(
    cfg: Dict[str, Any] | None,
    provider_override: str | None,
    model_override: str | None,
    temperature: float,
    max_tokens: int,
) -> LLMClient:
    cfg = cfg or {}
    providers = cfg.get("providers") or {}
    default_provider = providers.get("default") or "ollama"
    provider = provider_override or default_provider
    provider_cfg = providers.get(provider) or {}
    model = model_override or provider_cfg.get("default_model")
    if not model:
        raise ValueError("model not specified")
    if provider == "ollama":
        return OllamaClient(provider, model, temperature, max_tokens, provider_cfg)
    raise ValueError("unsupported provider")


def list_models_from_config(cfg: Dict[str, Any] | None) -> List[str]:
    cfg = cfg or {}
    models: List[str] = []
    providers = cfg.get("providers") or {}
    for provider_name, provider_cfg in providers.items():
        if provider_name == "default":
            continue
        available = provider_cfg.get("models") or []
        for m in available:
            models.append(f"{provider_name}:{m}")
    return models
