import os
from typing import Any

import yaml


def _merge_provider_config(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in (overlay.get("providers") or {}).items():
        if key == "default":
            base.setdefault("providers", {})["default"] = value
            continue
        if isinstance(value, dict):
            base.setdefault("providers", {}).setdefault(key, {}).update(value)


def load_llm_config(config_path: str | None) -> dict[str, Any] | None:
    if not config_path or not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    secrets_path = os.path.join(os.path.dirname(config_path), "llm_secrets.yaml")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}
        _merge_provider_config(config, secrets)

    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        providers = config.setdefault("providers", {})
        provider_name = providers.get("default") or "openai"
        providers.setdefault(provider_name, {})["api_key"] = api_key

    return config
