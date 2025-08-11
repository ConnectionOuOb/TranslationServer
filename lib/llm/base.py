from typing import Any, Dict, List, Optional


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = config or {}

    async def complete(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def list_models(self) -> List[str]:
        raise NotImplementedError
