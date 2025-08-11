import aiohttp
from typing import Any, Dict, List
from .base import LLMClient


class OllamaClient(LLMClient):
    async def complete(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = (self.config.get("base_url") or "http://localhost:11434") + "/api/chat"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        usage = data.get("eval_count") or 0
        return {"content": content, "tokens_used": usage}

    async def list_models(self) -> List[str]:
        models = self.config.get("models") or []
        return [str(m) for m in models]
