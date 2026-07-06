import aiohttp
from typing import Any, Dict, List
from .base import LLMClient


class OpenAIClient(LLMClient):
    async def complete(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        base_url = (self.config.get("base_url") or "http://localhost:8080/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        chat_template_kwargs = self.config.get("chat_template_kwargs")
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    detail = data.get("error") or data
                    raise RuntimeError(f"LLM API error {resp.status}: {detail}")

        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        content = (message.get("content") or "").strip()
        usage = (data.get("usage") or {}).get("total_tokens") or 0
        return {"content": content, "tokens_used": usage}

    async def list_models(self) -> List[str]:
        models = self.config.get("models") or []
        return [str(m) for m in models]
