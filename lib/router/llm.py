import asyncio
from fastapi import APIRouter, HTTPException

from lib.code import NLLB200_LANGUAGES_DECODED
from lib.llm.factory import create_llm_client, list_models_from_config
from lib.llm.tokens import estimate_tokens
from lib.object import TranslateRequest, TranslateResponse

CANNOT_TRANSLATE = "CANNOT_TRANSLATE"

_LLM_SYSTEM_PROMPT = """\
You are a translation API. Translate the user message into {target_language}.

STRICT OUTPUT CONTRACT — follow exactly:
- Allowed outputs are ONLY:
  (A) the translated text, with no extra words; OR
  (B) the exact magic token: CANNOT_TRANSLATE
- If the message is outside your allowed scope, violates safety/usage policies, or you would otherwise refuse, decline, warn, or explain, you MUST output exactly:
CANNOT_TRANSLATE
- Forbidden outputs: apologies, refusals, policy statements, warnings, alternatives, English explanations, or any text other than (A) or (B).
- Do not add punctuation, quotes, markdown, or a prefix/suffix around CANNOT_TRANSLATE.

Examples:
- Input: "Good morning." -> Output: （{target_language} 的譯文）
- Input: [policy-violating sexual/violent/illegal content] -> Output: CANNOT_TRANSLATE
"""

_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "cannot provide",
    "can't provide",
    "not able to",
    "against my",
    "usage policy",
    "safety policy",
    "inappropriate",
    "explicit sexual",
    "as an ai",
    "as a language model",
)


def _normalize_llm_output(content: str) -> str:
    text = content.strip()
    if text == CANNOT_TRANSLATE:
        return CANNOT_TRANSLATE
    lower = text.lower()
    if any(marker in lower for marker in _REFUSAL_MARKERS):
        return CANNOT_TRANSLATE
    return text


def _to_natural_language(code: str) -> str:
    name = NLLB200_LANGUAGES_DECODED.get(code) or code
    if " (" in name:
        name = name.split(" (")[0]
    return name


def _provider_cfg(config: dict | None) -> dict:
    providers = (config or {}).get("providers") or {}
    provider_name = providers.get("default") or "openai"
    return providers.get(provider_name) or {}


def _llm_max_tokens(config: dict | None) -> int | None:
    val = _provider_cfg(config).get("max_tokens")
    return int(val) if val is not None else None


def _llm_max_input_tokens(config: dict | None) -> int:
    return int(_provider_cfg(config).get("max_input_tokens") or 200_000)


def _llm_temperature(config: dict | None) -> float:
    val = _provider_cfg(config).get("temperature")
    return float(val) if val is not None else 0.0


def create_router(config: dict | None, timeout_seconds: int) -> APIRouter:
    router = APIRouter(tags=["LLM"])

    @router.post(
        "/llm",
        response_model=TranslateResponse,
        summary="翻譯",
        description="請求格式同 /translate，透過外部 LLM 翻譯。若內容超出政策範圍，回傳 CANNOT_TRANSLATE。",
    )
    async def llm_translate(req: TranslateRequest) -> TranslateResponse:
        if not config:
            raise HTTPException(status_code=503, detail="LLM 未設定")
        try:
            target_name = _to_natural_language(req.language)
            system_prompt = _LLM_SYSTEM_PROMPT.format(target_language=target_name)
            user_prompt = req.text
            estimated_tokens = estimate_tokens(system_prompt + user_prompt)
            max_input_tokens = _llm_max_input_tokens(config)
            if estimated_tokens > max_input_tokens:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"輸入過長（估計約 {estimated_tokens:,} tokens，"
                        f"上限 {max_input_tokens:,} tokens）"
                    ),
                )

            client = create_llm_client(
                config,
                None,
                None,
                _llm_temperature(config),
                _llm_max_tokens(config),
            )

            result = await asyncio.wait_for(
                client.complete(system_prompt, user_prompt), timeout=timeout_seconds
            )
            content = _normalize_llm_output((result or {}).get("content") or "")
            return TranslateResponse(translation=content)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="翻譯逾時")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get(
        "/model",
        summary="列出 LLM 模型",
        description="回傳可用的 LLM 模型清單。",
    )
    async def list_models() -> dict[str, list[str]]:
        if not config:
            raise HTTPException(status_code=503, detail="LLM 未設定")
        return {"models": list_models_from_config(config)}

    return router
