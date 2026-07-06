import asyncio
from fastapi import APIRouter, HTTPException

from lib.code import NLLB200_LANGUAGES_DECODED
from lib.llm.factory import create_llm_client, list_models_from_config
from lib.llm.tokens import estimate_tokens
from lib.object import TranslateRequest, TranslateResponse


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


def create_router(config: dict | None, timeout_seconds: int) -> APIRouter:
    router = APIRouter(tags=["LLM"])

    @router.post(
        "/llm",
        response_model=TranslateResponse,
        summary="翻譯",
        description="請求格式同 /translate，透過外部 LLM 翻譯。",
    )
    async def llm_translate(req: TranslateRequest) -> TranslateResponse:
        if not config:
            raise HTTPException(status_code=503, detail="LLM 未設定")
        try:
            target_name = _to_natural_language(req.language)
            system_prompt = (
                "You are a professional translator. Translate the user's input into the "
                f"target language: {target_name}. Only output the translated text."
            )
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
                0.2,
                _llm_max_tokens(config),
            )

            result = await asyncio.wait_for(
                client.complete(system_prompt, user_prompt), timeout=timeout_seconds
            )
            content = (result or {}).get("content") or ""
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
