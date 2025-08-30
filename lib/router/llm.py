import asyncio
from fastapi import APIRouter, HTTPException, status
from lib.code import NLLB200_LANGUAGES_DECODED
from lib.llm.factory import create_llm_client, list_models_from_config
from lib.object import TranslateRequest, TranslateResponse


def _to_natural_language(code: str) -> str:
    name = NLLB200_LANGUAGES_DECODED.get(code) or code
    # strip script suffix in parentheses for prompt simplicity
    if " (" in name:
        name = name.split(" (")[0]
    return name


def create_router(config: dict | None, timeout_seconds: int) -> APIRouter:
    router = APIRouter(tags=["LLM"])

    @router.post(
        "/llm",
        response_model=TranslateResponse,
        summary="Translate via LLM",
        description="Translate text using an LLM; request body matches /translate with extra LLM options.",
        responses={
            status.HTTP_200_OK: {
                "description": "Translation successful",
                "model": TranslateResponse,
            },
        },
    )
    async def llm_translate(req: TranslateRequest) -> TranslateResponse:
        try:
            target_name = _to_natural_language(req.language)
            system_prompt = (
                "You are a professional translator. Translate the user's input into the "
                f"target language: {target_name}. Only output the translated text."
            )
            user_prompt = req.text

            client = create_llm_client(
                config,
                None,
                None,
                0.2,
                512,
            )

            # Enforce timeout similar to local LLM
            result = await asyncio.wait_for(
                client.complete(system_prompt, user_prompt), timeout=timeout_seconds
            )
            content = (result or {}).get("content") or ""
            return TranslateResponse(translation=content)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="LLM translation timeout")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get(
        "/model",
        summary="List available LLM models",
        description="List models from configured providers",
    )
    async def list_models() -> dict[str, list[str]]:
        models = list_models_from_config(config)
        return {"models": models}

    return router
