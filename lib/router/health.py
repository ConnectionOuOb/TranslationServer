import aiohttp
from fastapi import APIRouter

from lib.model import ModelManager
from lib.object import HealthResponse, LlmHealth, NllbHealth

router = APIRouter(tags=["Health"])


def _provider_cfg(config: dict | None) -> dict:
    providers = (config or {}).get("providers") or {}
    provider_name = providers.get("default") or "openai"
    return provider_name, providers.get(provider_name) or {}


async def _check_llm_reachable(
    base_url: str | None, api_key: str | None = None
) -> bool | None:
    if not base_url:
        return None
    url = f"{base_url.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status < 500
    except Exception:
        return False


def create_router(model_manager: ModelManager, llm_config: dict | None) -> APIRouter:
    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="健康檢查",
        description="確認 API 與模型是否正常。`status=ok` 表示可用。",
    )
    async def health() -> HealthResponse:
        nllb = NllbHealth(**model_manager.health_info())

        provider_name, provider_cfg = _provider_cfg(llm_config)
        configured = llm_config is not None and bool(provider_cfg)
        base_url = provider_cfg.get("base_url") if configured else None
        api_key = provider_cfg.get("api_key") if configured else None
        reachable = (
            await _check_llm_reachable(base_url, api_key) if configured else None
        )

        llm = LlmHealth(
            configured=configured,
            provider=provider_name if configured else None,
            base_url=base_url,
            default_model=provider_cfg.get("default_model") if configured else None,
            reachable=reachable,
        )

        degraded = not nllb.loaded or (configured and reachable is False)
        return HealthResponse(
            status="degraded" if degraded else "ok",
            api="alive",
            nllb=nllb,
            llm=llm,
        )

    return router
