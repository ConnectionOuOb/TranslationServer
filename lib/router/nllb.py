from fastapi import APIRouter, HTTPException

from lib import code
from lib.model import ModelManager
from lib.object import TranslateRequest, TranslateResponse

router = APIRouter(tags=["NLLB 200"])


def create_router(model_manager: ModelManager) -> APIRouter:
    @router.post(
        "/translate",
        response_model=TranslateResponse,
        summary="翻譯",
        description="傳入原文與目標語言代碼，回傳譯文。",
    )
    async def translate(req: TranslateRequest) -> TranslateResponse:
        if not model_manager.is_loaded():
            raise HTTPException(status_code=503, detail="模型尚未就緒")
        try:
            translated = await model_manager.translate(req.text, req.language)
            return TranslateResponse(translation=translated)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get(
        "/encode",
        summary="語言名稱 → 代碼",
        description="查詢語言名稱對應的 FLORES-200 代碼。",
    )
    async def encode() -> dict[str, str]:
        return code.NLLB200_LANGUAGES_ENCODED

    @router.get(
        "/decode",
        summary="代碼 → 語言名稱",
        description="查詢 FLORES-200 代碼對應的語言名稱。",
    )
    async def decode() -> dict[str, str]:
        return code.NLLB200_LANGUAGES_DECODED

    return router
