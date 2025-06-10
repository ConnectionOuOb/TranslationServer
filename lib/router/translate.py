from fastapi import APIRouter, HTTPException, status
from lib.model import ModelManager
from lib.object import TranslateRequest, TranslateResponse

router = APIRouter(tags=["Translate"])


def create_router(model_manager: ModelManager) -> APIRouter:
    # Translate endpoint
    @router.post(
        "/translate",
        response_model=TranslateResponse,
        summary="Translate text",
        description="Translate text from source language to target language",
        responses={
            status.HTTP_200_OK: {
                "description": "Translation successful",
                "model": TranslateResponse,
            },
        },
    )
    async def translate(req: TranslateRequest) -> TranslateResponse:
        """
        Translate text from source language to target language.
        Args:
            req: TranslateRequest
        Returns:
            TranslateResponse
        """
        try:
            translated = await model_manager.translate(req.text, req.language)
            return TranslateResponse(translation=translated)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return router
