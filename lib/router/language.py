from fastapi import APIRouter
from lib import code


router = APIRouter(tags=["Language"])


def create_router() -> APIRouter:
    @router.get("/encode", summary="Get NLLB 200 language encoded")
    async def encode() -> dict[str, str]:
        return code.NLLB200_LANGUAGES_ENCODED

    @router.get("/decode", summary="Get NLLB 200 language decoded")
    async def decode() -> dict[str, str]:
        return code.NLLB200_LANGUAGES_DECODED

    return router
