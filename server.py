import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from lib import log
from lib.args import parse_args
from lib.model import ModelManager
from lib.object import TranslateRequest, TranslateResponse

settings = parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager = app.state.model_manager
    asyncio.create_task(model_manager.monitor_idle(settings.timeout))
    yield


def main() -> None:
    # FastAPI settings
    app = FastAPI(
        title="Translation API",
        version="0.1.0",
        description="Translation API",
        contact={
            "name": "Connection Lee",
            "email": "connection.bt12@nycu.edu.tw",
        },
        lifespan=lifespan,
    )
    app.state.model_manager = ModelManager(settings.model_name)

    # Translate endpoint
    @app.post(
        "/translate",
        response_model=TranslateResponse,
        tags=["translate"],
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
            translated = await app.state.model_manager.translate(
                req.text, req.target_language
            )
            return TranslateResponse(translation=translated)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_config=log.get_config(),
    )


if __name__ == "__main__":
    main()
