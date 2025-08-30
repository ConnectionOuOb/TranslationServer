import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from lib import log
from lib.args import parse_args
from lib.model import ModelManager
from lib.router import translate, language
from lib.router import llm as llm_router
import os
import yaml

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
    # Load LLM config if provided
    llm_config = None
    if settings.llm_config_path and os.path.exists(settings.llm_config_path):
        with open(settings.llm_config_path, "r", encoding="utf-8") as f:
            llm_config = yaml.safe_load(f) or {}
    app.state.llm_config = llm_config

    app.include_router(translate.create_router(app.state.model_manager))
    app.include_router(language.create_router())
    app.include_router(llm_router.create_router(app.state.llm_config, settings.timeout))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_config=log.get_config(),
    )


if __name__ == "__main__":
    main()
