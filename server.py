import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from lib import log
from lib.args import parse_args
from lib.model import ModelManager
from lib.router import health, llm as llm_router, nllb
from lib.llm_config import load_llm_config

settings = parse_args()

API_DESCRIPTION = """
## 用法

```bash
# NLLB 翻譯
curl -X POST /translate -d '{"text":"Hello","language":"zho_Hans"}'

# LLM 翻譯（格式相同）
curl -X POST /llm -d '{"text":"Hello","language":"zho_Hans"}'

# 查語言代碼
curl /encode
curl /decode

# 健康檢查
curl /health
```

`language` 常用代碼：`zho_Hans`（簡中）、`zho_Hant`（繁中）、`eng_Latn`（英）、`jpn_Jpan`（日）

## LLM API Key

`/llm` 需設定 API Key。於 `docker/llm_secrets.yaml` 填入 `api_key`（勿提交 git），或設定環境變數 `LLM_API_KEY`。啟動時自動載入。
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager = app.state.model_manager
    await model_manager.load_model()
    yield


def main() -> None:
    app = FastAPI(
        title="Translation API",
        version="0.2.0",
        description=API_DESCRIPTION,
        contact={
            "name": "Connection Lee",
            "email": "connection.bt12@nycu.edu.tw",
        },
        license_info={
            "name": "MIT",
        },
        openapi_tags=[
            {"name": "Health", "description": "健康檢查"},
            {"name": "NLLB 200", "description": "本地 NLLB 翻譯"},
            {"name": "LLM", "description": "外部 LLM 翻譯"},
        ],
        lifespan=lifespan,
    )
    app.state.model_manager = ModelManager(settings.model_name)

    llm_config = load_llm_config(settings.llm_config_path)
    app.state.llm_config = llm_config

    app.include_router(health.create_router(app.state.model_manager, llm_config))
    app.include_router(nllb.create_router(app.state.model_manager))
    app.include_router(llm_router.create_router(llm_config, settings.timeout))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_config=log.get_config(),
    )


if __name__ == "__main__":
    main()
