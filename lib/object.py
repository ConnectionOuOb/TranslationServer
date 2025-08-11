from pydantic import BaseModel
from lib import define


class Settings:
    port: int
    timeout: int
    model_name: str
    llm_config_path: str | None

    def __init__(
        self,
        port: int,
        timeout: int,
        model_name: str,
        llm_config_path: str | None = None,
    ):
        self.port = port
        self.timeout = timeout
        self.model_name = model_name
        self.llm_config_path = llm_config_path

    def check(self) -> None:
        self._check_port()
        self._check_timeout()

        if self.model_name not in define.SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Model must be one of {define.SUPPORTED_MODEL_NAMES}")

    def _check_port(self) -> None:
        if self.port < 1024 or self.port > 65535:
            raise ValueError("Port must be between 1024 and 65535")

    def _check_timeout(self) -> None:
        if self.timeout < 1:
            raise ValueError("Timeout must be greater than 0")


class TranslateRequest(BaseModel):
    text: str
    language: str


class TranslateResponse(BaseModel):
    translation: str
