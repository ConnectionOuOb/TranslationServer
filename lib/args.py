import argparse
from lib.object import Settings


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Translation API")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port to listen on, default is 8080",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds, default is 60",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="facebook/nllb-200-3.3B",
        help="Model name, default is facebook/nllb-200-3.3B",
    )
    parser.add_argument(
        "-c",
        "--llm-config",
        type=str,
        default=None,
        help="Path to YAML config for LLM providers/models (optional)",
    )

    args = parser.parse_args()

    settings = Settings(args.port, args.timeout, args.model_name, args.llm_config)
    settings.check()

    return settings
