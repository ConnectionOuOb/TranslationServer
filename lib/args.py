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
        default="facebook/nllb-200-distilled-600M",
        help="Model name, default is facebook/nllb-200-distilled-600M",
    )

    args = parser.parse_args()

    settings = Settings(args.port, args.timeout, args.model_name)
    settings.check()

    return settings
