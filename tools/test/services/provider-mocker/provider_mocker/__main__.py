"""Run the fixture service without downloading or loading any model."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic provider protocol fixtures"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("provider_mocker.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
