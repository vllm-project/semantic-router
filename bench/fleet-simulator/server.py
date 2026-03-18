#!/usr/bin/env python3
"""Start the inference-fleet-sim dashboard server.

Usage
-----
    python server.py                   # default: http://localhost:8000
    python server.py --port 8080
    python server.py --host 0.0.0.0 --port 8000 --reload
"""
import argparse
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="inference-fleet-sim dashboard server"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload (development)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes")
    args = parser.parse_args()

    print(f"\n  inference-fleet-sim dashboard")
    print(f"  {'─'*36}")
    print(f"  Dashboard : http://{args.host}:{args.port}/")
    print(f"  API docs  : http://{args.host}:{args.port}/api/docs")
    print(f"  OpenAPI   : http://{args.host}:{args.port}/api/openapi.json\n")

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
