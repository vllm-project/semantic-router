#!/usr/bin/env python3
"""Start the vllm-sr-sim service.

Usage
-----
    python server.py                   # default: http://localhost:8000
    python server.py --port 8080
    python server.py --host 0.0.0.0 --port 8000 --reload
"""
import sys

from run_sim import main

if __name__ == "__main__":
    main(["serve", *sys.argv[1:]])
