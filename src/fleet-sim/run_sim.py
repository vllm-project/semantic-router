#!/usr/bin/env python3
"""Thin CLI entrypoint for vllm-sr-sim."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main(argv=None):
    from fleet_sim.cli_parser import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    main()
