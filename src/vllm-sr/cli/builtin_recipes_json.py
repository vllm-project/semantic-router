"""Emit built-in Recipe listings and compositions as JSON for internal consumers.

``list`` prints the ``vllm-sr recipe builtin list`` document. ``compose`` reads
a request with config_path, bundle, recipe, bindings and model_names from stdin
and prints the composed config. A rejected request prints its reason and exits
with REJECTED_EXIT_CODE; nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cli.builtin_recipes import (
    compose_builtin_recipe,
    list_builtin_recipes,
    load_provider_config,
)
from cli.parser import ConfigParseError

# EX_DATAERR: distinct from the statuses Python reports for its own failures.
REJECTED_EXIT_CODE = 65


def _compose(request: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(request["config_path"])
    return compose_builtin_recipe(
        request["recipe"],
        bundle=request["bundle"],
        config=load_provider_config(config_path),
        config_path=config_path,
        bindings=request["bindings"],
        model_names=request["model_names"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("list", "compose"))
    if parser.parse_args(argv).operation == "list":
        result = list_builtin_recipes()
    else:
        request = json.load(sys.stdin)
        try:
            result = _compose(request)
        except (ValueError, ConfigParseError) as error:
            print(error)
            return REJECTED_EXIT_CODE
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
