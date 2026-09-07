"""CLI for training multi-modal-embed-small."""

from __future__ import annotations

from .runtime import apply_overrides, build_parser, load_config, print_config


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    if args.print_config:
        print_config(config)
        return
    from .distributed import (  # noqa: PLC0415 - optional runtime dependency
        cleanup_distributed,
        setup_distributed,
    )
    from .runner import run  # noqa: PLC0415 - optional runtime dependency

    context = setup_distributed()
    try:
        run(config, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
