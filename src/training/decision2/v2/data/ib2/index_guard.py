"""G0 for IB2: IB1's Index-row matcher (``v2.data.ib1.index_guard``) unchanged, with IB2's template strings.

    python3 -m v2.data.ib2.index_guard reference|scan|controls ...   (arguments as IB1's)

Only the set of fixed instruction and option strings that are reported apart instead of matched as data differs.
"""

from __future__ import annotations

from v2.data.ib1 import index_guard as guard
from v2.data.ib2.families import TEMPLATE_STRINGS


def main(argv: list[str] | None = None) -> int:
    guard.TEMPLATE_STRINGS = TEMPLATE_STRINGS
    return guard.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
