"""G0 / G0u for IB4: IB1's Index-row matcher and IB3's URL / host guard, unchanged, with IB4's template strings.

python3 -m v2.data.ib4.index_guard reference|scan|controls ...           (arguments as IB1's)
python3 -m v2.data.ib4.index_guard url-reference|url-scan|url-controls ... (arguments as IB3's)
"""

from __future__ import annotations

from v2.data.ib1 import index_guard as guard
from v2.data.ib3 import index_guard as url_guard
from v2.data.ib4.families import TEMPLATE_STRINGS


def main(argv: list[str] | None = None) -> int:
    guard.TEMPLATE_STRINGS = TEMPLATE_STRINGS
    url_guard.TEMPLATE_STRINGS = TEMPLATE_STRINGS
    return url_guard.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
