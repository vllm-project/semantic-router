"""Headless Chromium rendering for proxy builders (one browser per worker process).

``setup.sh`` provides the Playwright package, the Chromium build and the shared libraries; the
environment it prints (``--env``) must be active. Pages are rendered offline from HTML strings.
"""

from __future__ import annotations

from typing import Any

_playwright = None
_browser = None


def start() -> None:
    global _playwright, _browser
    if _browser is not None:
        return
    from playwright.sync_api import sync_playwright

    _playwright = sync_playwright().start()
    _browser = _playwright.chromium.launch(
        args=["--no-sandbox", "--disable-gpu", "--font-render-hinting=none"]
    )


def stop() -> None:
    global _playwright, _browser
    if _browser is not None:
        _browser.close()
        _playwright.stop()
    _playwright = _browser = None


def html(
    content: str,
    width: int,
    height: int | None = None,
    selectors: dict[str, str] | None = None,
    scale: float = 1.0,
    full_page: bool = True,
) -> tuple[bytes, dict[str, list[dict[str, float]]], tuple[int, int]]:
    """PNG screenshot of ``content`` plus bounding boxes (CSS px x ``scale``) per selector.

    Returns ``(png, boxes, (width, height))`` where ``boxes[key]`` lists ``{x, y, w, h, text}`` for
    every visible element matching ``selectors[key]``.
    """
    start()
    page = _browser.new_page(
        viewport={"width": width, "height": height or 64}, device_scale_factor=scale
    )
    try:
        page.set_content(content, wait_until="load")
        page.wait_for_timeout(30)
        boxes: dict[str, list[dict[str, Any]]] = {}
        for key, selector in (selectors or {}).items():
            boxes[key] = page.eval_on_selector_all(
                selector,
                """(els, s) => els.map(e => { const r = e.getBoundingClientRect();
                    const st = getComputedStyle(e);
                    return {x: (r.left + window.scrollX) * s, y: (r.top + window.scrollY) * s,
                            w: r.width * s, h: r.height * s,
                            text: (e.innerText || e.value || e.getAttribute('aria-label') || e.getAttribute('placeholder') || '').trim(),
                            id: e.id || '', visible: st.visibility !== 'hidden' && st.display !== 'none' && r.width > 0 && r.height > 0}; })""",
                scale,
            )
        size = page.evaluate(
            "() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight]"
        )
        png = page.screenshot(full_page=full_page, type="png")
    finally:
        page.close()
    w = int(round(min(size[0], width) * scale)) if full_page else int(width * scale)
    h = int(round(size[1] * scale)) if full_page else int((height or 64) * scale)
    return png, boxes, (w, h)
