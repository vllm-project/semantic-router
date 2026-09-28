// Grafana runs in a same-origin iframe, which has its own window.fetch. Carry
// the Dashboard's readable CSRF cookie on its normal API writes, just as the
// parent frontend does. Never read or forward the HttpOnly session credential.
(() => {
  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const request = input instanceof Request ? input : null;
    const url = new URL(request ? request.url : input, window.location.href);
    const method = (init?.method ?? request?.method ?? "GET").toUpperCase();
    if (
      url.origin !== window.location.origin ||
      !url.pathname.startsWith("/embedded/grafana/") ||
      ["GET", "HEAD", "OPTIONS"].includes(method)
    ) {
      return originalFetch(input, init);
    }
    const headers = new Headers(init?.headers ?? request?.headers);
    if (!headers.has("X-CSRF-Token")) {
      const cookie = document.cookie
        .split(";")
        .map((part) => part.trim())
        .find((part) => part.startsWith("vsr_csrf="));
      if (cookie) headers.set("X-CSRF-Token", cookie.slice("vsr_csrf=".length));
    }
    return originalFetch(input, { ...init, headers });
  };
})();
