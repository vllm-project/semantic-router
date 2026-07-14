package proxy

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestHTTPAndWebSocketProxyJoinTargetBasePathConsistently(t *testing.T) {
	t.Parallel()

	for _, targetBase := range []string{
		"http://upstream.example/base",
		"http://upstream.example/base/",
	} {
		t.Run(targetBase, func(t *testing.T) {
			t.Parallel()
			const stripPrefix = "/embedded"
			const incoming = "http://dashboard.example/embedded/socket%20name"

			httpProxy, err := NewReverseProxy(targetBase, stripPrefix, false)
			if err != nil {
				t.Fatalf("NewReverseProxy: %v", err)
			}
			httpRequest := httptest.NewRequest(http.MethodGet, incoming, nil)
			httpProxy.Director(httpRequest)

			target, err := url.Parse(targetBase)
			if err != nil {
				t.Fatalf("url.Parse: %v", err)
			}
			incomingURL, err := url.Parse(incoming)
			if err != nil {
				t.Fatalf("url.Parse incoming: %v", err)
			}
			webSocketPath := joinProxyTargetPath(
				target.Path,
				stripProxyPath(incomingURL.Path, stripPrefix),
			)
			webSocketTarget, err := webSocketRequestTarget(webSocketPath, "")
			if err != nil {
				t.Fatalf("webSocketRequestTarget: %v", err)
			}

			const want = "/base/socket%20name"
			if got := httpRequest.URL.EscapedPath(); got != want {
				t.Fatalf("HTTP target path = %q, want %q", got, want)
			}
			if webSocketTarget != want {
				t.Fatalf("WebSocket target path = %q, want %q", webSocketTarget, want)
			}
		})
	}
}

func TestProxyPathWithoutTargetBaseRemainsRootRelative(t *testing.T) {
	t.Parallel()
	if got := joinProxyTargetPath("", stripProxyPath("/embedded/socket", "/embedded")); got != "/socket" {
		t.Fatalf("joined path = %q, want /socket", got)
	}
}
