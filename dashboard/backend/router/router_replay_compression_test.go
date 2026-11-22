package router

import (
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// A run read as long as an agent session's: megabytes of repetitive JSON.
var replayRunJSON = `{"object":"router_replay.run","calls":[` +
	strings.TrimSuffix(strings.Repeat(`{"record_id":"r","agent":"main","model":"deepseek/deepseek-v4.1-flash","markers":["tool_loop"]},`, 4000), ",") +
	`]}`

func replayProxy(t *testing.T, body string, contentType string) (*httptest.Server, func(*http.Request) *httptest.ResponseRecorder) {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, body)
	}))
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte("setup:\n  mode: false\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	resolver := setupmode.New(configPath, false)
	routerAPIProxy, err := proxy.NewReverseProxy(upstream.URL, "/api/router", true)
	if err != nil {
		t.Fatal(err)
	}
	attachRouterReplayResponseRedaction(routerAPIProxy)
	cfg := &config.Config{RouterAPIURL: upstream.URL}
	return upstream, func(request *http.Request) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		serveRouterAPIProxy(recorder, request, cfg, nil, routerAPIProxy, nil, resolver, nil)
		return recorder
	}
}

func TestReplayRunReadIsGzippedForABrowserThatAcceptsIt(t *testing.T) {
	t.Parallel()
	upstream, serve := replayProxy(t, replayRunJSON, "application/json")
	defer upstream.Close()

	request := httptest.NewRequest(http.MethodGet, "/api/router/api/v1/observability/replays/run?session_id=s", nil)
	request.Header.Set("Accept-Encoding", "gzip, deflate, br")
	recorder := serve(request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d", recorder.Code)
	}
	if got := recorder.Header().Get("Content-Encoding"); got != "gzip" {
		t.Fatalf("Content-Encoding = %q, want gzip", got)
	}
	if !strings.Contains(recorder.Header().Get("Vary"), "Accept-Encoding") {
		t.Fatalf("Vary = %q, want Accept-Encoding", recorder.Header().Get("Vary"))
	}
	if recorder.Body.Len()*8 > len(replayRunJSON) {
		t.Fatalf("compressed to %d of %d bytes: expected far smaller", recorder.Body.Len(), len(replayRunJSON))
	}
	reader, err := gzip.NewReader(recorder.Body)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if string(decoded) != replayRunJSON {
		t.Fatal("the decompressed body differs from the Router's answer")
	}
}

func TestReplayAnswersStayPlainWhereGzipDoesNotApply(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name, body, contentType, acceptEncoding, path string
	}{
		{"no gzip accepted", replayRunJSON, "application/json", "", "/api/router/api/v1/observability/replays/run?session_id=s"},
		{"gzip refused", replayRunJSON, "application/json", "gzip;q=0, br", "/api/router/api/v1/observability/replays/run?session_id=s"},
		{"a small answer", `{"object":"router_replay.runs","data":[]}`, "application/json", "gzip", "/api/router/api/v1/observability/replays/runs"},
		{"not JSON", replayRunJSON, "text/plain", "gzip", "/api/router/api/v1/observability/replays/run?session_id=s"},
		{"not Replay", replayRunJSON, "application/json", "gzip", "/api/router/docs"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			upstream, serve := replayProxy(t, tc.body, tc.contentType)
			defer upstream.Close()
			request := httptest.NewRequest(http.MethodGet, tc.path, nil)
			if tc.acceptEncoding != "" {
				request.Header.Set("Accept-Encoding", tc.acceptEncoding)
			}
			recorder := serve(request)
			if got := recorder.Header().Get("Content-Encoding"); got != "" {
				t.Fatalf("Content-Encoding = %q, want none", got)
			}
			if recorder.Body.String() != tc.body {
				t.Fatal("the body was changed")
			}
		})
	}
}

func TestAcceptsGzip(t *testing.T) {
	t.Parallel()
	for header, want := range map[string]bool{
		"gzip, deflate, br, zstd": true,
		"br;q=1, GZIP;q=0.5":      true,
		"gzip;q=0":                false,
		"gzip; q=0.000":           false,
		"identity":                false,
		"":                        false,
	} {
		if got := acceptsGzip(header); got != want {
			t.Errorf("acceptsGzip(%q) = %v, want %v", header, got, want)
		}
	}
}
