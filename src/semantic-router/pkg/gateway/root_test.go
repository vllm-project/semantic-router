package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func TestHandlerRootRequestsStayOutOfRouting(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		keys   []string
		header http.Header
		status int
	}{
		{name: "public GET", method: http.MethodGet, path: "/", status: http.StatusNotFound},
		{name: "query string", method: http.MethodGet, path: "/?trace=1", status: http.StatusNotFound},
		{name: "public HEAD", method: http.MethodHead, path: "/", status: http.StatusNotFound},
		{name: "bearer", method: http.MethodGet, path: "/", keys: []string{"key"}, header: http.Header{"Authorization": {"Bearer key"}}, status: http.StatusNotFound},
		{name: "api-key", method: http.MethodGet, path: "/", keys: []string{"key"}, header: http.Header{"Api-Key": {"key"}}, status: http.StatusNotFound},
		{name: "authenticated HEAD", method: http.MethodHead, path: "/", keys: []string{"key"}, header: http.Header{"Authorization": {"Bearer key"}}, status: http.StatusNotFound},
		{name: "missing key", method: http.MethodGet, path: "/", keys: []string{"key"}, status: http.StatusUnauthorized},
		{name: "wrong key", method: http.MethodGet, path: "/", keys: []string{"key"}, header: http.Header{"Authorization": {"Bearer wrong"}}, status: http.StatusUnauthorized},
		{name: "unauthenticated HEAD", method: http.MethodHead, path: "/", keys: []string{"key"}, status: http.StatusUnauthorized},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p := &fakeProcessor{}
			u := &fakeUpstream{resp: func() *upstream.Response {
				return &upstream.Response{
					StatusCode: http.StatusNotFound,
					Header:     http.Header{"Content-Type": {"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"detail":"Not Found"}`)),
				}
			}}
			h := newTestHandler(t, p, u, test.keys...)
			records := make(chan AccessRecord, 1)
			h.opts.AccessLog = func(record AccessRecord) { records <- record }
			server := httptest.NewServer(h)
			defer server.Close()
			request, err := http.NewRequest(test.method, server.URL+test.path, nil)
			if err != nil {
				t.Fatal(err)
			}
			if test.header != nil {
				request.Header = test.header.Clone()
			}
			response, err := server.Client().Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != test.status || response.Header.Get("Content-Type") != "application/json" {
				t.Fatalf("response = %d %v %s", response.StatusCode, response.Header, body)
			}
			if test.method == http.MethodHead {
				if len(body) != 0 {
					t.Errorf("HEAD returned a body: %s", body)
				}
			} else if test.status == http.StatusNotFound {
				var envelope struct {
					Error struct {
						Type    string `json:"type"`
						Code    string `json:"code"`
						Message string `json:"message"`
					} `json:"error"`
				}
				if err := json.Unmarshal(body, &envelope); err != nil {
					t.Fatal(err)
				}
				if envelope.Error.Type != "invalid_request_error" || envelope.Error.Code != "not_found" ||
					!strings.Contains(envelope.Error.Message, "GET /") ||
					!strings.Contains(envelope.Error.Message, "POST /v1/chat/completions") ||
					!strings.Contains(envelope.Error.Message, "GET /v1/models") {
					t.Errorf("root response must explain the API entrypoints: %s", body)
				}
			} else if string(body) != unauthorizedBody {
				t.Errorf("authentication response changed: %s", body)
			}
			if test.status == http.StatusUnauthorized && response.Header.Get("WWW-Authenticate") != `Bearer realm="vllm-semantic-router"` {
				t.Errorf("authentication challenge = %q", response.Header.Get("WWW-Authenticate"))
			}
			record := <-records
			if record.Status != test.status || record.UpstreamHost != "" || record.UpstreamServiceTime != 0 {
				t.Errorf("root access log must not record a model call: %+v", record)
			}
			p.mu.Lock()
			opened := len(p.models)
			p.mu.Unlock()
			u.mu.Lock()
			forwarded := len(u.got)
			u.mu.Unlock()
			if opened != 0 || forwarded != 0 {
				t.Errorf("root request entered routing: %d sessions, %d upstream calls", opened, forwarded)
			}
		})
	}
}

func TestHandlerKeepsOtherRequestsInRouting(t *testing.T) {
	for _, test := range []struct{ method, path string }{
		{http.MethodGet, "/v1/models"},
		{http.MethodPost, "/v1/chat/completions"},
		{http.MethodGet, "/provider-extension"},
		{http.MethodPost, "/"},
	} {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			p := &fakeProcessor{}
			u := &fakeUpstream{resp: okUpstream("ok", http.Header{})}
			response := httptest.NewRecorder()
			newTestHandler(t, p, u).ServeHTTP(response, httptest.NewRequest(test.method, test.path, nil))
			if response.Code != http.StatusOK || len(p.models) != 1 || len(u.got) != 1 {
				t.Fatalf("request no longer routes: status %d, sessions %d, upstream calls %d", response.Code, len(p.models), len(u.got))
			}
		})
	}
}
