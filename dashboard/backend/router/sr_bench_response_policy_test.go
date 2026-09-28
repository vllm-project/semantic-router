package router

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSRBenchResponsePolicyCoversEveryResponseShape(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		path   string
		status int
	}{
		{name: "API root", path: srBenchAPIPath, status: http.StatusOK},
		{name: "JSON success", path: srBenchAPIPath + "/catalog", status: http.StatusOK},
		{name: "handler error", path: srBenchAPIPath + "/runs/missing", status: http.StatusNotFound},
		{name: "report", path: srBenchAPIPath + "/runs/run-1/report", status: http.StatusOK},
		{name: "events", path: srBenchAPIPath + "/runs/run-1/events", status: http.StatusOK},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
			})
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			response := httptest.NewRecorder()

			withSRBenchResponsePolicy(next).ServeHTTP(response, request)

			if response.Code != test.status {
				t.Fatalf("status = %d, want %d", response.Code, test.status)
			}
			assertSRBenchNoStoreHeaders(t, response.Header())
		})
	}
}

func TestSRBenchResponsePolicyIncludesAuthenticationFailures(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc(srBenchAPIPath+"/catalog", func(http.ResponseWriter, *http.Request) {
		t.Fatal("sr-bench handler ran while authentication was unavailable")
	})
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, srBenchAPIPath+"/catalog", nil)

	wrapWithAuth(mux, nil).ServeHTTP(response, request)

	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusServiceUnavailable)
	}
	assertSRBenchNoStoreHeaders(t, response.Header())
}

func TestSRBenchResponsePolicyDoesNotMatchSiblingPaths(t *testing.T) {
	t.Parallel()

	for _, path := range []string{"/", "/api/status", srBenchAPIPath + "evil/catalog"} {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, path, nil)
		withSRBenchResponsePolicy(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		})).ServeHTTP(response, request)

		if response.Header().Get("Cache-Control") != "" || response.Header().Get("Pragma") != "" {
			t.Fatalf("non-benchmark path %q received benchmark cache policy: %v", path, response.Header())
		}
	}
}

func assertSRBenchNoStoreHeaders(t *testing.T, header http.Header) {
	t.Helper()
	if got := header.Get("Cache-Control"); got != "private, no-store" {
		t.Fatalf("Cache-Control = %q, want %q", got, "private, no-store")
	}
	if got := header.Get("Pragma"); got != "no-cache" {
		t.Fatalf("Pragma = %q, want %q", got, "no-cache")
	}
}
