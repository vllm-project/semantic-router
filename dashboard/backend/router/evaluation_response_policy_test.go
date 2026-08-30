package router

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEvaluationResponsePolicyCoversEveryResponseShape(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		path   string
		status int
	}{
		{name: "API root", path: evaluationAPIPath, status: http.StatusOK},
		{name: "JSON success", path: evaluationAPIPath + "/catalog", status: http.StatusOK},
		{name: "handler error", path: evaluationAPIPath + "/runs/missing", status: http.StatusNotFound},
		{name: "artifact", path: evaluationAPIPath + "/runs/run-1/artifacts/metrics", status: http.StatusOK},
		{name: "artifact range", path: evaluationAPIPath + "/runs/run-1/artifacts/metrics", status: http.StatusPartialContent},
		{name: "SSE", path: evaluationAPIPath + "/runs/run-1/events", status: http.StatusOK},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if test.name == "SSE" {
					if _, ok := w.(http.Flusher); !ok {
						t.Fatal("evaluation response policy hid http.Flusher from the SSE handler")
					}
					w.Header().Set("Content-Type", "text/event-stream")
				}
				w.WriteHeader(test.status)
			})
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			if test.name == "artifact range" {
				request.Header.Set("Range", "bytes=0-9")
			}
			response := httptest.NewRecorder()

			withEvaluationResponsePolicy(next).ServeHTTP(response, request)

			if response.Code != test.status {
				t.Fatalf("status = %d, want %d", response.Code, test.status)
			}
			assertEvaluationNoStoreHeaders(t, response.Header())
		})
	}
}

func TestEvaluationResponsePolicyIncludesAuthenticationFailures(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc(evaluationAPIPath+"/catalog", func(http.ResponseWriter, *http.Request) {
		t.Fatal("evaluation handler ran while authentication was unavailable")
	})
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, evaluationAPIPath+"/catalog", nil)

	wrapWithAuth(mux, nil).ServeHTTP(response, request)

	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusServiceUnavailable)
	}
	assertEvaluationNoStoreHeaders(t, response.Header())
}

func TestEvaluationResponsePolicyDoesNotMatchSiblingPaths(t *testing.T) {
	t.Parallel()

	for _, path := range []string{"/", "/api/status", evaluationAPIPath + "evil/catalog"} {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, path, nil)
		withEvaluationResponsePolicy(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		})).ServeHTTP(response, request)

		if response.Header().Get("Cache-Control") != "" || response.Header().Get("Pragma") != "" {
			t.Fatalf("non-evaluation path %q received evaluation cache policy: %v", path, response.Header())
		}
	}
}

func assertEvaluationNoStoreHeaders(t *testing.T, header http.Header) {
	t.Helper()
	if got := header.Get("Cache-Control"); got != "private, no-store" {
		t.Fatalf("Cache-Control = %q, want %q", got, "private, no-store")
	}
	if got := header.Get("Pragma"); got != "no-cache" {
		t.Fatalf("Pragma = %q, want %q", got, "no-cache")
	}
}
