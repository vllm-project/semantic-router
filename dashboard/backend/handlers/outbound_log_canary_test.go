package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func captureOutboundHandlerLogs(run func()) string {
	var output bytes.Buffer
	previous := log.Writer()
	restored := false
	defer func() {
		if !restored {
			log.SetOutput(previous)
		}
	}()

	log.SetOutput(&output)
	run()
	log.SetOutput(previous)
	restored = true
	return output.String()
}

func assertSecretsAbsent(t *testing.T, surfaces []string, secrets ...string) {
	t.Helper()
	for _, surface := range surfaces {
		for _, secret := range secrets {
			if strings.Contains(surface, secret) {
				t.Fatalf("diagnostic surface leaked %q: %s", secret, surface)
			}
		}
	}
}

func TestFetchRawDiagnosticsAreContentFree(t *testing.T) {
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("client-error-secret")
	}}
	body, err := json.Marshal(FetchRawRequest{
		URL: "https://example.com/path-secret/config?token=query-secret",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/fetch-raw", bytes.NewReader(body))
	w := httptest.NewRecorder()

	logs := captureOutboundHandlerLogs(func() {
		fetchRawHandlerWithClient(client)(w, req)
	})
	assertSecretsAbsent(t, []string{logs, w.Body.String()},
		"path-secret", "query-secret", "client-error-secret")
}

func TestOpenWebDiagnosticsAreContentFree(t *testing.T) {
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("client-error-secret")
	}}
	body, err := json.Marshal(OpenWebRequest{
		URL: "https://example.com/path-secret/article?token=query-secret",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/open-web", bytes.NewReader(body))
	w := httptest.NewRecorder()

	logs := captureOutboundHandlerLogs(func() {
		openWebHandlerWithClient(client)(w, req)
	})
	assertSecretsAbsent(t, []string{logs, w.Body.String()},
		"path-secret", "query-secret", "client-error-secret")
}

func TestWebSearchDiagnosticsAreContentFree(t *testing.T) {
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Body:       io.NopCloser(strings.NewReader("upstream-body-secret")),
		}, nil
	}}
	body, err := json.Marshal(WebSearchRequest{Query: "query-secret"})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/web-search", bytes.NewReader(body))
	w := httptest.NewRecorder()

	logs := captureOutboundHandlerLogs(func() {
		webSearchHandlerWithDependencies(
			newImmediateWebSearchOutbound(client),
			newRateLimiter(),
		)(w, req)
	})
	assertSecretsAbsent(t, []string{logs, w.Body.String()},
		"query-secret", "upstream-body-secret")
}
