package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestFetchRawHandlerRejectsUserinfoBeforeRequest(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected request")
	}}
	body, err := json.Marshal(FetchRawRequest{
		URL: "https://user:password@example.com/config.yaml?token=query-secret",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/fetch-raw", bytes.NewReader(body))
	w := httptest.NewRecorder()

	fetchRawHandlerWithClient(client)(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", w.Code, w.Body.String())
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("outbound requests = %d, want zero", got)
	}
	for _, secret := range []string{"user", "password", "query-secret"} {
		if strings.Contains(w.Body.String(), secret) {
			t.Fatalf("error leaked %q: %s", secret, w.Body.String())
		}
	}
}

func TestFetchRawHandlerRejectsOversizedResponse(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(strings.Repeat(
				"x",
				fetchRawMaxSize+1,
			))),
		}, nil
	}}
	body, err := json.Marshal(FetchRawRequest{URL: "https://example.com/config.yaml"})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/fetch-raw", bytes.NewReader(body))
	w := httptest.NewRecorder()

	fetchRawHandlerWithClient(client)(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", w.Code, w.Body.String())
	}
	if strings.Contains(w.Body.String(), strings.Repeat("x", 32)) {
		t.Fatalf("oversized response leaked into error: %s", w.Body.String())
	}
}

func TestFetchRawHandlerDoesNotExposeClientErrors(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("https://user:pass@example.com/a?token=query-secret body-secret")
	}}
	body, err := json.Marshal(FetchRawRequest{URL: "https://example.com/config?token=request-secret"})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/fetch-raw", bytes.NewReader(body))
	w := httptest.NewRecorder()

	fetchRawHandlerWithClient(client)(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", w.Code, w.Body.String())
	}
	for _, secret := range []string{"user:pass", "query-secret", "request-secret", "body-secret"} {
		if strings.Contains(w.Body.String(), secret) {
			t.Fatalf("client error leaked %q: %s", secret, w.Body.String())
		}
	}
}
