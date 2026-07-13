package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"
)

func TestShouldPreferJinaFetch(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		url  string
		req  OpenWebRequest
		want bool
	}{
		{
			name: "ordinary html stays on direct-first path",
			url:  "https://example.com/article",
			req:  OpenWebRequest{},
			want: false,
		},
		{
			name: "explicit force_jina wins",
			url:  "https://example.com/article",
			req: OpenWebRequest{
				ForceJina: true,
			},
			want: true,
		},
		{
			name: "with_images keeps jina path",
			url:  "https://example.com/article",
			req: OpenWebRequest{
				WithImages: true,
			},
			want: true,
		},
		{
			name: "pdf urls keep jina path",
			url:  "https://example.com/guide.pdf",
			req:  OpenWebRequest{},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := shouldPreferJinaFetch(tt.url, tt.req); got != tt.want {
				t.Fatalf("shouldPreferJinaFetch(%q) = %v, want %v", tt.url, got, tt.want)
			}
		})
	}
}

func TestNormalizeOpenWebMaxLength(t *testing.T) {
	t.Parallel()

	if got := normalizeOpenWebMaxLength(0); got != openWebMaxContentLength {
		t.Fatalf("normalizeOpenWebMaxLength(0) = %d, want %d", got, openWebMaxContentLength)
	}

	if got := normalizeOpenWebMaxLength(1200); got != 1200 {
		t.Fatalf("normalizeOpenWebMaxLength(1200) = %d, want 1200", got)
	}

	if got := normalizeOpenWebMaxLength(openWebMaxContentLength + 1); got != openWebMaxContentLength {
		t.Fatalf(
			"normalizeOpenWebMaxLength(max+1) = %d, want %d",
			got,
			openWebMaxContentLength,
		)
	}
}

func TestTruncateOpenWebContentPreservesUTF8(t *testing.T) {
	t.Parallel()

	got, truncated := truncateOpenWebContent("你好世界🌏abc", 4)
	if !truncated {
		t.Fatal("truncateOpenWebContent did not report truncation")
	}
	if !utf8.ValidString(got) {
		t.Fatalf("truncateOpenWebContent returned invalid UTF-8: %q", got)
	}
	if !strings.HasPrefix(got, "你好世界") {
		t.Fatalf("truncateOpenWebContent prefix = %q, want rune boundary prefix", got)
	}
}

func TestOpenWebHandlerDoesNotSetWildcardCORS(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("unexpected request")
	}}
	req := httptest.NewRequest(http.MethodOptions, "/api/tools/open-web", nil)
	w := httptest.NewRecorder()

	openWebHandlerWithClient(client)(w, req)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin = %q, want absent", got)
	}
}

func TestOpenWebHandlerRejectsUserinfoBeforeRequest(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected request")
	}}
	body, err := json.Marshal(OpenWebRequest{
		URL: "https://user:password@example.com/article?token=query-secret",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/open-web", bytes.NewReader(body))
	w := httptest.NewRecorder()

	openWebHandlerWithClient(client)(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", w.Code, w.Body.String())
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("outbound calls = %d, want zero", got)
	}
	for _, secret := range []string{"user", "password", "query-secret"} {
		if strings.Contains(w.Body.String(), secret) {
			t.Fatalf("validation error leaked %q: %s", secret, w.Body.String())
		}
	}
}

func TestOpenWebHandlerRejectsLoopbackInProduction(t *testing.T) {
	t.Parallel()

	body, err := json.Marshal(OpenWebRequest{URL: "http://127.0.0.1/admin", ForceJina: true})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/open-web", bytes.NewReader(body))
	w := httptest.NewRecorder()

	OpenWebHandler()(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", w.Code, w.Body.String())
	}
}

func TestFetchWebDirectRejectsOversizedResponse(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(strings.Repeat(
				"x",
				openWebMaxResponseSize+1,
			))),
		}, nil
	}}

	_, err := fetchWebDirect(
		context.Background(),
		client,
		"https://example.com/article",
		time.Second,
		openWebMaxContentLength,
	)
	if !errors.Is(err, errOutboundResponseTooLarge) {
		t.Fatalf("fetchWebDirect() error = %v, want response too large", err)
	}
}

func TestFetchWebWithJinaRejectsOversizedResponse(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(strings.Repeat(
				"x",
				openWebMaxResponseSize+1,
			))),
		}, nil
	}}

	_, err := fetchWebWithJina(
		context.Background(),
		client,
		"https://example.com/article",
		time.Second,
		"markdown",
		openWebMaxContentLength,
		false,
	)
	if !errors.Is(err, errOutboundResponseTooLarge) {
		t.Fatalf("fetchWebWithJina() error = %v, want response too large", err)
	}
}

func TestOpenWebHandlerDoesNotExposeClientErrorsOrURLQuery(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return nil, errors.New("https://user:pass@example.com/a?token=query-secret body-secret")
	}}
	body, err := json.Marshal(OpenWebRequest{
		URL: "https://example.com/article?token=request-secret",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/open-web", bytes.NewReader(body))
	w := httptest.NewRecorder()

	openWebHandlerWithClient(client)(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", w.Code, w.Body.String())
	}
	for _, secret := range []string{"user:pass", "query-secret", "request-secret", "body-secret"} {
		if strings.Contains(w.Body.String(), secret) {
			t.Fatalf("fetch error leaked %q: %s", secret, w.Body.String())
		}
	}
}
