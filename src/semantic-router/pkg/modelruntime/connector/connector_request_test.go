package connector

import (
	"context"
	"crypto/tls"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestDoRequestAppliesPerCallHeadersBeforeAuthorize(t *testing.T) {
	var seen http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	authorize := func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer fixed")
		return nil
	}
	client, err := New(server.URL, authorize, testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	result, err := client.DoRequest(context.Background(), testOperation, Request{
		Body: []byte(`{}`),
		Headers: map[string]string{
			"X-Request-Id":  "shadow-1",
			"Authorization": "Bearer per-call",
			"Content-Type":  "application/vnd.custom+json",
		},
	})
	if err != nil {
		t.Fatalf("DoRequest() error = %v", err)
	}
	if string(result.Body) != `{"ok":true}` || result.StatusCode != http.StatusOK || result.Attempts != 1 {
		t.Fatalf("result = %+v", result)
	}
	if seen.Get("X-Request-Id") != "shadow-1" {
		t.Fatalf("per-call header not sent: %v", seen)
	}
	if seen.Get("Content-Type") != "application/vnd.custom+json" {
		t.Fatalf("per-call header did not override default: %v", seen)
	}
	if seen.Get("Authorization") != "Bearer fixed" {
		t.Fatalf("authorize hook must run after per-call headers: %v", seen)
	}
}

func TestDoRequestReportsAttemptsAndStatusOnSuccess(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	options := testOptions()
	options.MaxRetries = 1
	client := newTestClient(t, server, options)

	result, err := client.DoRequest(context.Background(), testOperation, Request{Body: []byte(`{}`)})
	if err != nil {
		t.Fatalf("DoRequest() error = %v", err)
	}
	if result.Attempts != 2 || result.StatusCode != http.StatusAccepted {
		t.Fatalf("result = %+v, want attempts=2 status=202", result)
	}
}

func TestDoRequestMarksOversizedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("x", 2048)))
	}))
	defer server.Close()

	client := newTestClient(t, server, testOptions())
	_, err := client.DoRequest(context.Background(), testOperation, Request{Body: []byte(`{}`)})
	if !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("error = %v, want ErrResponseTooLarge", err)
	}
	var connectorErr *Error
	if !errors.As(err, &connectorErr) || connectorErr.Kind != KindResponse || connectorErr.Retryable {
		t.Fatalf("error = %#v, want non-retryable KindResponse", err)
	}
}

func TestNewHonorsTLSConfig(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	strict := newTestClient(t, server, testOptions())
	if _, err := strict.Do(context.Background(), testOperation, []byte(`{}`)); err == nil {
		t.Fatal("default TLS config must reject a self-signed test certificate")
	}

	options := testOptions()
	options.TLSConfig = &tls.Config{InsecureSkipVerify: true} //nolint:gosec // exercising the explicit opt-in
	relaxed := newTestClient(t, server, options)
	if _, err := relaxed.Do(context.Background(), testOperation, []byte(`{}`)); err != nil {
		t.Fatalf("Do() with TLSConfig error = %v", err)
	}
}
