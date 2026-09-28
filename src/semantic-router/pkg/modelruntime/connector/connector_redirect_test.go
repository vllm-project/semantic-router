package connector

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// TestDoRequestNeverFollowsRedirects proves a redirecting remote cannot move
// the request body or its credential headers to another origin: the redirect
// target receives nothing, the failure is classified as a redirect, and no
// retry is attempted even for a retry-safe operation.
func TestDoRequestNeverFollowsRedirects(t *testing.T) {
	var leaked atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		leaked.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		_, _ = w.Write([]byte(`{"stolen":true}`))
	}))
	defer other.Close()

	var configuredHits atomic.Int32
	var location string
	var status int
	configured := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		configuredHits.Add(1)
		if r.Header.Get("Authorization") != "Bearer secret" || r.Header.Get("X-Api-Key") != "secret" {
			t.Errorf("configured origin did not receive the credential headers: %v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		if string(body) != `{"prompt":"private"}` {
			t.Errorf("configured origin received body %q", body)
		}
		w.Header().Set("Location", location)
		w.WriteHeader(status)
	}))
	defer configured.Close()

	authorize := func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer secret")
		return nil
	}
	options := testOptions()
	options.MaxRetries = 2
	client, err := New(configured.URL, authorize, options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name     string
		status   int
		location string
	}{
		{name: "cross-origin 307", status: http.StatusTemporaryRedirect, location: other.URL + "/v1/chat/completions"},
		{name: "cross-origin 308", status: http.StatusPermanentRedirect, location: other.URL + "/classify"},
		{name: "cross-origin 302", status: http.StatusFound, location: other.URL + "/classify"},
		{name: "same-origin 307", status: http.StatusTemporaryRedirect, location: configured.URL + "/elsewhere"},
		{name: "relative 301", status: http.StatusMovedPermanently, location: "/elsewhere"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configuredHits.Store(0)
			leaked.Store(0)
			location = tt.location
			status = tt.status

			_, err := client.DoRequest(context.Background(), testOperation, Request{
				Body:    []byte(`{"prompt":"private"}`),
				Headers: map[string]string{"X-Api-Key": "secret"},
			})
			var connectorErr *Error
			if !errors.As(err, &connectorErr) {
				t.Fatalf("DoRequest() error = %v, want *Error", err)
			}
			if connectorErr.Kind != KindRedirect || connectorErr.StatusCode != tt.status {
				t.Fatalf("error = %+v, want KindRedirect with status %d", connectorErr, tt.status)
			}
			if !errors.Is(err, ErrRedirectRejected) {
				t.Fatalf("error %v does not wrap ErrRedirectRejected", err)
			}
			if connectorErr.Retryable || connectorErr.Attempt != 1 {
				t.Fatalf("redirect must fail on the first attempt without retry: %+v", connectorErr)
			}
			if got := configuredHits.Load(); got != 1 {
				t.Fatalf("configured origin hits = %d, want 1", got)
			}
			if got := leaked.Load(); got != 0 {
				t.Fatalf("redirect target received %d request(s); body or credentials left the configured origin", got)
			}
		})
	}
}

func TestErrorDoesNotExposeRedirectPath(t *testing.T) {
	err := &Error{
		Kind:       KindRedirect,
		Operation:  "classify",
		StatusCode: http.StatusTemporaryRedirect,
		Attempt:    1,
		Cause:      errors.New("redirect rejected: http://other.example"),
	}
	if got := err.Error(); got != `connector operation "classify" failed on attempt 1 with HTTP status 307` {
		t.Fatalf("Error() = %q", got)
	}
}

func TestRedirectTargetReducesLocationToOrigin(t *testing.T) {
	tests := []struct {
		name     string
		location string
		want     string
	}{
		{name: "absolute", location: "https://other.example:8443/v1/chat?key=x", want: "https://other.example:8443"},
		{name: "relative", location: "/v1/chat", want: "relative location"},
		{name: "missing", location: "", want: "no location"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := &http.Response{Header: http.Header{}}
			if tt.location != "" {
				response.Header.Set("Location", tt.location)
			}
			if got := redirectTarget(response); got != tt.want {
				t.Fatalf("redirectTarget() = %q, want %q", got, tt.want)
			}
		})
	}
}
