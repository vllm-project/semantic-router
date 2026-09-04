package connector

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

var testOperation = Operation{
	Name:      "classify",
	Method:    http.MethodPost,
	Path:      "/classify",
	RetrySafe: true,
}

func testOptions() Options {
	return Options{
		AttemptTimeout:   time.Second,
		MaxRetries:       0,
		MaxRequestBytes:  1024,
		MaxResponseBytes: 1024,
		MaxErrorBytes:    64,
	}
}

func newTestClient(t *testing.T, server *httptest.Server, options Options) *Client {
	t.Helper()
	client, err := New(server.URL, nil, options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func TestNewValidatesEndpointAndBounds(t *testing.T) {
	tests := []struct {
		name    string
		baseURL string
		options Options
	}{
		{name: "relative endpoint", baseURL: "localhost:8080", options: testOptions()},
		{name: "unsupported scheme", baseURL: "grpc://localhost:8080", options: testOptions()},
		{name: "missing attempt timeout", baseURL: "http://localhost:8080", options: func() Options {
			options := testOptions()
			options.AttemptTimeout = 0
			return options
		}()},
		{name: "negative retries", baseURL: "http://localhost:8080", options: func() Options {
			options := testOptions()
			options.MaxRetries = -1
			return options
		}()},
		{name: "missing body bound", baseURL: "http://localhost:8080", options: func() Options {
			options := testOptions()
			options.MaxResponseBytes = 0
			return options
		}()},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := New(test.baseURL, nil, test.options); err == nil {
				t.Fatal("New() error = nil, want validation failure")
			}
		})
	}
}

func TestDoAppliesAuthAndPreservesBasePath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/models/classify" {
			t.Errorf("path = %q, want /models/classify", request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer token" {
			t.Errorf("Authorization = %q", got)
		}
		if got := request.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
		body, _ := io.ReadAll(request.Body)
		if string(body) != `{"input":"hello"}` {
			t.Errorf("body = %q", body)
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	client, err := New(server.URL+"/models", func(_ context.Context, request *http.Request) error {
		request.Header.Set("Authorization", "Bearer token")
		return nil
	}, testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer client.Close()

	body, err := client.Do(context.Background(), testOperation, []byte(`{"input":"hello"}`))
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if string(body) != `{"ok":true}` {
		t.Fatalf("Do() body = %q", body)
	}
}

func TestDoRejectsOversizedRequestBeforeSending(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("server received an oversized request")
	}))
	defer server.Close()
	options := testOptions()
	options.MaxRequestBytes = 3
	client := newTestClient(t, server, options)

	_, err := client.Do(context.Background(), testOperation, []byte("four"))
	assertConnectorError(t, err, KindRequest, 0, false)
}

func TestDoRejectsOversizedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("12345"))
	}))
	defer server.Close()
	options := testOptions()
	options.MaxResponseBytes = 4
	client := newTestClient(t, server, options)

	_, err := client.Do(context.Background(), testOperation, nil)
	connectorErr := assertConnectorError(t, err, KindResponse, 1, false)
	if !strings.Contains(connectorErr.Error(), "exceeds limit") {
		t.Fatalf("error = %v, want response limit", connectorErr)
	}
}

func TestDoBoundsStatusBodyWithoutExposingItInError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("secret-response"))
	}))
	defer server.Close()
	options := testOptions()
	options.MaxErrorBytes = 6
	client := newTestClient(t, server, options)

	_, err := client.Do(context.Background(), testOperation, nil)
	connectorErr := assertConnectorError(t, err, KindStatus, 1, false)
	if connectorErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d", connectorErr.StatusCode)
	}
	body, truncated := connectorErr.ResponseBody()
	if string(body) != "secret" || !truncated {
		t.Fatalf("ResponseBody() = %q, %t", body, truncated)
	}
	if strings.Contains(connectorErr.Error(), "secret") {
		t.Fatalf("Error() exposed response body: %v", connectorErr)
	}
	body[0] = 'X'
	bodyAgain, _ := connectorErr.ResponseBody()
	if string(bodyAgain) != "secret" {
		t.Fatal("ResponseBody() did not return a defensive copy")
	}
}

func TestDoRetriesOnlyRetrySafeOperations(t *testing.T) {
	for _, test := range []struct {
		name         string
		retrySafe    bool
		wantAttempts int32
		wantError    bool
	}{
		{name: "retry safe", retrySafe: true, wantAttempts: 2},
		{name: "not retry safe", retrySafe: false, wantAttempts: 1, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			testRetrySafety(t, test.retrySafe, test.wantAttempts, test.wantError)
		})
	}
}

func testRetrySafety(t *testing.T, retrySafe bool, wantAttempts int32, wantError bool) {
	t.Helper()
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		attempt := attempts.Add(1)
		body, _ := io.ReadAll(request.Body)
		if string(body) != "payload" {
			t.Errorf("attempt %d body = %q", attempt, body)
		}
		if attempt == 1 {
			http.Error(w, "try again", http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()
	options := testOptions()
	options.MaxRetries = 1
	client := newTestClient(t, server, options)
	operation := testOperation
	operation.RetrySafe = retrySafe

	body, err := client.Do(context.Background(), operation, []byte("payload"))
	if wantError && err == nil {
		t.Fatal("Do() error = nil")
	}
	if !wantError && (err != nil || string(body) != "ok") {
		t.Fatalf("Do() = %q, %v", body, err)
	}
	if got := attempts.Load(); got != wantAttempts {
		t.Fatalf("attempts = %d, want %d", got, wantAttempts)
	}
}

func TestDoAuthorizesEveryAttempt(t *testing.T) {
	var attempts atomic.Int32
	var authorizations atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()
	options := testOptions()
	options.MaxRetries = 1
	client, err := New(server.URL, func(context.Context, *http.Request) error {
		authorizations.Add(1)
		return nil
	}, options)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if _, err := client.Do(context.Background(), testOperation, nil); err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	if got := authorizations.Load(); got != 2 {
		t.Fatalf("authorization calls = %d, want 2", got)
	}
}

func TestDoHonorsCallerCancellationDuringRetryDelay(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	options := testOptions()
	options.MaxRetries = 5
	client := newTestClient(t, server, options)
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(10*time.Millisecond, cancel)

	_, err := client.Do(ctx, testOperation, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Do() error = %v, want context.Canceled", err)
	}
}

func TestDoReturnsAuthorizationFailureWithoutSending(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("server received a request after authorization failed")
	}))
	defer server.Close()
	want := errors.New("credential unavailable")
	client, err := New(server.URL, func(context.Context, *http.Request) error {
		return want
	}, testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	_, err = client.Do(context.Background(), testOperation, nil)
	assertConnectorError(t, err, KindAuthorization, 1, false)
	if !errors.Is(err, want) {
		t.Fatalf("Do() error = %v, want wrapped authorization error", err)
	}
}

func assertConnectorError(
	t *testing.T,
	err error,
	kind ErrorKind,
	attempt int,
	retryable bool,
) *Error {
	t.Helper()
	var connectorErr *Error
	if !errors.As(err, &connectorErr) {
		t.Fatalf("error = %T %v, want *connector.Error", err, err)
	}
	if connectorErr.Kind != kind || connectorErr.Attempt != attempt || connectorErr.Retryable != retryable {
		t.Fatalf(
			"error = %+v, want kind=%s attempt=%d retryable=%t",
			connectorErr, kind, attempt, retryable,
		)
	}
	return connectorErr
}
