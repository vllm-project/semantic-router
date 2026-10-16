package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestVLLMClientAllowlistedRejectionReasonReplacesByteCount(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"The model does not support response_format json_schema","type":"invalid_request_error"}}`))
	}))
	defer server.Close()

	client := newLimitedVLLMClient(server, 8192)
	_, err := client.Generate(context.Background(), "classifier", "test", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	message := err.Error()
	if !strings.Contains(message, "status 400") ||
		!strings.Contains(message, "the classifier backend rejected the response_format this client requires") {
		t.Fatalf("error = %v, want status and the fixed reason", message)
	}
	if strings.Contains(message, "json_schema") || strings.Contains(message, "invalid_request_error") {
		t.Fatalf("error = %v, want no remote body text", message)
	}
	if strings.Contains(message, "not logged") {
		t.Fatalf("error = %v, want the allowlisted form rather than the byte-count form", message)
	}
}

func TestVLLMClientRejectionBodyWithUserTextAndSecretsStaysUnlogged(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"unprocessable: the submitted prompt contract sk-live-abcdef1234567890 and the account password hunter2"}}`))
	}))
	defer server.Close()

	client := newLimitedVLLMClient(server, 8192)
	_, err := client.Generate(context.Background(), "classifier", "test", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	message := err.Error()
	if !strings.Contains(message, "status 400") || !strings.Contains(message, "not logged") {
		t.Fatalf("error = %v, want the byte-count form", message)
	}
	for _, secret := range []string{"sk-live-abcdef1234567890", "hunter2", "prompt contract"} {
		if strings.Contains(message, secret) {
			t.Fatalf("error = %v, leaked %q", message, secret)
		}
	}
}

func TestVLLMClientRejectionReasonIgnoresBodiesWithoutAnEnvelope(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("<html>502 bad gateway page</html>"))
	}))
	defer server.Close()

	client := newLimitedVLLMClient(server, 8192)
	_, err := client.Generate(context.Background(), "classifier", "test", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	message := err.Error()
	if !strings.Contains(message, "status 400") || !strings.Contains(message, "not logged") {
		t.Fatalf("error = %v, want the byte-count form", message)
	}
	if strings.Contains(message, "bad gateway page") {
		t.Fatalf("error = %v, want no remote body text", message)
	}
}
