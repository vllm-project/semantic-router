package selection

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var (
	_ func(string) *RouterR1Client        = NewRouterR1Client
	_ func(string) *AutoMixVerifierClient = NewAutoMixVerifierClient
)

func TestRouterR1ClientRejectsOversizedResponse(t *testing.T) {
	body, err := json.Marshal(RouterR1Response{SelectedModel: "model-a", Thinking: strings.Repeat("x", 64)})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer server.Close()

	client := NewRouterR1Client(server.URL)
	client.maxResponseBytes = int64(len(body))
	if _, routeErr := client.Route(context.Background(), "route this"); routeErr != nil {
		t.Fatalf("Route() at limit error = %v", routeErr)
	}
	client.maxResponseBytes = int64(len(body) - 1)
	_, err = client.Route(context.Background(), "route this")
	if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
		t.Fatalf("Route() error = %v, want response limit error", err)
	}
}

func TestRouterR1ClientTruncatesErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(strings.Repeat("x", int(maxSelectionErrorBodyBytes+1))))
	}))
	defer server.Close()

	_, err := NewRouterR1Client(server.URL).Route(context.Background(), "route this")
	if err == nil || !strings.Contains(err.Error(), "truncated=true") {
		t.Fatalf("Route() error = %v, want truncated error body", err)
	}
}

func TestAutoMixVerifierClientRejectsOversizedResponse(t *testing.T) {
	body, err := json.Marshal(AutoMixVerifyResponse{Confidence: 0.9, Samples: []string{strings.Repeat("x", 64)}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer server.Close()

	client := NewAutoMixVerifierClient(server.URL)
	client.SetMaxResponseBytes(int64(len(body)))
	if _, verifyErr := client.Verify(context.Background(), "question", "answer", "", 0.7); verifyErr != nil {
		t.Fatalf("Verify() at limit error = %v", verifyErr)
	}
	client.SetMaxResponseBytes(int64(len(body) - 1))
	_, err = client.Verify(context.Background(), "question", "answer", "", 0.7)
	if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
		t.Fatalf("Verify() error = %v, want response limit error", err)
	}
}
