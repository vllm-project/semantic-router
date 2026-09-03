package soak

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClientRequestsBufferedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		assertStreamMode(t, req, false)
		if got := req.Header.Get("Accept"); got != "" {
			t.Errorf("Accept = %q, want empty", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "MoM", 1, 1, false)
	if err := client.Chat(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestClientRequestsStreamingResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		assertStreamMode(t, req, true)
		if got := req.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("Accept = %q, want text/event-stream", got)
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = w.Write([]byte("data: {\"choices\":[]}\n\ndata: [DONE]\n\n"))
	}))
	defer server.Close()

	client := NewClient(server.URL, "MoM", 1, 1, true)
	if err := client.Chat(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func assertStreamMode(t *testing.T, req *http.Request, want bool) {
	t.Helper()
	var payload map[string]any
	if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
		t.Errorf("decode request: %v", err)
		return
	}
	if got, _ := payload["stream"].(bool); got != want {
		t.Errorf("stream = %v, want %v", got, want)
	}
}

func TestStreamingClientRejectsBufferedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer server.Close()

	client := NewClient(server.URL, "MoM", 1, 1, true)
	err := client.Chat(t.Context())
	if err == nil || !strings.Contains(err.Error(), "unexpected content-type") {
		t.Fatalf("Chat() error = %v, want unexpected content-type", err)
	}
}
