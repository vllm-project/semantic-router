package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestQueryWorkerChat_ReportsActualEndpointFailures(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	attemptedPaths := make([]string, 0, len(workerChatEndpointCandidates))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptedPaths = append(attemptedPaths, r.URL.Path)
		switch r.URL.Path {
		case "/v1/chat/completions":
			http.Error(w, "primary endpoint failed", http.StatusBadGateway)
		case "/api/openai/v1/chat/completions":
			http.NotFound(w, r)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	worker := ContainerEntry{
		Name:     "mira",
		Port:     mustServerPort(t, srv.URL),
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "test-token",
		DataDir:  tempDir,
		RoleKind: "worker",
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	_, err := h.queryWorkerChat(worker, "system", "user")
	if err == nil {
		t.Fatalf("expected queryWorkerChat to fail")
	}
	if !strings.Contains(err.Error(), "/v1/chat/completions") {
		t.Fatalf("expected primary endpoint to appear in error, got %v", err)
	}
	if !strings.Contains(err.Error(), "/api/openai/v1/chat/completions") {
		t.Fatalf("expected fallback endpoint to appear in error, got %v", err)
	}
	if strings.Contains(err.Error(), "/api/router/v1/chat/completions") {
		t.Fatalf("dashboard router path should not be probed as a worker endpoint, got %v", err)
	}
	if len(attemptedPaths) != len(workerChatEndpointCandidates) {
		t.Fatalf("expected %d endpoint attempts, got %d (%v)", len(workerChatEndpointCandidates), len(attemptedPaths), attemptedPaths)
	}
}

func TestQueryWorkerChatStream_ReportsActualEndpointFailures(t *testing.T) {
	tempDir := t.TempDir()
	h := NewOpenClawHandler(tempDir, false)

	attemptedPaths := make([]string, 0, len(workerChatEndpointCandidates))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attemptedPaths = append(attemptedPaths, r.URL.Path)
		switch r.URL.Path {
		case "/v1/chat/completions":
			http.Error(w, "stream endpoint failed", http.StatusBadGateway)
		case "/api/openai/v1/chat/completions":
			http.NotFound(w, r)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	worker := ContainerEntry{
		Name:     "tariq",
		Port:     mustServerPort(t, srv.URL),
		Image:    "ghcr.io/openclaw/openclaw:latest",
		Token:    "test-token",
		DataDir:  tempDir,
		RoleKind: "worker",
	}
	if err := h.saveRegistry([]ContainerEntry{worker}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	_, err := h.queryWorkerChatStreamWithMessages(
		worker,
		"room:tariq",
		[]openAIChatMessage{{Role: "user", Content: "hello"}},
		nil,
	)
	if err == nil {
		t.Fatalf("expected queryWorkerChatStreamWithMessages to fail")
	}
	if !strings.Contains(err.Error(), "/v1/chat/completions") {
		t.Fatalf("expected primary endpoint to appear in stream error, got %v", err)
	}
	if !strings.Contains(err.Error(), "/api/openai/v1/chat/completions") {
		t.Fatalf("expected fallback endpoint to appear in stream error, got %v", err)
	}
	if strings.Contains(err.Error(), "/api/router/v1/chat/completions") {
		t.Fatalf("dashboard router path should not be probed as a worker stream endpoint, got %v", err)
	}
	if len(attemptedPaths) != len(workerChatEndpointCandidates) {
		t.Fatalf("expected %d stream endpoint attempts, got %d (%v)", len(workerChatEndpointCandidates), len(attemptedPaths), attemptedPaths)
	}
}
