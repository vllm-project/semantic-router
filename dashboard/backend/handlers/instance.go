package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"time"
)

const instanceSocketEnv = "VLLM_SR_INSTANCE_SOCKET"

// instanceRequest can reach only the CLI-owned Unix socket. Browsers cannot
// choose a host, socket, image or container-runtime command.
func instanceRequest(ctx context.Context, method, path string, body []byte) (int, []byte, error) {
	socket := os.Getenv(instanceSocketEnv)
	if socket == "" {
		return 0, nil, errors.New("instance controller is not configured")
	}
	transport := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{Timeout: 2 * time.Second}).DialContext(ctx, "unix", socket)
	}}
	defer transport.CloseIdleConnections()
	client := &http.Client{
		Transport: transport, Timeout: 35 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	request, err := http.NewRequestWithContext(ctx, method, "http://instance"+path, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return 0, nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, decisionModelResponseLimit+1))
	if err == nil && len(data) > decisionModelResponseLimit {
		err = errors.New("instance response too large")
	}
	return response.StatusCode, data, err
}

func instanceEngineActive(ctx context.Context) bool {
	if os.Getenv(instanceSocketEnv) == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	code, data, err := instanceRequest(ctx, http.MethodGet, "/status", nil)
	var state struct {
		ObservedMode string `json:"observed_mode"`
	}
	return err == nil && code == http.StatusOK && json.Unmarshal(data, &state) == nil && state.ObservedMode == "engine"
}

// InstanceHandler observes the lifecycle selected at startup. Mode changes
// belong to the serving command, not the Dashboard control plane.
func InstanceHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "Instance mode is selected at startup", http.StatusMethodNotAllowed)
			return
		}
		var path string
		switch r.URL.Path {
		case "/api/instance":
			path = "/status"
		case "/api/instance/models":
			path = "/models"
		default:
			http.NotFound(w, r)
			return
		}
		code, data, err := instanceRequest(r.Context(), http.MethodGet, path, nil)
		if err != nil {
			if path == "/status" {
				ownership := "external"
				if os.Getenv("KUBERNETES_SERVICE_HOST") != "" {
					ownership = "kubernetes"
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"ownership": ownership, "controller_available": false, "observed_mode": "unknown", "operation": nil, "unavailable_reason": "Instance lifecycle is managed by its deployment owner"})
				return
			}
			http.Error(w, "Instance controller unavailable", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(code)
		_, _ = w.Write(data)
	}
}
