package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

const (
	SRBenchAPIPath          = "/api/sr-bench/v1"
	maxSRBenchRequestBytes  = 8 << 20
	maxSRBenchResponseBytes = 64 << 20
)

// SRBenchHandler is an authenticated transport to the independent sr-bench
// service. It owns no worker, run state, or process lifecycle.
type SRBenchHandler struct {
	origin   *url.URL
	token    string
	readonly bool
	client   *http.Client
}

func NewSRBenchHandler(origin, token string, readonly bool) (*SRBenchHandler, error) {
	if err := config.ValidateSRBenchConfig(origin, "SR_BENCH_TOKEN"); err != nil {
		return nil, err
	}
	parsed, _ := url.Parse(origin)
	return &SRBenchHandler{
		origin: parsed, token: token, readonly: readonly,
		client: &http.Client{
			Timeout:       30 * time.Second,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}, nil
}

func (h *SRBenchHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "private, no-store")
	w.Header().Set("Pragma", "no-cache")
	method, known := srBenchRouteMethod(r.URL.Path)
	if !known || r.URL.RawPath != "" {
		writeSRBenchError(w, http.StatusNotFound, "sr-bench endpoint not found")
		return
	}
	if method == "" && (r.Method == http.MethodGet || r.Method == http.MethodPost) {
		method = r.Method
	}
	if r.Method != method {
		if method == "" {
			method = "GET, POST"
		}
		w.Header().Set("Allow", method)
		writeSRBenchError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}
	actor, ok := dashboardauth.AuthFromContext(r)
	if !ok || actor.UserID == "" {
		writeSRBenchError(w, http.StatusUnauthorized, "Authentication is required")
		return
	}
	if h.readonly && r.Method != http.MethodGet {
		writeSRBenchError(w, http.StatusForbidden, "Dashboard is read-only")
		return
	}
	if strings.TrimSpace(h.token) == "" {
		writeSRBenchError(w, http.StatusServiceUnavailable, "sr-bench service authentication is not configured")
		return
	}
	h.forward(w, r, actor)
}

func (h *SRBenchHandler) forward(w http.ResponseWriter, r *http.Request, actor dashboardauth.AuthContext) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxSRBenchRequestBytes+1))
	if err != nil || len(body) > maxSRBenchRequestBytes {
		writeSRBenchError(w, http.StatusRequestEntityTooLarge, "sr-bench request exceeds the size limit")
		return
	}
	upstream := *h.origin
	upstream.Path = r.URL.Path
	upstream.RawQuery = r.URL.RawQuery
	request, err := http.NewRequestWithContext(r.Context(), r.Method, upstream.String(), bytes.NewReader(body))
	if err != nil {
		writeSRBenchError(w, http.StatusBadGateway, "sr-bench request could not be forwarded")
		return
	}
	// A fresh header set excludes browser credentials, forged actor headers,
	// forwarding headers, cookies, and caller-selected authorization tokens.
	request.Header.Set("Authorization", "Bearer "+h.token)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	request.Header.Set("X-SR-Bench-Actor-ID", actor.UserID)
	request.Header.Set("X-SR-Bench-Actor-Role", actor.Role)
	response, err := h.client.Do(request)
	if err != nil {
		writeSRBenchError(w, http.StatusBadGateway, "sr-bench service is unavailable; saved runs remain owned by the service")
		return
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxSRBenchResponseBytes+1))
	if err != nil || len(data) > maxSRBenchResponseBytes || !json.Valid(data) || response.StatusCode >= 300 && response.StatusCode < 400 {
		writeSRBenchError(w, http.StatusBadGateway, "sr-bench service returned an invalid response")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(response.StatusCode)
	_, _ = w.Write(data)
}

func srBenchRouteMethod(path string) (string, bool) {
	rest := strings.TrimPrefix(path, SRBenchAPIPath)
	if rest == path {
		return "", false
	}
	switch rest {
	case "/health", "/catalog", "/datasets", "/targets":
		return http.MethodGet, true
	case "/plans", "/comparisons", "/replays":
		return http.MethodPost, true
	case "/runs":
		// GET and POST are the only collection methods; the caller selects
		// between them in ServeHTTP before forwarding.
		return "", true
	}
	parts := strings.Split(strings.TrimPrefix(rest, "/"), "/")
	if len(parts) < 2 || parts[0] != "runs" || !validSRBenchRunID(parts[1]) {
		return "", false
	}
	if len(parts) == 2 {
		return http.MethodGet, true
	}
	if len(parts) == 3 {
		switch parts[2] {
		case "results", "report", "events", "calls":
			return http.MethodGet, true
		case "cancel", "regrade", "export", "recover-plan", "recover", "reconcile-usage":
			return http.MethodPost, true
		}
	}
	if len(parts) == 4 && parts[2] == "calls" && validSRBenchRunID(parts[3]) {
		return http.MethodGet, true
	}
	return "", false
}

func validSRBenchRunID(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for _, char := range value {
		switch {
		case char >= 'a' && char <= 'z', char >= 'A' && char <= 'Z', char >= '0' && char <= '9', char == '-', char == '_':
		default:
			return false
		}
	}
	return true
}

func writeSRBenchError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"message": message}})
}
