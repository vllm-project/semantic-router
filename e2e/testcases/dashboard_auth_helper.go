package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const (
	defaultDashboardAdminEmail    = "admin@e2e.local"
	defaultDashboardAdminPassword = "e2e-admin-password"
)

func dashboardAdminCredentials() (email, password string) {
	email = os.Getenv("E2E_DASHBOARD_ADMIN_EMAIL")
	if email == "" {
		email = defaultDashboardAdminEmail
	}
	password = os.Getenv("E2E_DASHBOARD_ADMIN_PASSWORD")
	if password == "" {
		password = defaultDashboardAdminPassword
	}
	return email, password
}

type bearerTransport struct {
	base  http.RoundTripper
	token string
}

func (t *bearerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set("Authorization", "Bearer "+t.token)
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(clone)
}

func loginDashboard(ctx context.Context, baseURL string, timeout time.Duration) (string, error) {
	email, password := dashboardAdminCredentials()
	payload, err := json.Marshal(map[string]string{"email": email, "password": password})
	if err != nil {
		return "", fmt.Errorf("marshal login payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/auth/login", bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("create login request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return "", fmt.Errorf("login request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("login: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var parsed struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("login response is not valid JSON: %w (body: %s)", err, truncateString(string(body), 200))
	}
	if parsed.Token == "" {
		return "", fmt.Errorf("login response missing token (body: %s)", truncateString(string(body), 200))
	}
	return parsed.Token, nil
}

func newAuthenticatedDashboardClient(ctx context.Context, baseURL string, timeout time.Duration) (*http.Client, error) {
	token, err := loginDashboard(ctx, baseURL, timeout)
	if err != nil {
		return nil, err
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: &bearerTransport{base: http.DefaultTransport, token: token},
	}, nil
}
