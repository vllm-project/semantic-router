package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	dashboardE2EAdminEmail    = "e2e-dashboard-admin@example.com"
	dashboardE2EAdminPassword = "dashboard-e2e-password"
)

func dashboardAuthToken(ctx context.Context, client *http.Client, baseURL string, verbose bool) (string, error) {
	body, err := json.Marshal(map[string]string{
		"email":    dashboardE2EAdminEmail,
		"password": dashboardE2EAdminPassword,
	})
	if err != nil {
		return "", fmt.Errorf("marshal login payload: %w", err)
	}

	url := strings.TrimRight(baseURL, "/") + "/api/auth/login"
	if verbose {
		fmt.Printf("[Dashboard] POST %s (auth login)\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create login request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-VSR-Auth-Mode", "bearer")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("login request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("login: expected 200, got %d: %s", resp.StatusCode, truncateString(string(respBody), 200))
	}

	var result struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("login response is not valid JSON: %w", err)
	}
	if strings.TrimSpace(result.Token) == "" {
		return "", fmt.Errorf("login response did not include a token")
	}

	return result.Token, nil
}

func setDashboardAuth(req *http.Request, token string) {
	req.Header.Set("Authorization", "Bearer "+token)
}

func newAuthenticatedDashboardRequest(
	ctx context.Context,
	client *http.Client,
	method string,
	url string,
	body io.Reader,
	baseURL string,
	verbose bool,
) (*http.Request, error) {
	token, err := dashboardAuthToken(ctx, client, baseURL, verbose)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, err
	}
	setDashboardAuth(req, token)
	return req, nil
}
