package fixtures

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	DashboardAdminEmail    = "e2e-dashboard-admin@example.com"
	DashboardAdminPassword = "dashboard-e2e-password"
	DashboardAdminName     = "Dashboard E2E Admin"
	dashboardSessionCookie = "vsr_session"
	dashboardManagementV1  = "application/vnd.vllm-semantic-router.management.v1+json"
)

type dashboardIdentity struct {
	Namespaces []struct {
		Namespace struct {
			NamespaceID     string `json:"namespaceId"`
			DesiredRevision uint64 `json:"desiredRevision"`
			AppliedRevision uint64 `json:"appliedRevision"`
		} `json:"namespace"`
	} `json:"namespaces"`
}

// EnsureDashboardAdmin completes the idempotent first-install Dashboard flow
// and returns the authenticated session. With durable routing this is also the
// Management action that creates the first Router publication.
func EnsureDashboardAdmin(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	verbose bool,
) (string, error) {
	loginPayload := map[string]string{
		"email": DashboardAdminEmail, "password": DashboardAdminPassword,
	}
	token, status, responseBody, err := dashboardSessionRequest(
		ctx, client, baseURL, "/api/auth/login", loginPayload, verbose,
	)
	if err != nil {
		return "", err
	}
	if status == http.StatusOK {
		return token, nil
	}
	if status != http.StatusUnauthorized {
		return "", fmt.Errorf("login: expected 200, got %d: %s", status, boundedText(responseBody, 200))
	}
	return registerDashboardAdmin(ctx, client, baseURL, loginPayload, verbose)
}

func registerDashboardAdmin(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	loginPayload map[string]string,
	verbose bool,
) (string, error) {
	bootstrapPayload := map[string]string{
		"email": DashboardAdminEmail, "password": DashboardAdminPassword,
		"name": DashboardAdminName,
	}
	deadline := time.Now().Add(90 * time.Second)
	var responseBody []byte
	for time.Now().Before(deadline) {
		token, status, body, err := dashboardSessionRequest(
			ctx, client, baseURL, "/api/auth/bootstrap/register", bootstrapPayload, verbose,
		)
		if err != nil {
			return "", err
		}
		responseBody = body
		if status == http.StatusOK {
			return token, nil
		}
		if status == http.StatusConflict {
			token, status, responseBody, err = dashboardSessionRequest(
				ctx, client, baseURL, "/api/auth/login", loginPayload, verbose,
			)
		}
		if err != nil {
			return "", err
		}
		if status == http.StatusOK {
			return token, nil
		}
		if status != http.StatusServiceUnavailable &&
			status != http.StatusConflict &&
			status != http.StatusUnauthorized {
			return "", fmt.Errorf("bootstrap registration: got %d: %s", status, boundedText(responseBody, 200))
		}
		if err := waitDashboardRetry(ctx, 2*time.Second); err != nil {
			return "", err
		}
	}
	return "", fmt.Errorf(
		"Dashboard first-administrator installation did not become ready: %s",
		boundedText(responseBody, 200),
	)
}

func waitDashboardRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	select {
	case <-ctx.Done():
		timer.Stop()
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// WaitForFirstRouterPublication verifies that the Router-backed Dashboard
// identity observes one fully applied namespace revision. This is the explicit
// coupled policy-and-routing publication gate for managed profile setup.
func WaitForFirstRouterPublication(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	sessionCookie string,
	timeout time.Duration,
	verbose bool,
) error {
	if client == nil || strings.TrimSpace(sessionCookie) == "" || timeout <= 0 {
		return errors.New("dashboard client, session cookie, and publication timeout are required")
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/api/router/management/v1/me"
	deadline := time.Now().Add(timeout)
	lastReason := "Router-backed identity is not visible"
	for time.Now().Before(deadline) {
		if verbose {
			fmt.Printf("[Dashboard] GET %s (first publication)\n", endpoint)
		}
		response, err := DoGETRequestWithHeaders(ctx, client, endpoint, map[string]string{
			"Accept": dashboardManagementV1,
			"Cookie": (&http.Cookie{
				Name: dashboardSessionCookie, Value: sessionCookie,
			}).String(),
		})
		published, reason, err := inspectDashboardPublication(response, err)
		if err != nil {
			return err
		}
		if published {
			return nil
		}
		lastReason = reason
		if err := waitDashboardRetry(ctx, time.Second); err != nil {
			return err
		}
	}
	return fmt.Errorf("first coupled Router publication did not apply: %s", lastReason)
}

func inspectDashboardPublication(response *HTTPResponse, requestErr error) (bool, string, error) {
	if requestErr != nil {
		return false, requestErr.Error(), nil
	}
	if response.StatusCode == http.StatusServiceUnavailable {
		return false, boundedText(response.Body, 200), nil
	}
	if response.StatusCode != http.StatusOK {
		return false, "", fmt.Errorf(
			"read Router-backed Dashboard identity: status %d: %s",
			response.StatusCode,
			boundedText(response.Body, 200),
		)
	}
	var identity dashboardIdentity
	if err := response.DecodeJSON(&identity); err != nil {
		return false, "", err
	}
	if dashboardIdentityHasAppliedRevision(identity) {
		return true, "", nil
	}
	return false, "no namespace has one fully applied revision", nil
}

func dashboardIdentityHasAppliedRevision(identity dashboardIdentity) bool {
	for _, scope := range identity.Namespaces {
		namespace := scope.Namespace
		if namespace.NamespaceID != "" && namespace.DesiredRevision > 0 &&
			namespace.DesiredRevision == namespace.AppliedRevision {
			return true
		}
	}
	return false
}

func dashboardSessionRequest(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	path string,
	payload map[string]string,
	verbose bool,
) (string, int, []byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return "", 0, nil, fmt.Errorf("marshal Dashboard session request: %w", err)
	}
	url := strings.TrimRight(baseURL, "/") + path
	if verbose {
		fmt.Printf("[Dashboard] POST %s (auth session)\n", url)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", 0, nil, fmt.Errorf("create Dashboard session request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return "", 0, nil, fmt.Errorf("Dashboard session request failed: %w", err)
	}
	defer func() {
		_ = response.Body.Close()
	}()
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, 8<<10))
	if err != nil {
		return "", 0, nil, fmt.Errorf("read Dashboard session response: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return "", response.StatusCode, responseBody, nil
	}
	for _, cookie := range response.Cookies() {
		if cookie.Name == dashboardSessionCookie && strings.TrimSpace(cookie.Value) != "" {
			return cookie.Value, response.StatusCode, responseBody, nil
		}
	}
	return "", response.StatusCode, responseBody, errors.New("Dashboard session response omitted its HttpOnly session cookie")
}

func boundedText(value []byte, limit int) string {
	text := string(value)
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}
