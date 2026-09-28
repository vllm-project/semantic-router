package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-route-bound-authorization", pkgtestcases.TestCase{
		Description: "Verify invitation, read-only route policy, unknown API denial, and immediate session revocation",
		Tags:        []string{"dashboard", "auth", "security"},
		Fn:          testDashboardRouteBoundAuthorization,
	})
}

func testDashboardRouteBoundAuthorization(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (resultErr error) {
	localPort, stop, connectionErr := setupServiceConnection(ctx, client, opts)
	if connectionErr != nil {
		return connectionErr
	}
	defer stop()

	httpClient := &http.Client{Timeout: 15 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	adminToken, loginErr := dashboardPolicyLogin(ctx, httpClient, baseURL, dashboardE2EAdminEmail, dashboardE2EAdminPassword)
	if loginErr != nil {
		return fmt.Errorf("bootstrap admin login: %w", loginErr)
	}

	unique := uuid.NewString()
	email := "e2e-read-" + unique + "@example.test"
	password := "E2e-read-" + uuid.NewString()
	var invitation struct {
		Invitation struct {
			ID   string `json:"id"`
			Role string `json:"role"`
		} `json:"invitation"`
		Token string `json:"token"`
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodPost, "/api/admin/invitations", adminToken,
		map[string]string{"kind": "personal", "email": email, "name": "E2E read user", "role": "read"},
		http.StatusCreated, &invitation, "create read invitation"); err != nil {
		return err
	}
	if invitation.Invitation.ID == "" || invitation.Invitation.Role != "read" || invitation.Token == "" {
		return errors.New("create read invitation: missing ID or token, or unexpected role")
	}

	var userID string
	userDeleted := false
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if userID != "" && !userDeleted {
			err := dashboardPolicyRequest(cleanupCtx, httpClient, baseURL, http.MethodDelete,
				"/api/admin/users/"+url.PathEscape(userID), adminToken, nil, http.StatusNoContent, nil, "delete test user")
			resultErr = errors.Join(resultErr, err)
		} else if userID == "" {
			err := dashboardPolicyRequest(cleanupCtx, httpClient, baseURL, http.MethodDelete,
				"/api/admin/invitations/"+url.PathEscape(invitation.Invitation.ID), adminToken, nil,
				http.StatusNoContent, nil, "revoke test invitation")
			resultErr = errors.Join(resultErr, err)
		}
	}()

	var accepted struct {
		User struct {
			ID   string `json:"id"`
			Role string `json:"role"`
		} `json:"user"`
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodPost,
		"/api/auth/invitations/"+url.PathEscape(invitation.Token)+"/accept", "",
		map[string]string{"password": password}, http.StatusOK, &accepted, "accept read invitation"); err != nil {
		return err
	}
	userID = accepted.User.ID
	if userID == "" || accepted.User.Role != "read" {
		return errors.New("accept read invitation: missing user ID or unexpected role")
	}

	readToken, loginErr := dashboardPolicyLogin(ctx, httpClient, baseURL, email, password)
	if loginErr != nil {
		return fmt.Errorf("read user login: %w", loginErr)
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodGet, "/api/router/config/all", readToken,
		nil, http.StatusOK, nil, "read config"); err != nil {
		return err
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodPost, "/api/router/config/update", readToken,
		map[string]any{}, http.StatusForbidden, nil, "deny config mutation"); err != nil {
		return err
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodGet, "/api/e2e-unmapped-"+unique, readToken,
		nil, http.StatusForbidden, nil, "deny unmapped API route"); err != nil {
		return err
	}
	if err := dashboardPolicyPreflight(ctx, httpClient, baseURL, "/api/router/v1/chat/completions"); err != nil {
		return err
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodPost,
		"/api/router/v1/chat/completions", "", nil, http.StatusUnauthorized, nil,
		"deny unauthenticated request after preflight"); err != nil {
		return err
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodOptions,
		"/api/e2e-unmapped-"+unique, "", nil, http.StatusForbidden, nil,
		"deny unmapped preflight"); err != nil {
		return err
	}

	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodPatch,
		"/api/admin/users/"+url.PathEscape(userID), adminToken,
		map[string]string{"status": "inactive"}, http.StatusOK, nil, "disable test user"); err != nil {
		return err
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodGet, "/api/router/config/all", readToken,
		nil, http.StatusUnauthorized, nil, "deny disabled user's existing token"); err != nil {
		return err
	}
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodDelete,
		"/api/admin/users/"+url.PathEscape(userID), adminToken, nil, http.StatusNoContent, nil, "delete test user"); err != nil {
		return err
	}
	userDeleted = true
	if err := dashboardPolicyRequest(ctx, httpClient, baseURL, http.MethodGet, "/api/router/config/all", readToken,
		nil, http.StatusUnauthorized, nil, "deny deleted user's existing token"); err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Println("[Dashboard] route-bound authorization OK: read=200, mutation=403, unknown=403, disabled/deleted token=401")
	}
	return nil
}

func dashboardPolicyLogin(ctx context.Context, client *http.Client, baseURL, email, password string) (string, error) {
	var response struct {
		Token string `json:"token"`
	}
	if err := dashboardPolicyRequest(ctx, client, baseURL, http.MethodPost, "/api/auth/login", "",
		map[string]string{"email": email, "password": password}, http.StatusOK, &response, "login"); err != nil {
		return "", err
	}
	if response.Token == "" {
		return "", errors.New("login: missing token")
	}
	return response.Token, nil
}

func dashboardPolicyPreflight(ctx context.Context, client *http.Client, baseURL, path string) error {
	const origin = "https://example.test"
	request, requestErr := http.NewRequestWithContext(ctx, http.MethodOptions, baseURL+path, nil)
	if requestErr != nil {
		return fmt.Errorf("create Dashboard preflight: %w", requestErr)
	}
	request.Header.Set("Origin", origin)
	request.Header.Set("Access-Control-Request-Method", http.MethodPost)
	response, responseErr := client.Do(request)
	if responseErr != nil {
		return fmt.Errorf("dashboard preflight request failed (%T)", responseErr)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusNoContent || response.Header.Get("Access-Control-Allow-Origin") != origin {
		return fmt.Errorf("dashboard preflight returned HTTP %d with allow-origin %q, want 204 and %q",
			response.StatusCode, response.Header.Get("Access-Control-Allow-Origin"), origin)
	}
	return nil
}

// dashboardPolicyRequest only reports the operation and HTTP status. Invitation
// tokens and passwords must not appear in E2E logs, including transport errors.
func dashboardPolicyRequest(ctx context.Context, client *http.Client, baseURL, method, path, bearer string,
	payload any, wantStatus int, result any, operation string,
) error {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("%s: encode request: %w", operation, err)
		}
		body = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, baseURL+path, body)
	if err != nil {
		return fmt.Errorf("%s: create request failed", operation)
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if bearer != "" {
		setDashboardAuth(req, bearer)
	}
	resp, err := client.Do(req)
	if err != nil {
		// url.Error contains the request URL and could expose the invitation token.
		return fmt.Errorf("%s: HTTP request failed (%T)", operation, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != wantStatus {
		return fmt.Errorf("%s: got HTTP %d, want %d", operation, resp.StatusCode, wantStatus)
	}
	if result != nil {
		if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(result); err != nil {
			return fmt.Errorf("%s: decode response: %w", operation, err)
		}
	}
	return nil
}
