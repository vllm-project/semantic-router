package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestDashboardHeadReachesReadHandlers(t *testing.T) {
	server, cfg := setupRouteInventoryServerWithConfig(t)
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service := auth.NewService(store, cfg.JWTSecret, cfg.JWTExpiryHours)
	const password = "head-route-integration-password"
	hash, err := service.HashPassword(password)
	if err != nil {
		t.Fatal(err)
	}
	user, err := store.CreateUser(t.Context(), "head-routes@example.test", "HEAD Routes", hash, auth.RoleAdmin, "active")
	if err != nil {
		t.Fatal(err)
	}
	token, _, err := service.Login(t.Context(), user.Email, password)
	if err != nil {
		t.Fatal(err)
	}
	httpServer := httptest.NewServer(server.Handler)
	defer httpServer.Close()
	client := httpServer.Client()
	client.Timeout = 5 * time.Second
	for _, path := range []string{
		"/healthz", "/api/setup/state", "/api/settings", "/api/instance",
		"/api/auth/me", "/api/auth/bootstrap/can-register", "/api/auth/invitations/missing",
		"/api/admin/users", "/api/admin/permissions", "/api/admin/audit-logs", "/api/admin/invitations",
		"/api/mcp/servers", "/api/mcp/tools", "/api/ml-pipeline/jobs",
		"/api/router/config/all", "/api/router/config/yaml", "/api/router/config/versions",
		"/api/router/config/global", "/api/router/config/global/raw", "/api/tools/web-search/stats",
		"/api/logs", "/api/v1/capabilities", "/api/workflows/health", "/api/recipe", "/api/recipe/probes", "/api/recipe/packages",
	} {
		t.Run(path, func(t *testing.T) {
			var getStatus int
			var contentType string
			for _, method := range []string{http.MethodGet, http.MethodHead} {
				request, err := http.NewRequestWithContext(t.Context(), method, httpServer.URL+path, nil)
				if err != nil {
					t.Fatal(err)
				}
				request.Header.Set("Authorization", "Bearer "+token)
				response, err := client.Do(request)
				if err != nil {
					t.Fatal(err)
				}
				body, err := io.ReadAll(response.Body)
				response.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				if method == http.MethodGet {
					getStatus = response.StatusCode
					contentType = response.Header.Get("Content-Type")
					if getStatus == http.StatusMethodNotAllowed || getStatus == http.StatusForbidden || getStatus == http.StatusUnauthorized {
						t.Fatalf("GET fixture rejected: %d %s", getStatus, body)
					}
				} else {
					if response.StatusCode != getStatus {
						t.Errorf("HEAD status=%d; GET status=%d", response.StatusCode, getStatus)
					}
					if response.Header.Get("Content-Type") != contentType {
						t.Errorf("HEAD Content-Type=%q; GET=%q", response.Header.Get("Content-Type"), contentType)
					}
					if len(body) != 0 {
						t.Errorf("HEAD returned body %q", body)
					}
				}
			}
		})
	}
}
