//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestAPIRouteCatalogRequiresPolicyMetadata(t *testing.T) {
	for _, route := range apiRoutes() {
		if route.Permission == "" {
			t.Fatalf("route %s missing permission", route.pattern())
		}
		if route.Sensitivity == "" {
			t.Fatalf("route %s missing sensitivity", route.pattern())
		}
	}
}

func TestAPIRouteCatalogHasUniqueMethodPathPairs(t *testing.T) {
	seen := make(map[string]struct{}, len(apiRoutes()))
	for _, route := range apiRoutes() {
		key := route.pattern()
		if _, ok := seen[key]; ok {
			t.Fatalf("duplicate route registration: %s", key)
		}
		seen[key] = struct{}{}
	}
}

func TestManagementAuthDisabledAllowsConfigRoutes(t *testing.T) {
	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeDisabled},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code == http.StatusUnauthorized || rr.Code == http.StatusForbidden {
		t.Fatalf("expected disabled auth to allow config read, got %d", rr.Code)
	}
}

func TestManagementAuthBearerRejectsAnonymousConfigRead(t *testing.T) {
	t.Setenv("VSR_MGMT_TOKEN", "test-management-token")

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for anonymous config read, got %d", rr.Code)
	}
	assertManagementErrorCode(t, rr, "UNAUTHORIZED")
}

func TestManagementAuthBearerAllowsViewerConfigRead(t *testing.T) {
	const token = "viewer-management-token"
	t.Setenv("VSR_MGMT_TOKEN", token)

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code == http.StatusUnauthorized || rr.Code == http.StatusForbidden {
		t.Fatalf("expected viewer token to allow config read, got %d body=%s", rr.Code, rr.Body.String())
	}
}

func TestManagementAuthBearerRejectsViewerConfigWrite(t *testing.T) {
	const token = "viewer-management-token"
	t.Setenv("VSR_MGMT_TOKEN", token)

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodPatch, "/config/router", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for viewer config write, got %d body=%s", rr.Code, rr.Body.String())
	}
	assertManagementErrorCode(t, rr, "FORBIDDEN")
}

func TestManagementAuthBearerRejectsInvalidToken(t *testing.T) {
	t.Setenv("VSR_MGMT_TOKEN", "real-token")

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "admin"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for invalid token, got %d", rr.Code)
	}
}

func TestManagementAuthBearerRequiresConfiguredTokens(t *testing.T) {
	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode:  config.ManagementAuthModeBearer,
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	req.Header.Set("Authorization", "Bearer unused")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 when bearer auth has no tokens, got %d", rr.Code)
	}
	assertManagementErrorCode(t, rr, "MANAGEMENT_AUTH_NOT_CONFIGURED")
}

func TestHealthRouteAllowsAnonymousAccessWithBearerAuth(t *testing.T) {
	t.Setenv("VSR_MGMT_TOKEN", "secret-token")

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected anonymous /health to succeed, got %d", rr.Code)
	}
	if got := rr.Header().Get(managementRequestIDHeader); got == "" {
		t.Fatalf("expected %s response header", managementRequestIDHeader)
	}
}

func TestReadyRouteRequiresAuthWhenBearerEnabled(t *testing.T) {
	t.Setenv("VSR_MGMT_TOKEN", "secret-token")

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected anonymous /ready to be unauthorized, got %d", rr.Code)
	}
}

func TestManagementAuthPreservesCallerRequestID(t *testing.T) {
	t.Setenv("VSR_MGMT_TOKEN", "secret-token")

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})
	mux := server.setupRoutes()

	const requestID = "req-phase1-auth-001"
	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	req.Header.Set(managementRequestIDHeader, requestID)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if got := rr.Header().Get(managementRequestIDHeader); got != requestID {
		t.Fatalf("expected request id %q, got %q", requestID, got)
	}
}

func testManagementAPIServer(t *testing.T, management config.ManagementAPIConfig) *ClassificationAPIServer {
	t.Helper()
	if management.Auth.Mode == "" {
		management.Auth.Mode = config.ManagementAuthModeDisabled
	}
	if len(management.Auth.Roles) == 0 {
		management.Auth.Roles = config.DefaultManagementAPIRoles()
	}
	return &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config: &config.RouterConfig{
			ManagementAPI: management,
		},
	}
}

// TestManagementAuthPolicyConcurrentWithConfigPublish exercises the auth
// path against concurrent publishConfigMutation writers. Under go test -race,
// an unsynchronized s.config read/write pair fails this test.
func TestManagementAuthPolicyConcurrentWithConfigPublish(t *testing.T) {
	t.Setenv("VSR_MGMT_TOKEN", "race-token")

	server := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
			},
			Roles: config.DefaultManagementAPIRoles(),
		},
	})

	const workers = 32
	var wg sync.WaitGroup
	wg.Add(workers * 2)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				policy := server.managementAuthPolicy()
				if policy.Mode != config.ManagementAuthModeBearer && policy.Mode != config.ManagementAuthModeDisabled {
					t.Errorf("unexpected auth mode %q", policy.Mode)
					return
				}
			}
		}()
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				next := &config.RouterConfig{
					ManagementAPI: config.ManagementAPIConfig{
						Auth: config.ManagementAPIAuthConfig{
							Mode: config.ManagementAuthModeBearer,
							Tokens: []config.ManagementAPITokenRef{
								{Env: "VSR_MGMT_TOKEN", Role: "viewer"},
							},
							Roles: config.DefaultManagementAPIRoles(),
						},
						Port: 8080 + i + j,
					},
				}
				server.publishConfigMutation(next)
			}
		}(i)
	}
	wg.Wait()
}

func assertManagementErrorCode(t *testing.T, rr *httptest.ResponseRecorder, want string) {
	t.Helper()
	body := rr.Body.String()
	if !strings.Contains(body, `"code":"`+want+`"`) && !strings.Contains(body, `"code": "`+want+`"`) {
		t.Fatalf("expected error code %q in body, got %s", want, body)
	}
	if !strings.Contains(body, "request_id") {
		t.Fatalf("expected request_id in error body, got %s", body)
	}
}
