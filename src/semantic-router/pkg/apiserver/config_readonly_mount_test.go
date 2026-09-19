//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestManagementRouteReadonlyMount(t *testing.T) {
	mountPath := os.Getenv("RELEASE_READONLY_CONFIG")
	if mountPath == "" {
		t.Skip("set RELEASE_READONLY_CONFIG to a read-only single-file mount")
	}
	before, err := os.ReadFile(mountPath)
	if err != nil {
		t.Fatal(err)
	}
	candidate := strings.Replace(string(before), "default-business", "release-management-route", 1)
	if candidate == string(before) {
		t.Fatal("fixture route was not changed")
	}
	payload, err := json.Marshal(RouterConfigUpdateRequest{YAML: candidate})
	if err != nil {
		t.Fatal(err)
	}
	const token = "release-audit-dummy-token"
	t.Setenv("RELEASE_AUDIT_MGMT_TOKEN", token)
	for _, tc := range []struct {
		name    string
		mounted bool
	}{{"ordinary_file_control", false}, {"readonly_mount", true}} {
		t.Run(tc.name, func(t *testing.T) {
			configPath := mountPath
			if !tc.mounted {
				configPath = filepath.Join(t.TempDir(), "config.yaml")
				if err := os.WriteFile(configPath, before, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			management := config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeBearer, Tokens: []config.ManagementAPITokenRef{{Env: "RELEASE_AUDIT_MGMT_TOKEN", Role: "admin"}}, Roles: config.DefaultManagementAPIRoles()}}
			server := testManagementAPIServer(t, management)
			server.configPath = configPath
			mux := server.setupRoutes()
			anonymous := httptest.NewRecorder()
			mux.ServeHTTP(anonymous, httptest.NewRequest(http.MethodPut, "/api/v1/config", bytes.NewReader(payload)))
			if anonymous.Code != http.StatusUnauthorized {
				t.Fatalf("normal management auth gate missing: HTTP=%d", anonymous.Code)
			}
			request := httptest.NewRequest(http.MethodPut, "/api/v1/config", bytes.NewReader(payload))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Authorization", "Bearer "+token)
			setConfigPrecondition(t, request, configPath)
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)
			after, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("normal management registered route anonymous=%d authorized=%d unchanged=%t response=%s", anonymous.Code, response.Code, bytes.Equal(before, after), response.Body.String())
			if !tc.mounted {
				if response.Code != http.StatusOK {
					t.Fatalf("writable control failed: %d %s", response.Code, response.Body.String())
				}
				if bytes.Equal(before, after) {
					t.Fatal("writable control did not persist change")
				}
				return
			}
			if !bytes.Equal(before, after) {
				t.Fatal("readonly mounted source changed")
			}
			if response.Code != http.StatusForbidden || !strings.Contains(response.Body.String(), "CONFIG_READ_ONLY") {
				t.Fatalf("normal management route surfaces late filesystem failure instead of declared immutable capability: HTTP=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}
