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

// The mandatory contract uses a self-contained Kubernetes configuration source.
// RELEASE_READONLY_CONFIG additionally exercises the same route with a real
// read-only bind mount when the integration runner provides one.
func TestManagementRouteReadonlyConfiguration(t *testing.T) {
	original := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("before_readonly_check"))
	candidate := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("after_readonly_check"))
	payload, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(candidate)})
	if err != nil {
		t.Fatal(err)
	}
	const token = "release-audit-dummy-token"
	t.Setenv("RELEASE_AUDIT_MGMT_TOKEN", token)
	type readonlyConfigCase struct {
		name     string
		path     string
		readonly bool
		source   config.ConfigSource
	}
	cases := []readonlyConfigCase{
		{name: "ordinary_file_control"},
		{name: "kubernetes_source", readonly: true, source: config.ConfigSourceKubernetes},
	}
	if mountPath := os.Getenv("RELEASE_READONLY_CONFIG"); mountPath != "" {
		cases = append(cases, readonlyConfigCase{name: "readonly_mount", path: mountPath, readonly: true})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			configPath := tc.path
			before := original
			if configPath == "" {
				configPath = filepath.Join(t.TempDir(), "config.yaml")
				if writeErr := os.WriteFile(configPath, before, 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
			} else {
				var readErr error
				before, readErr = os.ReadFile(configPath)
				if readErr != nil {
					t.Fatal(readErr)
				}
			}
			management := config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeBearer, Tokens: []config.ManagementAPITokenRef{{Env: "RELEASE_AUDIT_MGMT_TOKEN", Role: "admin"}}, Roles: config.DefaultManagementAPIRoles()}}
			server := testManagementAPIServer(t, management)
			server.configPath = configPath
			server.config.ConfigSource = tc.source
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
			if !tc.readonly {
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
