//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestReplicaStatusHTTPContract(t *testing.T) {
	t.Setenv("VSR_STATUS_VIEWER_TOKEN", "status-viewer-example")
	t.Setenv("VSR_STATUS_DENIED_TOKEN", "status-denied-example")
	roles := config.DefaultManagementAPIRoles()
	roles["health-only"] = []string{string(PermHealthRead)}
	api := testManagementAPIServer(t, config.ManagementAPIConfig{
		Auth: config.ManagementAPIAuthConfig{
			Mode: config.ManagementAuthModeBearer,
			Tokens: []config.ManagementAPITokenRef{
				{Env: "VSR_STATUS_VIEWER_TOKEN", Role: "viewer"},
				{Env: "VSR_STATUS_DENIED_TOKEN", Role: "health-only"},
			},
			Roles: roles,
		},
	})
	api.runtimeRegistry = routerruntime.NewRegistry(api.config)
	server := httptest.NewServer(api.setupRoutes())
	defer server.Close()

	for _, tc := range []struct {
		name   string
		token  string
		status int
	}{
		{name: "anonymous", status: http.StatusUnauthorized},
		{name: "missing-permission", token: "status-denied-example", status: http.StatusForbidden},
		{name: "viewer", token: "status-viewer-example", status: http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request, err := http.NewRequest(http.MethodGet, server.URL+"/api/v1/status", nil)
			if err != nil {
				t.Fatal(err)
			}
			if tc.token != "" {
				request.Header.Set("Authorization", "Bearer "+tc.token)
			}
			response, err := server.Client().Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != tc.status {
				t.Fatalf("status=%d, want=%d, body=%s", response.StatusCode, tc.status, body)
			}
			if tc.status != http.StatusOK {
				return
			}
			var status struct {
				SchemaVersion string `json:"schema_version"`
				Scope         string `json:"scope"`
				InstanceID    string `json:"instance_id"`
				Conditions    []struct {
					Type   string `json:"type"`
					Status string `json:"status"`
					Reason string `json:"reason"`
				} `json:"conditions"`
			}
			if err := json.Unmarshal(body, &status); err != nil {
				t.Fatal(err)
			}
			if status.SchemaVersion != "v1" || status.Scope != "replica" || status.InstanceID == "" {
				t.Fatalf("unexpected status identity: %s", body)
			}
			if len(status.Conditions) == 0 {
				t.Fatalf("missing status conditions: %s", body)
			}
			for _, condition := range status.Conditions {
				if condition.Type == "" || condition.Reason == "" {
					t.Fatalf("incomplete condition: %+v", condition)
				}
				if condition.Status != "True" && condition.Status != "False" && condition.Status != "Unknown" {
					t.Fatalf("invalid condition status: %+v", condition)
				}
			}
			if strings.Contains(string(body), "status-viewer-example") || strings.Contains(string(body), "status-denied-example") {
				t.Fatalf("status contains a management credential: %s", body)
			}
		})
	}

	response, err := server.Client().Get(server.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("public liveness changed: status=%d", response.StatusCode)
	}
}
