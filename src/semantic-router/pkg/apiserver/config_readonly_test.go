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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclientset "k8s.io/client-go/kubernetes/fake"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
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

func TestManagementRoutePersistsConfigMapAndRequiresRollout(t *testing.T) {
	original := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("before_configmap_update"))
	candidate := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("after_configmap_update"))
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, original, 0o400); err != nil {
		t.Fatal(err)
	}
	const namespace, name = "router-test", "router-config"
	t.Setenv(configwriter.ConfigMapNameEnv, name)
	t.Setenv(configwriter.ConfigMapNamespaceEnv, namespace)
	client := fakeclientset.NewSimpleClientset(&corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Data:       map[string]string{"config.yaml": string(original)},
	})
	restoreWriter := stubInClusterConfigMapWriter(t, configwriter.NewConfigMapWriter(client))
	defer restoreWriter()

	const token = "configmap-route-test-token"
	t.Setenv("RELEASE_AUDIT_MGMT_TOKEN", token)
	management := config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode:   config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{{Env: "RELEASE_AUDIT_MGMT_TOKEN", Role: "admin"}},
		Roles:  config.DefaultManagementAPIRoles(),
	}}
	server := testManagementAPIServer(t, management)
	server.configPath = configPath
	mux := server.setupRoutes()
	payload, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(candidate)})
	if err != nil {
		t.Fatal(err)
	}
	put := func() *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodPut, "/api/v1/config", bytes.NewReader(payload))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Authorization", "Bearer "+token)
		request.Header.Set("If-Match", configDocumentETag(original))
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		return response
	}
	first := put()
	if first.Code != http.StatusAccepted || !strings.Contains(first.Body.String(), `"activation_status":"persisted"`) {
		t.Fatalf("ConfigMap write = HTTP %d: %s", first.Code, first.Body.String())
	}
	stored, err := client.CoreV1().ConfigMaps(namespace).Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if stored.Data["config.yaml"] != string(candidate) {
		t.Fatal("candidate document was not persisted to the ConfigMap")
	}
	mounted, err := os.ReadFile(configPath)
	if err != nil || !bytes.Equal(mounted, original) {
		t.Fatal("a ConfigMap update unexpectedly changed the mounted source")
	}
	get := httptest.NewRequest(http.MethodGet, "/api/v1/config", nil)
	get.Header.Set("Authorization", "Bearer "+token)
	readback := httptest.NewRecorder()
	mux.ServeHTTP(readback, get)
	if readback.Code != http.StatusOK || readback.Header().Get("ETag") != configDocumentETag(candidate) {
		t.Fatalf("ConfigMap readback = HTTP %d ETag %s: %s", readback.Code, readback.Header().Get("ETag"), readback.Body.String())
	}
	second := put()
	if second.Code != http.StatusConflict || !strings.Contains(second.Body.String(), "CONFIG_ROLLOUT_REQUIRED") {
		t.Fatalf("stale Pod accepted a second mutation: HTTP %d: %s", second.Code, second.Body.String())
	}
}
