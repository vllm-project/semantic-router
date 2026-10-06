package handlers

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

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

func installMountedConfigMapTestWriter(t *testing.T, configPath string) {
	t.Helper()
	const namespace, name = "router-test", "router-config"
	t.Setenv(configwriter.ConfigMapNameEnv, name)
	t.Setenv(configwriter.ConfigMapNamespaceEnv, namespace)
	mounted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	client := fakeclientset.NewSimpleClientset(&corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Data:       map[string]string{"config.yaml": string(mounted)},
	})
	restore := stubInClusterConfigMapWriter(t, configwriter.NewConfigMapWriter(client))
	t.Cleanup(restore)
}

func assertSingleBaselineBackup(t *testing.T, configDir string, want []byte) {
	t.Helper()
	entries, err := os.ReadDir(configBackupDir(configDir))
	if err != nil {
		t.Fatal(err)
	}
	var backups []string
	for _, entry := range entries {
		if isConfigBackupEntry(entry) {
			backups = append(backups, entry.Name())
		}
	}
	if len(backups) != 1 {
		t.Fatalf("backup count = %d, names = %v; want one unchanged baseline", len(backups), backups)
	}
	data, err := os.ReadFile(filepath.Join(configBackupDir(configDir), backups[0]))
	if err != nil || !bytes.Equal(data, want) {
		t.Fatalf("baseline backup = %q, error %v; want original mounted config", data, err)
	}
}

func TestStaleConfigMapDeployAndRollbackLeaveBaselineBackupUntouched(t *testing.T) {
	configDir := t.TempDir()
	configPath := createValidTestConfig(t, configDir)
	mounted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	installMountedConfigMapTestWriter(t, configPath)

	deployYAML := `routing:
  decisions:
    - name: deployed-default
      priority: 5
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`
	deploy := func(dsl string) *httptest.ResponseRecorder {
		body, marshalErr := json.Marshal(DeployRequest{YAML: deployYAML, DSL: dsl})
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		response := httptest.NewRecorder()
		DeployHandler(configPath, false, configDir)(response, httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(body)))
		return response
	}
	first := deploy("first DSL")
	if first.Code != http.StatusAccepted {
		t.Fatalf("first deploy = HTTP %d: %s", first.Code, first.Body.String())
	}
	var result DeployResponse
	if unmarshalErr := json.Unmarshal(first.Body.Bytes(), &result); unmarshalErr != nil || result.Version == "" {
		t.Fatalf("first deploy result = %+v, error %v", result, unmarshalErr)
	}
	assertSingleBaselineBackup(t, configDir, mounted)

	second := deploy("second DSL must not replace first")
	if second.Code != http.StatusConflict || !strings.Contains(second.Body.String(), errConfigRolloutRequired.Error()) {
		t.Fatalf("stale deploy = HTTP %d: %s", second.Code, second.Body.String())
	}
	assertSingleBaselineBackup(t, configDir, mounted)
	if dsl, readErr := os.ReadFile(archivedDSLPath(configDir)); readErr != nil || string(dsl) != "first DSL" {
		t.Fatalf("stale deploy changed DSL archive = %q, error %v", dsl, readErr)
	}

	rollbackBody, err := json.Marshal(map[string]string{"version": result.Version})
	if err != nil {
		t.Fatal(err)
	}
	rollback := httptest.NewRecorder()
	RollbackHandler(configPath, false, configDir)(rollback, httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(rollbackBody)))
	if rollback.Code != http.StatusConflict || !strings.Contains(rollback.Body.String(), errConfigRolloutRequired.Error()) {
		t.Fatalf("stale rollback = HTTP %d: %s", rollback.Code, rollback.Body.String())
	}
	assertSingleBaselineBackup(t, configDir, mounted)
}

func TestStaleConfigMapSetupActivationLeavesBaselineBackupUntouched(t *testing.T) {
	isolateConfigMutationRuntime(t)
	configDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, configDir)
	mounted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	installMountedConfigMapTestWriter(t, configPath)
	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatal(err)
	}
	handler := SetupActivateHandler(configPath, false, configDir, setupmode.New(configPath, false))
	first := httptest.NewRecorder()
	handler(first, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if first.Code != http.StatusAccepted {
		t.Fatalf("first setup activation = HTTP %d: %s", first.Code, first.Body.String())
	}
	assertSingleBaselineBackup(t, configDir, mounted)

	second := httptest.NewRecorder()
	handler(second, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if second.Code != http.StatusConflict || !strings.Contains(second.Body.String(), errConfigRolloutRequired.Error()) {
		t.Fatalf("stale setup activation = HTTP %d: %s", second.Code, second.Body.String())
	}
	assertSingleBaselineBackup(t, configDir, mounted)
}
