//go:build !windows

package apiserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclientset "k8s.io/client-go/kubernetes/fake"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

const (
	appliedProviderKey = "sk-test-applied-provider"
	appliedGatewayKey  = "sk-test-applied-gateway"
)

// literalKeyConfig is a document a client applies with its keys written in it.
func literalKeyConfig(t *testing.T, decision string) string {
	t.Helper()
	var doc map[string]any
	if err := yaml.Unmarshal(mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig(decision)), &doc); err != nil {
		t.Fatal(err)
	}
	doc["listeners"].([]any)[0].(map[string]any)["api_keys"] = []any{appliedGatewayKey}
	model := doc["providers"].(map[string]any)["models"].([]any)[0].(map[string]any)
	model["backend_refs"].([]any)[0].(map[string]any)["api_key"] = appliedProviderKey
	data, err := yaml.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func sendKeyConfig(t *testing.T, server *ClassificationAPIServer, document string) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: document})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPut, "/api/v1/config", bytes.NewReader(body))
	setConfigPrecondition(t, request, server.configPath)
	response := httptest.NewRecorder()
	server.handleConfigPut(response, request)
	return response
}

func putKeyConfig(t *testing.T, server *ClassificationAPIServer, document string) {
	t.Helper()
	if response := sendKeyConfig(t, server, document); response.Code != http.StatusOK {
		t.Fatalf("PUT status = %d: %s", response.Code, response.Body.String())
	}
}

func planKeyConfig(t *testing.T, server *ClassificationAPIServer, document string) routerConfigPlanResponse {
	t.Helper()
	body, err := json.Marshal(routerConfigPlanRequest{YAML: document, Mode: routerConfigMutationReplace})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	server.handleConfigPlan(response, httptest.NewRequest(http.MethodPost, apiConfigPlanPath, bytes.NewReader(body)))
	var plan routerConfigPlanResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &plan) != nil {
		t.Fatalf("plan status = %d: %s", response.Code, response.Body.String())
	}
	return plan
}

// filesHolding lists the files under dir whose bytes contain key.
func filesHolding(t *testing.T, dir, key string) []string {
	t.Helper()
	var files, found []string
	err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
		if err == nil && !entry.IsDir() {
			files = append(files, path)
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range files {
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if bytes.Contains(data, []byte(key)) {
			found = append(found, path)
		}
	}
	return found
}

func TestAppliedKeysStayOutOfTheConfigAndItsHistory(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	server := &ClassificationAPIServer{configPath: configPath, gatewayMode: config.GatewayStandalone}
	document := literalKeyConfig(t, "new_route")

	putKeyConfig(t, server, document)
	if plan := planKeyConfig(t, server, document); plan.Changed {
		t.Fatalf("plan of the applied document = %+v, want unchanged", plan)
	}
	putKeyConfig(t, server, literalKeyConfig(t, "newer_route"))

	keyDir := filepath.Join(filepath.Dir(configPath), config.RouterKeyDir)
	for _, key := range []string{appliedProviderKey, appliedGatewayKey} {
		held := filesHolding(t, filepath.Dir(configPath), key)
		if len(held) != 1 || filepath.Dir(held[0]) != keyDir {
			t.Fatalf("%s is in %v, want only its key file in %s", key, held, keyDir)
		}
	}
	cfg, err := config.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VLLMEndpoints[0].APIKey != appliedProviderKey || strings.Join(cfg.Listeners[0].ClientKeys(), ",") != appliedGatewayKey {
		t.Fatal("the Router does not read the stored keys")
	}
}

func TestAnEnvoyRouterKeepsAppliedListenerKeys(t *testing.T) {
	// Envoy holds the listener's keys, so they are already in the deployed config.
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(literalKeyConfig(t, "old_route")), 0o600); err != nil {
		t.Fatal(err)
	}
	server := &ClassificationAPIServer{configPath: configPath, gatewayMode: config.GatewayExtProc}
	putKeyConfig(t, server, literalKeyConfig(t, "new_route"))
	persisted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(persisted, []byte(appliedGatewayKey)) || bytes.Contains(persisted, []byte(appliedProviderKey)) {
		t.Fatalf("extproc: Envoy renders listener keys, the backend key moves:\n%s", persisted)
	}
}

func TestAFailedWriteLeavesNoKeyBehind(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	server := &ClassificationAPIServer{configPath: configPath, gatewayMode: config.GatewayStandalone}
	previous := atomicRename
	atomicRename = func(string, string) error { return errors.New("disk full") }
	t.Cleanup(func() { atomicRename = previous })

	if response := sendKeyConfig(t, server, literalKeyConfig(t, "new_route")); response.Code == http.StatusOK {
		t.Fatalf("a failed write answered %d", response.Code)
	}
	if held := filesHolding(t, filepath.Dir(configPath), appliedProviderKey); len(held) != 0 {
		t.Fatalf("the failed write left the key in %v", held)
	}
}

// A ConfigMap target that declares no key Secret, as the manifests outside
// the chart do, keeps a literal key as written and says so.
func TestAConfigMapWithoutAKeySecretKeepsTheKeyAndWarns(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	mounted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(configwriter.ConfigMapNameEnv, "router-config")
	t.Setenv(configwriter.ConfigMapNamespaceEnv, "router")
	client := fakeclientset.NewSimpleClientset(&corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "router", Name: "router-config"},
		Data:       map[string]string{"config.yaml": string(mounted)},
	})
	t.Cleanup(stubInClusterConfigMapWriter(t, configwriter.NewConfigMapWriter(client)))
	server := &ClassificationAPIServer{configPath: configPath, gatewayMode: config.GatewayStandalone}
	document := literalKeyConfig(t, "new_route")

	response := sendKeyConfig(t, server, document)
	var result RouterConfigUpdateResponse
	if response.Code != http.StatusAccepted || json.Unmarshal(response.Body.Bytes(), &result) != nil {
		t.Fatalf("PUT status = %d: %s", response.Code, response.Body.String())
	}
	if len(result.Warnings) != 2 || result.Warnings[0].Code != "literal_key_kept" || bytes.Contains(response.Body.Bytes(), []byte(appliedProviderKey)) {
		t.Fatalf("warnings = %+v, want both kept keys named without their values", result.Warnings)
	}
	saved, err := client.CoreV1().ConfigMaps("router").Get(t.Context(), "router-config", metav1.GetOptions{})
	if err != nil || !strings.Contains(saved.Data["config.yaml"], appliedProviderKey) {
		t.Fatalf("the ConfigMap lost the kept key (%v)", err)
	}
	if plan := planKeyConfig(t, server, document); plan.Changed {
		t.Fatalf("plan of the applied document = %+v, want unchanged", plan)
	}
}
