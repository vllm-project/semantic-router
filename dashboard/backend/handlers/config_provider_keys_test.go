package handlers

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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	fakeclientset "k8s.io/client-go/kubernetes/fake"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

const providerKeyFixture = "sk-test-keyref-123"

func saveConfigWithBackendKey(t *testing.T, configPath, key string) {
	t.Helper()
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{}
	models := body["providers"].(map[string]interface{})["models"].([]map[string]interface{})
	models[0]["backend_refs"].([]map[string]interface{})[0]["api_key"] = key
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(response, httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(data)))
	if response.Code != http.StatusOK {
		t.Fatalf("save status = %d: %s", response.Code, response.Body.String())
	}
}

// savedKeyFile returns the file the saved config names for the backend key,
// after checking the config holds no literal key.
func savedKeyFile(t *testing.T, configPath, key string) string {
	t.Helper()
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(key)) || bytes.Contains(data, []byte("api_key:")) {
		t.Fatalf("the saved config holds the key:\n%s", data)
	}
	cfg, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VLLMEndpoints[0].APIKey != key {
		t.Fatal("the Router does not resolve the stored key")
	}
	auth := cfg.ModelConfig["test-model"].AuthoredModel.BackendRefs[0].Auth
	if auth == nil || filepath.Dir(auth.APIKeyFile) != filepath.Join(filepath.Dir(configPath), "provider-keys") {
		t.Fatalf("auth = %+v, want a file in the provider key directory", auth)
	}
	return auth.APIKeyFile
}

func TestSavingABackendKeyStoresItInAnOwnerOnlyFile(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	saveConfigWithBackendKey(t, configPath, providerKeyFixture)

	path := savedKeyFile(t, configPath, providerKeyFixture)
	for name, want := range map[string]os.FileMode{path: 0o600, filepath.Dir(path): 0o700} {
		info, err := os.Stat(name)
		if err != nil || info.Mode().Perm() != want {
			t.Fatalf("%s mode = %v (%v), want %v", name, info.Mode().Perm(), err, want)
		}
	}
}

func TestSavingKeepsAnEnvironmentReference(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	body := backendBody("127.0.0.1:8000", map[string]interface{}{"api_key": "${OPENAI_API_KEY}"})
	if response := postConfig(t, configPath, body, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("save status = %d: %s", response.Code, response.Body.String())
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "api_key: ${OPENAI_API_KEY}") || strings.Contains(string(data), "api_key_file") {
		t.Fatalf("an environment reference must stay as written:\n%s", data)
	}
}

func TestTheNextSaveMovesALiteralKeyOut(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	literal := strings.Replace(string(data), "weight: 1\n", "weight: 1\n          api_key: "+providerKeyFixture+"\n", 1) + "global: {}\n"
	if err := os.WriteFile(configPath, []byte(literal), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := routerconfig.ParseYAMLBytes([]byte(literal)); err != nil {
		t.Fatalf("a config with a literal key no longer loads: %v", err)
	}

	// The editor sends the masked placeholder back for the key it never saw.
	saveConfigWithBackendKey(t, configPath, redactedConfigSecret)
	savedKeyFile(t, configPath, providerKeyFixture)
}

// installKeySecretTestWriter declares a ConfigMap target for configPath, with
// a key Secret when keySecret is set, and returns the fake cluster.
func installKeySecretTestWriter(t *testing.T, configPath string, keySecret bool) *fakeclientset.Clientset {
	t.Helper()
	const namespace = "router-test"
	t.Setenv(configwriter.ConfigMapNameEnv, "router-config")
	t.Setenv(configwriter.ConfigMapNamespaceEnv, namespace)
	mounted, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	objects := []runtime.Object{&corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "router-config", Namespace: namespace},
		Data:       map[string]string{"config.yaml": string(mounted)},
	}}
	if keySecret {
		t.Setenv(configwriter.KeySecretNameEnv, "router-keys")
		objects = append(objects, &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "router-keys", Namespace: namespace}})
	}
	client := fakeclientset.NewSimpleClientset(objects...)
	t.Cleanup(stubInClusterConfigMapWriter(t, configwriter.NewConfigMapWriter(client)))
	return client
}

func literalKeyDocument(t *testing.T, configPath string) []byte {
	t.Helper()
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	document := strings.Replace(string(data), "weight: 1\n", "weight: 1\n          api_key: "+providerKeyFixture+"\n", 1)
	return []byte(strings.Replace(document, "port: 8801\n", "port: 8801\n    api_keys: [gateway-key-fixture]\n", 1))
}

func TestAConfigMapSaveKeepsTheKeysInTheKeySecret(t *testing.T) {
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	configPath := createValidTestConfig(t, t.TempDir())
	client := installKeySecretTestWriter(t, configPath, true)

	if _, _, err := persistConfig(configPath, literalKeyDocument(t, configPath)); err != nil {
		t.Fatal(err)
	}
	saved, err := client.CoreV1().ConfigMaps("router-test").Get(t.Context(), "router-config", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	document := saved.Data["config.yaml"]
	if strings.Contains(document, providerKeyFixture) || strings.Contains(document, "gateway-key-fixture") {
		t.Fatalf("the ConfigMap holds a key:\n%s", document)
	}
	secret, err := client.CoreV1().Secrets("router-test").Get(t.Context(), "router-keys", metav1.GetOptions{})
	if err != nil || len(secret.Data) != 2 {
		t.Fatalf("key Secret = %v (%v), want both keys", secret, err)
	}
	for name, key := range secret.Data {
		reference := "api_key_env: " + name
		if string(key) == "gateway-key-fixture" {
			reference = "${" + name + "}"
		}
		if !strings.Contains(document, reference) {
			t.Fatalf("the ConfigMap does not name %s:\n%s", reference, document)
		}
		// The chart loads the Secret into the Router's and the Dashboard's environment.
		t.Setenv(name, string(key))
	}
	cfg, err := routerconfig.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VLLMEndpoints[0].APIKey != providerKeyFixture || cfg.Listeners[0].ClientKeys()[0] != "gateway-key-fixture" {
		t.Fatal("the Router does not resolve the stored keys from its environment")
	}
	// Model verification reads the backend's key from the same environment.
	target, err := resolveModelVerificationTarget(cfg, "test-model")
	if err != nil || target.apiKey != providerKeyFixture {
		t.Fatalf("model verification target key resolved = %v (%v)", target.apiKey == providerKeyFixture, err)
	}
}

// The manifests outside the chart declare a ConfigMap target without a key
// Secret. A save there keeps a literal key as written, and says so, so an
// unrelated edit of a config that already holds one still saves.
func TestAConfigMapSaveWithoutAKeySecretKeepsTheKeyAndWarns(t *testing.T) {
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	configPath := createValidTestConfig(t, t.TempDir())
	client := installKeySecretTestWriter(t, configPath, false)
	document := literalKeyDocument(t, configPath)

	stored, warnings, err := persistConfig(configPath, document)
	if err != nil || string(stored) != string(document) {
		t.Fatalf("save = %s (%v), want the document as written", stored, err)
	}
	if len(warnings) != 2 || warnings[0].Code != "literal_key_kept" {
		t.Fatalf("warnings = %+v, want both kept keys", warnings)
	}
	response := httptest.NewRecorder()
	writeConfigSavedResponse(response, warnings)
	if !strings.Contains(response.Body.String(), "literal_key_kept") || strings.Contains(response.Body.String(), providerKeyFixture) {
		t.Fatalf("response = %s, want the warning without the key", response.Body.String())
	}
	saved, err := client.CoreV1().ConfigMaps("router-test").Get(t.Context(), "router-config", metav1.GetOptions{})
	if err != nil || saved.Data["config.yaml"] != string(document) {
		t.Fatalf("the ConfigMap does not hold the saved document: %v", err)
	}
}

func TestAFailedSaveLeavesNoKeyFile(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	previous := atomicRename
	atomicRename = func(string, string) error { return errors.New("disk full") }
	t.Cleanup(func() { atomicRename = previous })

	if _, _, err := persistConfig(configPath, literalKeyDocument(t, configPath)); err == nil {
		t.Fatal("the save did not fail")
	}
	entries, err := os.ReadDir(filepath.Join(filepath.Dir(configPath), "provider-keys"))
	if err == nil && len(entries) != 0 {
		t.Fatalf("the failed save left %d key files", len(entries))
	}
}

func TestASaveMovesListenerKeysOnlyWhenTheRouterServesTheListener(t *testing.T) {
	for _, gateway := range []string{"standalone", "extproc"} {
		t.Run(gateway, func(t *testing.T) {
			t.Setenv("VLLM_SR_GATEWAY", gateway)
			configPath := createValidTestConfig(t, t.TempDir())
			stored, _, err := persistConfig(configPath, literalKeyDocument(t, configPath))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := routerconfig.ParseYAMLBytes(stored)
			if err != nil {
				t.Fatal(err)
			}
			listener := cfg.Listeners[0]
			if moved := len(listener.APIKeyFiles) == 1 && len(listener.APIKeys) == 0; moved != (gateway == "standalone") {
				t.Fatalf("%s listener = %+v; Envoy renders its own keys, the Router reads files", gateway, listener)
			}
			if keys := listener.ClientKeys(); len(keys) != 1 || keys[0] != "gateway-key-fixture" {
				t.Fatal("the listener no longer accepts its key")
			}
		})
	}
}
