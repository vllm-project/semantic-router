package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func importSetupRoundTrip(t *testing.T, configPath string, patch map[string]interface{}) map[string]interface{} {
	t.Helper()
	// The fixture server is on loopback; declare it the way an operator would
	// declare a real internal target (#1388).
	allowLoopbackForTest(t)
	payload := mustJSONRaw(t, patch)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	w := httptest.NewRecorder()
	body := mustJSONRaw(t, SetupImportRemoteRequest{URL: server.URL})
	SetupImportRemoteHandler(configPath, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("import failed: %d %s", w.Code, w.Body.String())
	}
	var response SetupImportRemoteResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	var imported map[string]interface{}
	if err := json.Unmarshal(response.Config, &imported); err != nil {
		t.Fatal(err)
	}
	return imported
}

func validateSetupRoundTrip(t *testing.T, configPath string, patch map[string]interface{}) json.RawMessage {
	t.Helper()
	w := httptest.NewRecorder()
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	SetupValidateHandler(configPath, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("validation failed: %d %s", w.Code, w.Body.String())
	}
	var response SetupValidateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	return response.Config
}

func activateSetupRoundTrip(t *testing.T, configPath, root string, configJSON json.RawMessage) *routerconfig.RouterConfig {
	t.Helper()
	w := httptest.NewRecorder()
	body := mustJSONRaw(t, SetupConfigRequest{Config: configJSON})
	SetupActivateHandler(configPath, false, root, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("activation failed: %d %s", w.Code, w.Body.String())
	}
	active, err := routerconfig.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	return active
}

func TestSetupSparsePIIRoundTripPreservesOmittedAndExplicitEmptyMapping(t *testing.T) {
	for _, clear := range []bool{false, true} {
		name := "omitted"
		if clear {
			name = "explicit-empty"
		}
		t.Run(name, func(t *testing.T) {
			configPath, root, _ := setupActivationRegressionRuntime(t)
			patch := createValidSetupPatch()
			pii := map[string]interface{}{"use_cpu": false}
			if clear {
				pii["pii_mapping_path"] = ""
			}
			global := map[string]interface{}{"model_catalog": map[string]interface{}{
				"modules": map[string]interface{}{"classifier": map[string]interface{}{"pii": pii}},
			}}
			patch["global"] = global
			imported := importSetupRoundTrip(t, configPath, patch)
			if !reflect.DeepEqual(imported["global"], global) {
				t.Fatalf("import authored omitted global fields: %#v", imported["global"])
			}
			returned := validateSetupRoundTrip(t, configPath, imported)
			var decoded map[string]interface{}
			if err := json.Unmarshal(returned, &decoded); err != nil {
				t.Fatal(err)
			}
			assertManagedSetupAuthoredGlobal(t, global, returned)
			resolvedGlobal := decoded["global"].(map[string]interface{})
			if !reflect.DeepEqual(resolvedGlobal["model_catalog"], global["model_catalog"]) {
				t.Fatalf("validation changed the authored model catalog: %#v", resolvedGlobal["model_catalog"])
			}
			active := activateSetupRoundTrip(t, configPath, root, returned)
			wantMapping := "models/Vela-1.0-Encoder-307M-PII/pii_mapping.json"
			if clear {
				wantMapping = ""
			}
			if active.PIIModel.UseCPU || active.PIIMappingPath != wantMapping || !active.PIIModel.UseMmBERT32K {
				t.Fatalf("sparse PII override changed inherited settings: %+v", active.PIIModel)
			}
			if active.IsPIIClassifierEnabled() == clear {
				t.Fatalf("PII enablement does not follow the effective mapping: %v", active.IsPIIClassifierEnabled())
			}
			if active.ManagementAPI.BindAddress != "0.0.0.0" {
				t.Fatalf("omitted management listener became authored: %+v", active.ManagementAPI)
			}
		})
	}
}

func TestSetupSparseDomainRoundTripUsesCanonicalBackendResolution(t *testing.T) {
	configPath, root, _ := setupActivationRegressionRuntime(t)
	patch := createValidSetupPatch()
	patch["global"] = map[string]interface{}{"model_catalog": map[string]interface{}{
		"external": []map[string]interface{}{{
			"name": "named-category", "model_role": "classification", "llm_model_name": "category-service",
			"llm_endpoint": map[string]interface{}{"address": "127.0.0.1", "port": 8080},
		}},
		"modules": map[string]interface{}{"classifier": map[string]interface{}{"domain": map[string]interface{}{
			"backend": map[string]interface{}{"protocol": "http_classify", "contract": "label_distribution.v1", "model": "named-category"},
		}}},
	}}
	returned := validateSetupRoundTrip(t, configPath, patch)
	active := activateSetupRoundTrip(t, configPath, root, returned)
	category := active.CategoryModel
	if category.Backend == nil || category.Backend.Model != "named-category" ||
		category.Variant != "" || category.UseModernBERT || category.UseMmBERT32K {
		t.Fatalf("sparse remote backend did not clear the inherited local variant: %+v", category)
	}
}

func TestSetupCandidatePreservesGlobalReplacementAndBootstrapInheritance(t *testing.T) {
	root := t.TempDir()
	configPath := createBootstrapSetupConfig(t, root)
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	baseGlobal := map[string]interface{}{"model_catalog": map[string]interface{}{
		"modules": map[string]interface{}{"classifier": map[string]interface{}{"pii": map[string]interface{}{"use_cpu": false}}},
	}}
	globalYAML, err := marshalYAMLBytes(map[string]interface{}{"global": baseGlobal})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, append(data, globalYAML...), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, replace := range []bool{false, true} {
		patch := createValidSetupPatch()
		want := baseGlobal
		if replace {
			want = map[string]interface{}{"model_catalog": map[string]interface{}{}}
			patch["global"] = want
		}
		returned := validateSetupRoundTrip(t, configPath, patch)
		var decoded map[string]interface{}
		if err := json.Unmarshal(returned, &decoded); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(decoded["global"], want) {
			t.Fatalf("whole-global replacement=%v changed semantics: %#v", replace, decoded["global"])
		}
	}
}

func TestSetupCandidateRepeatedSerializationPreservesAuthoredManagement(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", configPath)
	for _, listener := range []interface{}{map[string]interface{}{}, nil, map[string]interface{}{"bind_address": "127.0.0.1"}} {
		global := map[string]interface{}{"services": map[string]interface{}{"management_api": listener}}
		candidate, err := decodeStrictSetupConfig(mustJSONRaw(t, map[string]interface{}{"version": "v0.3", "global": global}))
		if err != nil {
			t.Fatal(err)
		}
		before, err := marshalYAMLBytes(candidate.globalOverrideRaw)
		if err != nil {
			t.Fatal(err)
		}
		var first []byte
		for i := 0; i < 3; i++ {
			data, activationErr := marshalYAMLBytes(candidate.canonicalTransport())
			if activationErr != nil {
				t.Fatal(activationErr)
			}
			if i == 0 {
				first = data
			} else if !bytes.Equal(first, data) {
				t.Fatal("serialization mutated a reusable candidate")
			}
			encoded, encodeErr := rawJSONMessage(candidate.canonicalTransport())
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
			var decoded map[string]interface{}
			if decodeErr := json.Unmarshal(encoded, &decoded); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			if !reflect.DeepEqual(decoded["global"], global) {
				t.Fatalf("authored management node was changed: %#v", decoded["global"])
			}
		}
		after, err := marshalYAMLBytes(candidate.globalOverrideRaw)
		if err != nil || !bytes.Equal(before, after) {
			t.Fatalf("authored global node was mutated: %v", err)
		}
	}
}
