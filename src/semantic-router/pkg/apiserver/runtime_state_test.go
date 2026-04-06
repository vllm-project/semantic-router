package apiserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type fakeResolvedClassificationService struct {
	batchErr      error
	updatedConfig *config.RouterConfig
}

func (s *fakeResolvedClassificationService) ClassifyIntent(req services.IntentRequest) (*services.IntentResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (s *fakeResolvedClassificationService) ClassifyIntentForEval(req services.IntentRequest) (*services.EvalResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (s *fakeResolvedClassificationService) DetectPII(req services.PIIRequest) (*services.PIIResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (s *fakeResolvedClassificationService) CheckSecurity(req services.SecurityRequest) (*services.SecurityResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (s *fakeResolvedClassificationService) ClassifyBatchUnifiedWithOptions(_ []string, _ interface{}) (*services.UnifiedBatchResponse, error) {
	if s.batchErr != nil {
		return nil, s.batchErr
	}
	return nil, fmt.Errorf("resolved service invoked")
}

func (s *fakeResolvedClassificationService) ClassifyFactCheck(req services.FactCheckRequest) (*services.FactCheckResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (s *fakeResolvedClassificationService) ClassifyUserFeedback(req services.UserFeedbackRequest) (*services.UserFeedbackResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (s *fakeResolvedClassificationService) HasUnifiedClassifier() bool      { return true }
func (s *fakeResolvedClassificationService) HasClassifier() bool             { return true }
func (s *fakeResolvedClassificationService) HasFactCheckClassifier() bool    { return true }
func (s *fakeResolvedClassificationService) HasHallucinationDetector() bool  { return true }
func (s *fakeResolvedClassificationService) HasHallucinationExplainer() bool { return true }
func (s *fakeResolvedClassificationService) HasFeedbackDetector() bool       { return true }
func (s *fakeResolvedClassificationService) UpdateConfig(newConfig *config.RouterConfig) {
	s.updatedConfig = newConfig
}

func (s *fakeResolvedClassificationService) RefreshRuntimeConfig(newConfig *config.RouterConfig) {
	s.updatedConfig = newConfig
}

func TestHandleBatchClassificationUsesResolvedClassificationService(t *testing.T) {
	resolvedSvc := &fakeResolvedClassificationService{}
	apiServer := &ClassificationAPIServer{
		classificationSvc: newLiveClassificationService(
			services.NewPlaceholderClassificationService(),
			func() classificationService { return resolvedSvc },
		),
		config: &config.RouterConfig{},
	}

	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/classify/batch",
		bytes.NewBufferString(`{"texts":["resolver should win"],"task_type":"intent"}`),
	)
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	apiServer.handleBatchClassification(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected status %d, got %d: %s", http.StatusInternalServerError, rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "resolved service invoked") {
		t.Fatalf("expected unified-classifier path to be used, got body: %s", rr.Body.String())
	}
}

func TestHandleOpenAIModelsUsesResolvedRuntimeConfig(t *testing.T) {
	staleCfg := testModelListConfig("StaleRouter", false, nil)
	liveCfg := testModelListConfig("LiveRouter", true, []string{"live-model"})

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            staleCfg,
		runtimeConfig: newLiveRuntimeConfig(
			staleCfg,
			func() *config.RouterConfig { return liveCfg },
			nil,
		),
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rr := httptest.NewRecorder()
	apiServer.handleOpenAIModels(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, rr.Code, rr.Body.String())
	}

	var resp OpenAIModelList
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	got := map[string]bool{}
	for _, model := range resp.Data {
		got[model.ID] = true
	}

	if !got["LiveRouter"] {
		t.Fatalf("expected live auto model name, got %+v", resp.Data)
	}
	if got["StaleRouter"] {
		t.Fatalf("did not expect stale auto model name in response: %+v", resp.Data)
	}
	if !got["live-model"] {
		t.Fatalf("expected live config models to be included, got %+v", resp.Data)
	}
}

func TestHandleClassifierInfoUsesResolvedRuntimeConfig(t *testing.T) {
	staleCfg := testModelListConfig("StaleRouter", false, nil)
	liveCfg := testModelListConfig("LiveRouter", false, nil)

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            staleCfg,
		runtimeConfig: newLiveRuntimeConfig(
			staleCfg,
			func() *config.RouterConfig { return liveCfg },
			nil,
		),
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/classifier/info", nil)
	rr := httptest.NewRecorder()
	apiServer.handleClassifierInfo(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, rr.Code, rr.Body.String())
	}

	var resp map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	var cfg map[string]interface{}
	if err := json.Unmarshal(resp["config"], &cfg); err != nil {
		t.Fatalf("failed to decode config payload: %v", err)
	}

	if cfg["AutoModelName"] != "LiveRouter" {
		t.Fatalf("expected live config payload, got %#v", cfg["AutoModelName"])
	}
}

func TestHandleClassifierInfoNormalizesYAMLStylePluginConfig(t *testing.T) {
	liveCfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "health_decision",
					Plugins: []config.DecisionPlugin{
						{
							Type: "system_prompt",
							Configuration: config.MustStructuredPayload(map[interface{}]interface{}{
								"enabled": true,
								"nested": map[interface{}]interface{}{
									"mode": "replace",
								},
							}),
						},
					},
				},
			},
		},
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		runtimeConfig: newLiveRuntimeConfig(
			liveCfg,
			func() *config.RouterConfig { return liveCfg },
			nil,
		),
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/classifier/info", nil)
	rr := httptest.NewRecorder()
	apiServer.handleClassifierInfo(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, rr.Code, rr.Body.String())
	}

	resp := decodeJSONObject(t, rr.Body.Bytes())
	cfgPayload := requireJSONObject(t, resp, "config")
	decisions := requireJSONArray(t, cfgPayload, "Decisions", 1)
	decision := requireJSONObjectValue(t, decisions[0], "decision")
	plugins := requireJSONArray(t, decision, "Plugins", 1)
	plugin := requireJSONObjectValue(t, plugins[0], "plugin")
	configuration := requireJSONObject(t, plugin, "configuration")
	nested := requireJSONObject(t, configuration, "nested")
	if nested["mode"] != "replace" {
		t.Fatalf("expected nested plugin config to be normalized, got %#v", nested)
	}
	if configuration["enabled"] != true {
		t.Fatalf("expected enabled=true in normalized plugin config, got %#v", configuration)
	}
}

func TestBuildModelsInfoResponseUsesResolvedRuntimeConfig(t *testing.T) {
	staleCfg := &config.RouterConfig{}
	liveCfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "live-category-model",
					Threshold:           0.42,
					CategoryMappingPath: "live-mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{
					{CategoryMetadata: config.CategoryMetadata{Name: "billing"}},
				},
			},
		},
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: &fakeResolvedClassificationService{},
		config:            staleCfg,
		runtimeConfig: newLiveRuntimeConfig(
			staleCfg,
			func() *config.RouterConfig { return liveCfg },
			nil,
		),
	}

	resp := apiServer.buildModelsInfoResponse()

	if len(resp.Models) == 0 {
		t.Fatalf("expected model info entries from live config")
	}

	found := false
	for _, model := range resp.Models {
		if model.Name != "category_classifier" {
			continue
		}
		found = true
		if model.ModelPath != "live-category-model" {
			t.Fatalf("expected live model path, got %q", model.ModelPath)
		}
		if model.Metadata["mapping_path"] != "live-mapping.json" {
			t.Fatalf("expected live mapping path, got %+v", model.Metadata)
		}
	}

	if !found {
		t.Fatalf("expected category_classifier entry, got %+v", resp.Models)
	}
}

func TestRuntimeRegistryResolvesSharedDependencies(t *testing.T) {
	cfg := &config.RouterConfig{}
	memoryStore := newMockMemoryStore()
	manager := &vectorstore.Manager{}
	fileStore := &vectorstore.FileStore{}

	registry := routerruntime.NewRegistry(cfg)
	registry.SetMemoryStore(memoryStore)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{
		Manager:   manager,
		FileStore: fileStore,
	})

	apiServer := &ClassificationAPIServer{
		runtimeRegistry: registry,
	}

	if got := apiServer.currentMemoryStore(); got != memoryStore {
		t.Fatalf("currentMemoryStore() = %v, want %v", got, memoryStore)
	}
	if got := apiServer.currentVectorStoreManager(); got != manager {
		t.Fatalf("currentVectorStoreManager() = %v, want %v", got, manager)
	}
	if got := apiServer.currentFileStore(); got != fileStore {
		t.Fatalf("currentFileStore() = %v, want %v", got, fileStore)
	}
}

func TestLiveRuntimeConfigUpdateDoesNotReplaceGlobalConfig(t *testing.T) {
	previousCfg := config.Get()
	stableCfg := previousCfg
	if stableCfg == nil {
		stableCfg = &config.RouterConfig{}
	}
	config.Replace(stableCfg)
	t.Cleanup(func() {
		config.Replace(stableCfg)
		if previousCfg != nil {
			config.Replace(previousCfg)
		}
	})

	updatedCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "runtime-only"},
	}
	runtimeCfg := newLiveRuntimeConfig(stableCfg, nil, nil)

	runtimeCfg.Update(updatedCfg)

	if got := runtimeCfg.Current(); got != updatedCfg {
		t.Fatalf("Current() = %p, want %p", got, updatedCfg)
	}
	if got := config.Get(); got != stableCfg {
		t.Fatalf("config.Get() = %p, want unchanged %p", got, stableCfg)
	}
}

func TestNewLegacyRuntimeRegistryBridgesCompatibilityGlobalsIntoRegistry(t *testing.T) {
	previousCfg := config.Get()
	initialCfg := previousCfg
	if initialCfg == nil {
		initialCfg = &config.RouterConfig{}
		config.Replace(initialCfg)
	}
	t.Cleanup(func() {
		config.Replace(initialCfg)
		if previousCfg != nil {
			config.Replace(previousCfg)
		}
	})

	previousSvc := services.GetGlobalClassificationService()
	services.SetGlobalClassificationService(nil)
	t.Cleanup(func() { services.SetGlobalClassificationService(previousSvc) })

	previousMemoryStore := memory.GetGlobalMemoryStore()
	memory.SetGlobalMemoryStore(nil)
	t.Cleanup(func() { memory.SetGlobalMemoryStore(previousMemoryStore) })

	previousManager := vectorStoreManager
	previousPipeline := globalPipeline
	previousEmbedder := globalEmbedder
	previousFileStore := globalFileStore
	vectorStoreManager = nil
	globalPipeline = nil
	globalEmbedder = nil
	globalFileStore = nil
	t.Cleanup(func() {
		vectorStoreManager = previousManager
		globalPipeline = previousPipeline
		globalEmbedder = previousEmbedder
		globalFileStore = previousFileStore
	})

	initialSvc := services.NewPlaceholderClassificationService()
	initialStore := newMockMemoryStore()
	initialManager := &vectorstore.Manager{}
	initialFileStore := &vectorstore.FileStore{}
	services.SetGlobalClassificationService(initialSvc)
	memory.SetGlobalMemoryStore(initialStore)
	vectorStoreManager = initialManager
	globalFileStore = initialFileStore

	registry, cleanup, err := newLegacyRuntimeRegistry("/unused/config.yaml")
	if err != nil {
		t.Fatalf("newLegacyRuntimeRegistry() error = %v", err)
	}
	defer cleanup()

	if got := registry.CurrentConfig(); got != initialCfg {
		t.Fatalf("CurrentConfig() = %p, want %p", got, initialCfg)
	}
	if got := registry.ClassificationService(); got != initialSvc {
		t.Fatalf("ClassificationService() = %p, want %p", got, initialSvc)
	}
	if got := registry.MemoryStore(); got != initialStore {
		t.Fatalf("MemoryStore() = %v, want %v", got, initialStore)
	}
	if runtime := registry.VectorStoreRuntime(); runtime == nil || runtime.Manager != initialManager || runtime.FileStore != initialFileStore {
		t.Fatalf("VectorStoreRuntime() = %+v, want manager/file store snapshot", runtime)
	}

	updatedCfg := &config.RouterConfig{
		BackendModels: config.BackendModels{DefaultModel: "updated"},
	}
	updatedSvc := services.NewPlaceholderClassificationService()
	updatedStore := newMockMemoryStore()
	updatedManager := &vectorstore.Manager{}
	updatedFileStore := &vectorstore.FileStore{}

	services.SetGlobalClassificationService(updatedSvc)
	memory.SetGlobalMemoryStore(updatedStore)
	vectorStoreManager = updatedManager
	globalFileStore = updatedFileStore
	config.Replace(updatedCfg)

	waitForRegistryState(t, func() bool {
		runtime := registry.VectorStoreRuntime()
		return registry.CurrentConfig() == updatedCfg &&
			registry.ClassificationService() == updatedSvc &&
			registry.MemoryStore() == updatedStore &&
			runtime != nil &&
			runtime.Manager == updatedManager &&
			runtime.FileStore == updatedFileStore
	})
}

func waitForRegistryState(t *testing.T, ready func() bool) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if ready() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out waiting for registry state")
}

func testModelListConfig(autoModelName string, includeConfigModels bool, models []string) *config.RouterConfig {
	modelConfig := make(map[string]config.ModelParams, len(models))
	for _, model := range models {
		modelConfig[model] = config.ModelParams{
			PreferredEndpoints: []string{"primary"},
		}
	}

	return &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "primary",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
			},
			ModelConfig: modelConfig,
		},
		RouterOptions: config.RouterOptions{
			AutoModelName:             autoModelName,
			IncludeConfigModelsInList: includeConfigModels,
		},
	}
}

func decodeJSONObject(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	return payload
}

func requireJSONObject(t *testing.T, payload map[string]interface{}, key string) map[string]interface{} {
	t.Helper()

	value, ok := payload[key]
	if !ok {
		t.Fatalf("expected key %q in payload: %#v", key, payload)
	}
	return requireJSONObjectValue(t, value, key)
}

func requireJSONObjectValue(t *testing.T, value interface{}, label string) map[string]interface{} {
	t.Helper()

	object, ok := value.(map[string]interface{})
	if !ok {
		t.Fatalf("expected %s object, got %#v", label, value)
	}
	return object
}

func requireJSONArray(t *testing.T, payload map[string]interface{}, key string, expectedLen int) []interface{} {
	t.Helper()

	value, ok := payload[key]
	if !ok {
		t.Fatalf("expected key %q in payload: %#v", key, payload)
	}
	items, ok := value.([]interface{})
	if !ok {
		t.Fatalf("expected %s array, got %#v", key, value)
	}
	if len(items) != expectedLen {
		t.Fatalf("expected %s length %d, got %#v", key, expectedLen, value)
	}
	return items
}
