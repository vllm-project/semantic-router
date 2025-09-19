package extproc_test

import (
	"context"
	"fmt"
	"io"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/metadata"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// MockStream implements the ext_proc.ExternalProcessor_ProcessServer interface for testing
type MockStream struct {
	Requests  []*ext_proc.ProcessingRequest
	Responses []*ext_proc.ProcessingResponse
	Ctx       context.Context
	SendError error
	RecvError error
	RecvIndex int
}

func NewMockStream(requests []*ext_proc.ProcessingRequest) *MockStream {
	return &MockStream{
		Requests:  requests,
		Responses: make([]*ext_proc.ProcessingResponse, 0),
		Ctx:       context.Background(),
		RecvIndex: 0,
	}
}

func (m *MockStream) Send(response *ext_proc.ProcessingResponse) error {
	if m.SendError != nil {
		return m.SendError
	}
	m.Responses = append(m.Responses, response)
	return nil
}

func (m *MockStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if m.RecvError != nil {
		return nil, m.RecvError
	}
	if m.RecvIndex >= len(m.Requests) {
		return nil, io.EOF // Simulate end of stream
	}
	req := m.Requests[m.RecvIndex]
	m.RecvIndex++
	return req, nil
}

func (m *MockStream) Context() context.Context {
	return m.Ctx
}

func (m *MockStream) SendMsg(interface{}) error    { return nil }
func (m *MockStream) RecvMsg(interface{}) error    { return nil }
func (m *MockStream) SetHeader(metadata.MD) error  { return nil }
func (m *MockStream) SendHeader(metadata.MD) error { return nil }
func (m *MockStream) SetTrailer(metadata.MD)       {}

var _ ext_proc.ExternalProcessor_ProcessServer = &MockStream{}

// CreateTestConfig creates a standard test configuration
func CreateTestConfig() *config.RouterConfig {
	return &config.RouterConfig{
		BertModel: struct {
			ModelID   string  `yaml:"model_id"`
			Threshold float32 `yaml:"threshold"`
			UseCPU    bool    `yaml:"use_cpu"`
		}{
			ModelID:   "sentence-transformers/all-MiniLM-L12-v2",
			Threshold: 0.8,
			UseCPU:    true,
		},
		Classifier: struct {
			CategoryModel struct {
				ModelID             string  `yaml:"model_id"`
				Threshold           float32 `yaml:"threshold"`
				UseCPU              bool    `yaml:"use_cpu"`
				UseModernBERT       bool    `yaml:"use_modernbert"`
				CategoryMappingPath string  `yaml:"category_mapping_path"`
			} `yaml:"category_model"`
			PIIModel struct {
				ModelID        string  `yaml:"model_id"`
				Threshold      float32 `yaml:"threshold"`
				UseCPU         bool    `yaml:"use_cpu"`
				PIIMappingPath string  `yaml:"pii_mapping_path"`
			} `yaml:"pii_model"`
		}{
			CategoryModel: struct {
				ModelID             string  `yaml:"model_id"`
				Threshold           float32 `yaml:"threshold"`
				UseCPU              bool    `yaml:"use_cpu"`
				UseModernBERT       bool    `yaml:"use_modernbert"`
				CategoryMappingPath string  `yaml:"category_mapping_path"`
			}{
				ModelID:             "../../../../models/category_classifier_modernbert-base_model",
				UseCPU:              true,
				UseModernBERT:       true,
				CategoryMappingPath: "../../../../models/category_classifier_modernbert-base_model/category_mapping.json",
			},
			PIIModel: struct {
				ModelID        string  `yaml:"model_id"`
				Threshold      float32 `yaml:"threshold"`
				UseCPU         bool    `yaml:"use_cpu"`
				PIIMappingPath string  `yaml:"pii_mapping_path"`
			}{
				ModelID:        "../../../../models/pii_classifier_modernbert-base_presidio_token_model",
				UseCPU:         true,
				PIIMappingPath: "../../../../models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json",
			},
		},
		Categories: []config.Category{
			{
				Name:        "coding",
				Description: "Programming tasks",
				ModelScores: []config.ModelScore{
					{Model: "model-a", Score: 0.9},
					{Model: "model-b", Score: 0.8},
				},
			},
		},
		DefaultModel: "model-b",
		SemanticCache: struct {
			BackendType         string   `yaml:"backend_type,omitempty"`
			Enabled             bool     `yaml:"enabled"`
			SimilarityThreshold *float32 `yaml:"similarity_threshold,omitempty"`
			MaxEntries          int      `yaml:"max_entries,omitempty"`
			TTLSeconds          int      `yaml:"ttl_seconds,omitempty"`
			EvictionPolicy      string   `yaml:"eviction_policy,omitempty"`
			BackendConfigPath   string   `yaml:"backend_config_path,omitempty"`
		}{
			BackendType:         "memory",
			Enabled:             false, // Disable for most tests
			SimilarityThreshold: &[]float32{0.9}[0],
			MaxEntries:          100,
			EvictionPolicy:      "lru",
			TTLSeconds:          3600,
		},
		PromptGuard: config.PromptGuardConfig{
			Enabled:   false, // Disable for most tests
			ModelID:   "test-jailbreak-model",
			Threshold: 0.5,
		},
		ModelConfig: map[string]config.ModelParams{
			"model-a": {
				PIIPolicy: config.PIIPolicy{
					AllowByDefault: true,
				},
				PreferredEndpoints: []string{"test-endpoint1"},
			},
			"model-b": {
				PIIPolicy: config.PIIPolicy{
					AllowByDefault: true,
				},
				PreferredEndpoints: []string{"test-endpoint1", "test-endpoint2"},
			},
		},
		Tools: config.ToolsConfig{
			Enabled:         false, // Disable for most tests
			TopK:            3,
			ToolsDBPath:     "",
			FallbackToEmpty: true,
		},
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:            "test-endpoint1",
				Address:         "127.0.0.1",
				Port:            8000,
				Models:          []string{"model-a", "model-b"},
				Weight:          1,
				HealthCheckPath: "/health",
			},
			{
				Name:            "test-endpoint2",
				Address:         "127.0.0.1",
				Port:            8001,
				Models:          []string{"model-b"},
				Weight:          2,
				HealthCheckPath: "/health",
			},
		},
	}
}

// CreateTestRouter creates a properly initialized router for testing
func CreateTestRouter(cfg *config.RouterConfig) (*extproc.OpenAIRouter, error) {
	// Create mock components
	categoryMapping, err := classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
	if err != nil {
		return nil, err
	}

	piiMapping, err := classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
	if err != nil {
		return nil, err
	}

	// Initialize the BERT model for similarity search
	if err := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU); err != nil {
		return nil, fmt.Errorf("failed to initialize BERT model: %w", err)
	}

	// Create semantic cache
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.InMemoryCacheType,
		Enabled:             cfg.SemanticCache.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.SemanticCache.MaxEntries,
		TTLSeconds:          cfg.SemanticCache.TTLSeconds,
		EvictionPolicy:      cache.EvictionPolicyType(cfg.SemanticCache.EvictionPolicy),
	}
	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, err
	}

	// Create tools database
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: cfg.BertModel.Threshold,
		Enabled:             cfg.Tools.Enabled,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	// Create classifier
	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, nil)
	if err != nil {
		return nil, err
	}

	// Create PII checker
	piiChecker := pii.NewPolicyChecker(cfg, cfg.ModelConfig)

	// Create router manually with proper initialization
	router := &extproc.OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: cfg.GetCategoryDescriptions(),
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
	}

	return router, nil
}
