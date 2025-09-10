package extproc_test

import (
	"context"
	"fmt"
	"io"
	"log"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/metadata"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/utils/classification"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/utils/pii"
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
		SemanticCache: config.SemanticCacheConfig{
			Enabled:             false, // Disable for most tests
			SimilarityThreshold: &[]float32{0.9}[0],
			MaxEntries:          100,
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

	// Initialize models using candle-binding
	err = initializeTestModels(cfg, categoryMapping, piiMapping)
	if err != nil {
		return nil, err
	}

	// Create semantic cache
	cacheOptions := cache.SemanticCacheOptions{
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.SemanticCache.MaxEntries,
		TTLSeconds:          cfg.SemanticCache.TTLSeconds,
		Enabled:             cfg.SemanticCache.Enabled,
	}
	semanticCache := cache.NewSemanticCache(cacheOptions)

	// Create tools database
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: cfg.BertModel.Threshold,
		Enabled:             cfg.Tools.Enabled,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	// Create classifier
	classifier := classification.NewClassifier(cfg, categoryMapping, piiMapping, nil)

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

	// Initialize internal fields for testing
	router.InitializeForTesting()

	return router, nil
}

// initializeTestModels initializes the BERT and classifier models for testing
func initializeTestModels(cfg *config.RouterConfig, categoryMapping *classification.CategoryMapping, piiMapping *classification.PIIMapping) error {
	// Initialize the BERT model for similarity search
	err := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize BERT model: %w", err)
	}

	// Initialize the classifier model if enabled
	if categoryMapping != nil {
		// Get the number of categories from the mapping
		numClasses := categoryMapping.GetCategoryCount()
		if numClasses < 2 {
			log.Printf("Warning: Not enough categories for classification, need at least 2, got %d", numClasses)
		} else {
			// Use the category classifier model
			classifierModelID := cfg.Classifier.CategoryModel.ModelID
			if classifierModelID == "" {
				classifierModelID = cfg.BertModel.ModelID
			}

			if cfg.Classifier.CategoryModel.UseModernBERT {
				// Initialize ModernBERT classifier
				err = candle_binding.InitModernBertClassifier(classifierModelID, cfg.Classifier.CategoryModel.UseCPU)
				if err != nil {
					return fmt.Errorf("failed to initialize ModernBERT classifier model: %w", err)
				}
				log.Printf("Initialized ModernBERT category classifier (classes auto-detected from model)")
			} else {
				// Initialize linear classifier
				err = candle_binding.InitClassifier(classifierModelID, numClasses, cfg.Classifier.CategoryModel.UseCPU)
				if err != nil {
					return fmt.Errorf("failed to initialize classifier model: %w", err)
				}
				log.Printf("Initialized linear category classifier with %d categories", numClasses)
			}
		}
	}

	// Initialize PII classifier if enabled
	if piiMapping != nil {
		// Get the number of PII types from the mapping
		numPIIClasses := piiMapping.GetPIITypeCount()
		if numPIIClasses < 2 {
			log.Printf("Warning: Not enough PII types for classification, need at least 2, got %d", numPIIClasses)
		} else {
			// Use the PII classifier model
			piiClassifierModelID := cfg.Classifier.PIIModel.ModelID
			if piiClassifierModelID == "" {
				piiClassifierModelID = cfg.BertModel.ModelID
			}

			// Initialize ModernBERT PII token classifier for entity detection
			err = candle_binding.InitModernBertPIITokenClassifier(piiClassifierModelID, cfg.Classifier.PIIModel.UseCPU)
			if err != nil {
				return fmt.Errorf("failed to initialize ModernBERT PII token classifier model: %w", err)
			}
			log.Printf("Initialized ModernBERT PII token classifier for entity detection")
		}
	}

	return nil
}
