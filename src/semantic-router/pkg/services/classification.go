package services

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Global classification service instance
var (
	globalClassificationService *ClassificationService
	globalClassificationMu      sync.RWMutex
)

// ClassificationService provides classification functionality
type ClassificationService struct {
	classifier        *classification.Classifier
	unifiedClassifier *classification.UnifiedClassifier // New unified classifier
	config            *config.RouterConfig
	configMutex       sync.RWMutex // Protects config access
}

// NewClassificationService creates a new classification service
func NewClassificationService(classifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	return &ClassificationService{
		classifier:        classifier,
		unifiedClassifier: nil, // Will be initialized separately
		config:            config,
	}
}

// NewUnifiedClassificationService creates a new service with unified classifier
func NewUnifiedClassificationService(unifiedClassifier *classification.UnifiedClassifier, legacyClassifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	return &ClassificationService{
		classifier:        legacyClassifier,
		unifiedClassifier: unifiedClassifier,
		config:            config,
	}
}

// SetGlobalClassificationService publishes the compatibility global service used
// by legacy API-server startup paths that do not yet receive a runtime registry.
func SetGlobalClassificationService(service *ClassificationService) {
	globalClassificationMu.Lock()
	globalClassificationService = service
	globalClassificationMu.Unlock()
}

// NewClassificationServiceWithAutoDiscovery creates a service with auto-discovery
func NewClassificationServiceWithAutoDiscovery(config *config.RouterConfig) (*ClassificationService, error) {
	// Debug: Check current working directory
	wd, _ := os.Getwd()
	logging.Debugf("Debug: Current working directory: %s", wd)
	logging.Debugf("Debug: Attempting to discover models in: ./models")

	// Always try to auto-discover and initialize unified classifier for batch processing
	// Use model path from config, fallback to "./models" if not specified
	modelsPath := "./models"
	if config != nil && config.CategoryModel.ModelID != "" {
		// Extract the models directory from the model path
		// e.g., "models/mom-domain-classifier" -> "models"
		if idx := strings.Index(config.CategoryModel.ModelID, "/"); idx > 0 {
			modelsPath = config.CategoryModel.ModelID[:idx]
		}
	}

	// Pass mom_registry to auto-discovery for LoRA detection
	var modelRegistry map[string]string
	if config != nil {
		modelRegistry = config.MoMRegistry
	}
	unifiedClassifier, ucErr := classification.AutoInitializeUnifiedClassifierWithRegistry(modelsPath, modelRegistry)
	if ucErr != nil {
		logging.Infof("Unified classifier auto-discovery failed: %v", ucErr)
	}
	// create legacy classifier
	legacyClassifier, lcErr := classification.NewLegacyClassifierFromConfig(config)
	if lcErr != nil {
		logging.Warnf("Legacy classifier initialization failed: %v", lcErr)
		if config != nil {
			return nil, fmt.Errorf("configured classifier initialization failed: %w", lcErr)
		}
	}
	if unifiedClassifier == nil && legacyClassifier == nil {
		logging.Warnf("No classifier initialized. Using placeholder service.")
	}
	return NewUnifiedClassificationService(unifiedClassifier, legacyClassifier, config), nil
}

// GetGlobalClassificationService returns the global classification service instance
func GetGlobalClassificationService() *ClassificationService {
	globalClassificationMu.RLock()
	defer globalClassificationMu.RUnlock()
	return globalClassificationService
}

// HasClassifier returns true if the service has a real classifier (not placeholder)
func (s *ClassificationService) HasClassifier() bool {
	return s.classifier != nil
}

// UsesLocalNativeEmbeddings lets request entrypoints avoid consuming the
// process-local admission budget when every active embedding component is
// provider-backed or no real classifier is installed.
func (s *ClassificationService) UsesLocalNativeEmbeddings(hasImage bool) bool {
	return s != nil && s.classifier != nil && s.classifier.UsesLocalNativeEmbeddings(hasImage)
}

// NewPlaceholderClassificationService creates a placeholder service for API-only mode
func NewPlaceholderClassificationService() *ClassificationService {
	return &ClassificationService{
		classifier: nil, // No classifier - will return placeholder responses
		config:     nil,
	}
}

// ClassifyIntent performs intent classification using signal-driven architecture
func (s *ClassificationService) ClassifyIntent(req IntentRequest) (*IntentResponse, error) {
	start := time.Now()

	input, err := req.resolveSignalInput()
	if err != nil {
		return nil, err
	}

	// Check if classifier is available
	if s.classifier == nil {
		return placeholderIntentResponse(time.Since(start).Milliseconds()), nil
	}

	signals, err := s.evaluateIntentSignals(input, req.Options)
	if err != nil {
		return nil, err
	}

	decisionResult := s.evaluateIntentRoutingDecision(signals)
	category, confidence := s.resolveIntentCategory(input.evaluationText, decisionResult)

	processingTime := time.Since(start).Milliseconds()

	// Build response from signals and decision
	response := s.buildIntentResponseFromSignals(signals, decisionResult, category, confidence, processingTime, req)

	return response, nil
}

func placeholderIntentResponse(processingTime int64) *IntentResponse {
	return &IntentResponse{
		Classification: Classification{
			Category:         "general",
			Confidence:       0.5,
			ProcessingTimeMs: processingTime,
		},
		RecommendedModel: "general-model",
		RoutingDecision:  "placeholder_response",
	}
}

func (s *ClassificationService) evaluateIntentSignals(
	input intentSignalInput,
	options *IntentOptions,
) (*classification.SignalResults, error) {
	forceEvaluateAll := options != nil && options.EvaluateAllSignals
	signals := s.classifier.EvaluateAllSignalsWithContext(
		input.evaluationText,
		input.contextText,
		input.currentUserText,
		input.priorUserMessages,
		input.nonUserMessages,
		input.hasAssistantReply,
		forceEvaluateAll,
		"",
		nil,
		classification.ConversationFacts{},
		input.imageURL,
	)
	if evaluationErr := signals.EvaluationError(); evaluationErr != nil {
		if errors.Is(evaluationErr, classification.ErrInvalidImageSignalInput) {
			return nil, ErrInvalidImageInput
		}
		return nil, evaluationErr
	}
	return signals, nil
}

func (s *ClassificationService) evaluateIntentRoutingDecision(
	signals *classification.SignalResults,
) *decision.DecisionResult {
	if s.config == nil || len(s.config.Decisions) == 0 {
		return nil
	}

	decisionResult, err := s.classifier.EvaluateDecisionWithEngine(signals)
	if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
		logging.Warnf("Decision evaluation failed, continuing with classification: %v", err)
	}
	return decisionResult
}

func (s *ClassificationService) resolveIntentCategory(
	evaluationText string,
	decisionResult *decision.DecisionResult,
) (string, float64) {
	if decisionResult != nil && decisionResult.Decision != nil {
		return decisionResult.Decision.Name, decisionResult.Confidence
	}

	category, confidence, _, err := s.classifier.ClassifyCategoryWithEntropy(evaluationText)
	if err == nil {
		return category, confidence
	}

	// When domain evaluation was skipped due to low confidence and no decision
	// matches, retain the existing graceful fallback instead of failing the API.
	logging.Warnf("Classification fallback failed: %v, using default 'other' category", err)
	return "other", 0
}

// NOTE: ClassifyIntentUnified removed - ClassifyIntent now always uses signal-driven architecture
// For batch operations, use ClassifyBatchUnifiedWithOptions()

// GetClassifier returns the classifier instance (for signal-driven methods)
func (s *ClassificationService) GetClassifier() *classification.Classifier {
	return s.classifier
}

// GetConfig returns the current configuration
func (s *ClassificationService) GetConfig() *config.RouterConfig {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.config
}

// UpdateConfig updates the configuration
func (s *ClassificationService) UpdateConfig(newConfig *config.RouterConfig) {
	s.configMutex.Lock()
	defer s.configMutex.Unlock()
	s.config = newConfig
	// Update the global config as well
	config.Replace(newConfig)
}
