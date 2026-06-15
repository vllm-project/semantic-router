package services

import (
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
		// Return placeholder response
		processingTime := time.Since(start).Milliseconds()
		return &IntentResponse{
			Classification: Classification{
				Category:         "general",
				Confidence:       0.5,
				ProcessingTimeMs: processingTime,
			},
			RecommendedModel: "general-model",
			RoutingDecision:  "placeholder_response",
		}, nil
	}

	// Use signal-driven architecture: evaluate all signals first
	// Check if we should force evaluate all signals (for eval scenarios)
	forceEvaluateAll := req.Options != nil && req.Options.EvaluateAllSignals
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
	)

	// Evaluate decision with engine (if decisions are configured)
	// Pass pre-computed signals to avoid re-evaluation
	var decisionResult *decision.DecisionResult
	if s.config != nil && len(s.config.Decisions) > 0 {
		decisionResult, err = s.classifier.EvaluateDecisionWithEngine(signals)
		if err != nil {
			// Log error but continue with classification
			// Note: "no decisions configured" error is expected when decisions list is empty
			if !strings.Contains(err.Error(), "no decisions configured") {
				logging.Warnf("Decision evaluation failed, continuing with classification: %v", err)
			}
		}
	}

	// Get category classification (for backward compatibility and when no decision matches)
	var category string
	var confidence float64
	if decisionResult != nil && decisionResult.Decision != nil {
		// Use decision name as category
		category = decisionResult.Decision.Name
		confidence = decisionResult.Confidence
	} else {
		// Fallback to traditional classification
		category, confidence, _, err = s.classifier.ClassifyCategoryWithEntropy(input.evaluationText)
		if err != nil {
			// Graceful fallback when classification fails
			// When domain signal was skipped due to low confidence and no decision matches,
			// fall back to "other" category instead of returning an error
			logging.Warnf("Classification fallback failed: %v, using default 'other' category", err)
			category = "other"
			confidence = 0.0
		}
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response from signals and decision
	response := s.buildIntentResponseFromSignals(signals, decisionResult, category, confidence, processingTime, req)

	return response, nil
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
