package services

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ClassificationService provides classification functionality
type ClassificationService struct {
	classifier        *classification.Classifier
	recipeClassifiers *classification.RecipeClassifiers
	unifiedClassifier *classification.UnifiedClassifier // New unified classifier
	config            *config.RouterConfig
	configMutex       sync.RWMutex // Protects config access
	evalSelector      EvalModelSelector
}

func (s *ClassificationService) SetEvalModelSelector(selector EvalModelSelector) {
	if s == nil {
		return
	}
	s.configMutex.Lock()
	s.evalSelector = selector
	s.configMutex.Unlock()
}

func (s *ClassificationService) evalModelSelectorSnapshot() EvalModelSelector {
	if s == nil {
		return nil
	}
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.evalSelector
}

// NewRecipeClassificationService creates a model-aware service backed by
// isolated Recipe classifiers. Model-less calls use the optional explicit
// Recipe named "default" when one is published.
func NewRecipeClassificationService(classifiers *classification.RecipeClassifiers, routerConfig *config.RouterConfig) *ClassificationService {
	var defaultClassifier *classification.Classifier
	if classifiers != nil {
		defaultClassifier = classifiers.Default()
	}
	return &ClassificationService{
		classifier:        defaultClassifier,
		recipeClassifiers: classifiers,
		config:            routerConfig,
	}
}

// NewClassificationService creates a new classification service
func NewClassificationService(classifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	return &ClassificationService{
		classifier:        classifier,
		unifiedClassifier: nil, // Will be initialized separately
		config:            config,
	}
}

// NewUnifiedClassificationService creates an explicitly injected service.
// Runtime assembly uses NewRecipeClassificationService so named requests
// cannot be evaluated against a root-level classifier.
func NewUnifiedClassificationService(unifiedClassifier *classification.UnifiedClassifier, defaultClassifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	return &ClassificationService{
		classifier:        defaultClassifier,
		unifiedClassifier: unifiedClassifier,
		config:            config,
	}
}

// NewFileClassificationService creates the canonical file-authoritative
// composition: optional unified batch discovery plus an explicit Recipe
// classifier graph. It never constructs a classifier from root routing fields.
func NewFileClassificationService(config *config.RouterConfig) (*ClassificationService, error) {
	// Batch classification owns a separate optional unified model. Routing
	// classification is always compiled from explicit Recipes below.
	modelsPath := "./models"
	if config != nil && config.CategoryModel.ModelID != "" {
		if idx := strings.Index(config.CategoryModel.ModelID, "/"); idx > 0 {
			modelsPath = config.CategoryModel.ModelID[:idx]
		}
	}

	var modelRegistry map[string]string
	if config != nil {
		modelRegistry = config.MoMRegistry
	}
	unifiedClassifier, ucErr := classification.AutoInitializeUnifiedClassifierWithRegistry(modelsPath, modelRegistry)
	if ucErr != nil {
		logging.Infof("Unified classifier auto-discovery failed: %v", ucErr)
	}
	recipeClassifiers, err := classification.NewRecipeClassifiersFromConfig(config)
	if err != nil {
		return nil, fmt.Errorf("initialize file-authority Recipe classifiers: %w", err)
	}
	service := NewRecipeClassificationService(recipeClassifiers, config)
	service.unifiedClassifier = unifiedClassifier
	return service, nil
}

// HasClassifier returns true if the service has a real classifier (not placeholder)
func (s *ClassificationService) HasClassifier() bool {
	return s.classifierSnapshot() != nil
}

func (s *ClassificationService) classifierSnapshot() *classification.Classifier {
	classifier, _ := s.runtimeSnapshot()
	return classifier
}

func (s *ClassificationService) runtimeSnapshot() (
	*classification.Classifier,
	*config.RouterConfig,
) {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.classifier, s.config
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
	classifier, runtimeConfig, err := s.runtimeSnapshotForRequestModel(
		req.Model,
	)
	if err != nil {
		return nil, err
	}
	// Check if classifier is available
	if classifier == nil {
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
	signals := classifier.EvaluateAllSignalsWithRequestFacts(
		input.evaluationText,
		input.contextText,
		input.currentUserText,
		input.priorUserMessages,
		input.nonUserMessages,
		input.hasAssistantReply,
		forceEvaluateAll,
		"",
		nil,
		input.conversationFacts,
		input.imageURL,
		input.requestFacts,
	)

	// Evaluate decision with engine (if decisions are configured)
	// Pass pre-computed signals to avoid re-evaluation
	var decisionResult *decision.DecisionResult
	if classifier.Config != nil && len(classifier.Config.Decisions) > 0 {
		decisionResult, err = classifier.EvaluateDecisionWithEngine(signals)
		if err != nil {
			// Log error but continue with classification
			// Note: "no decisions configured" error is expected when decisions list is empty
			if !strings.Contains(err.Error(), "no decisions configured") {
				logging.Warnf("Decision evaluation failed, continuing with classification: %v", err)
			}
		}
	}

	category, confidence := resolveIntentCategory(
		classifier,
		decisionResult,
		input.evaluationText,
	)

	processingTime := time.Since(start).Milliseconds()

	// Build response from signals and decision
	response := s.buildIntentResponseFromSignals(
		signals,
		decisionResult,
		category,
		confidence,
		processingTime,
		req,
		classifier,
		runtimeConfig,
	)

	return response, nil
}

func (s *ClassificationService) runtimeSnapshotForRequestModel(
	modelName string,
) (*classification.Classifier, *config.RouterConfig, error) {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	classifier, err := s.classifierForRequestModel(modelName)
	return classifier, s.config, err
}

func (s *ClassificationService) classifierForRequestModel(modelName string) (*classification.Classifier, error) {
	if s == nil || s.recipeClassifiers == nil || s.config == nil {
		if s == nil {
			return nil, nil
		}
		return s.classifier, nil
	}
	recipe, requestedModel, ok := s.recipeForClassificationModel(modelName)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrUnknownRoutingModel, requestedModel)
	}
	classifier, ok := s.recipeClassifiers.ForRecipe(recipe.Name)
	if !ok {
		return nil, fmt.Errorf("classifier for routing recipe %q is unavailable", recipe.Name)
	}
	return classifier, nil
}

// recipeForClassificationModel keeps the model-less classification API
// separate from inference routing. An empty model may use the optional
// explicit default Recipe; every named model must be an Entrypoint.
func (s *ClassificationService) recipeForClassificationModel(modelName string) (*config.RoutingRecipe, string, bool) {
	requestedModel := strings.TrimSpace(modelName)
	if requestedModel == "" {
		recipe := s.config.DefaultRecipe()
		return recipe, requestedModel, recipe != nil
	}
	recipe, ok := s.config.RecipeForRequestModel(requestedModel)
	return recipe, requestedModel, ok
}

// NOTE: ClassifyIntentUnified removed - ClassifyIntent now always uses signal-driven architecture
// For batch operations, use ClassifyBatchUnifiedWithOptions()

// GetClassifier returns the classifier instance (for signal-driven methods)
func (s *ClassificationService) GetClassifier() *classification.Classifier {
	return s.classifierSnapshot()
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
