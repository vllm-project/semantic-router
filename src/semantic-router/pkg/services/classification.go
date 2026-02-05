package services

import (
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
var globalClassificationService *ClassificationService

// ClassificationService provides classification functionality
type ClassificationService struct {
	classifier        *classification.Classifier
	unifiedClassifier *classification.UnifiedClassifier // New unified classifier
	config            *config.RouterConfig
	configMutex       sync.RWMutex // Protects config access
}

// NewClassificationService creates a new classification service
func NewClassificationService(classifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	service := &ClassificationService{
		classifier:        classifier,
		unifiedClassifier: nil, // Will be initialized separately
		config:            config,
	}
	// Set as global service for API access
	globalClassificationService = service
	return service
}

// NewUnifiedClassificationService creates a new service with unified classifier
func NewUnifiedClassificationService(unifiedClassifier *classification.UnifiedClassifier, legacyClassifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	service := &ClassificationService{
		classifier:        legacyClassifier,
		unifiedClassifier: unifiedClassifier,
		config:            config,
	}
	// Set as global service for API access
	globalClassificationService = service
	return service
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
	legacyClassifier, lcErr := createLegacyClassifier(config)
	if lcErr != nil {
		logging.Warnf("Legacy classifier initialization failed: %v", lcErr)
	}
	if unifiedClassifier == nil && legacyClassifier == nil {
		logging.Warnf("No classifier initialized. Using placeholder service.")
	}
	return NewUnifiedClassificationService(unifiedClassifier, legacyClassifier, config), nil
}

// createLegacyClassifier creates a legacy classifier with proper model loading
func createLegacyClassifier(config *config.RouterConfig) (*classification.Classifier, error) {
	// Load category mapping
	var categoryMapping *classification.CategoryMapping

	// Check if we should load categories from MCP server
	// Note: tool_name is optional and will be auto-discovered if not specified
	useMCPCategories := config.CategoryModel.ModelID == "" &&
		config.MCPCategoryModel.Enabled

	if useMCPCategories {
		// Categories will be loaded from MCP server during initialization
		logging.Infof("Category mapping will be loaded from MCP server")
		// Create empty mapping initially - will be populated during initialization
		categoryMapping = nil
	} else if config.CategoryMappingPath != "" {
		// Load from file as usual
		var err error
		categoryMapping, err = classification.LoadCategoryMapping(config.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}

	// Load PII mapping
	var piiMapping *classification.PIIMapping
	if config.PIIMappingPath != "" {
		var err error
		piiMapping, err = classification.LoadPIIMapping(config.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}

	// Load jailbreak mapping
	var jailbreakMapping *classification.JailbreakMapping
	if config.PromptGuard.JailbreakMappingPath != "" {
		var err error
		jailbreakMapping, err = classification.LoadJailbreakMapping(config.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}

	// Create classifier
	classifier, err := classification.NewClassifier(config, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	return classifier, nil
}

// GetGlobalClassificationService returns the global classification service instance
func GetGlobalClassificationService() *ClassificationService {
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

// IntentRequest represents a request for intent classification
type IntentRequest struct {
	Text    string         `json:"text"`
	Options *IntentOptions `json:"options,omitempty"`
}

// IntentOptions contains options for intent classification
type IntentOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
	EvaluateAllSignals  bool    `json:"evaluate_all_signals,omitempty"` // Force evaluate all configured signals (for eval scenarios)
}

// MatchedSignals represents all matched signals from signal evaluation
type MatchedSignals struct {
	Keywords     []string `json:"keywords,omitempty"`
	Embeddings   []string `json:"embeddings,omitempty"`
	Domains      []string `json:"domains,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Preferences  []string `json:"preferences,omitempty"`
	Language     []string `json:"language,omitempty"`
	Latency      []string `json:"latency,omitempty"`
	Context      []string `json:"context,omitempty"`
	Complexity   []string `json:"complexity,omitempty"`
}

// DecisionResult represents the result of decision evaluation
type DecisionResult struct {
	DecisionName string   `json:"decision_name"`
	Confidence   float64  `json:"confidence"`
	MatchedRules []string `json:"matched_rules"`
}

// EvalDecisionResult represents the decision result for eval scenarios (without confidence)
type EvalDecisionResult struct {
	DecisionName     string          `json:"decision_name"`
	UsedSignals      *MatchedSignals `json:"used_signals"`      // Signals used by this decision (from decision rules)
	MatchedSignals   *MatchedSignals `json:"matched_signals"`   // Signals that matched
	UnmatchedSignals *MatchedSignals `json:"unmatched_signals"` // Signals that didn't match
}

// EvalResponse represents the response from eval classification
// This is specifically designed for evaluation scenarios with comprehensive signal information
type EvalResponse struct {
	OriginalText      string                                  `json:"original_text"` // The original query text
	DecisionResult    *EvalDecisionResult                     `json:"decision_result,omitempty"`
	RecommendedModels []string                                `json:"recommended_models,omitempty"` // All models from matched decision's modelRefs
	RoutingDecision   string                                  `json:"routing_decision,omitempty"`
	Metrics           *classification.SignalMetricsCollection `json:"metrics"` // Performance and confidence for each signal
}

// IntentResponse represents the response from intent classification
type IntentResponse struct {
	Classification   Classification     `json:"classification"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RecommendedModel string             `json:"recommended_model,omitempty"`
	RoutingDecision  string             `json:"routing_decision,omitempty"`

	// Signal-driven fields
	MatchedSignals *MatchedSignals `json:"matched_signals,omitempty"`
	DecisionResult *DecisionResult `json:"decision_result,omitempty"`
}

// Classification represents basic classification result
type Classification struct {
	Category         string  `json:"category"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// buildIntentResponseFromSignals builds an IntentResponse from signals and decision result
func (s *ClassificationService) buildIntentResponseFromSignals(
	signals *classification.SignalResults,
	decisionResult *decision.DecisionResult,
	category string,
	confidence float64,
	processingTime int64,
	req IntentRequest,
) *IntentResponse {
	response := &IntentResponse{
		Classification: Classification{
			Category:         category,
			Confidence:       confidence,
			ProcessingTimeMs: processingTime,
		},
	}

	// Add probabilities if requested
	if req.Options != nil && req.Options.ReturnProbabilities {
		response.Probabilities = map[string]float64{
			category: confidence,
		}
	}

	// Add recommended model based on category or decision
	if decisionResult != nil && decisionResult.Decision != nil && len(decisionResult.Decision.ModelRefs) > 0 {
		modelRef := decisionResult.Decision.ModelRefs[0]
		if modelRef.LoRAName != "" {
			response.RecommendedModel = modelRef.LoRAName
		} else {
			response.RecommendedModel = modelRef.Model
		}
	} else if model := s.getRecommendedModel(category, confidence); model != "" {
		response.RecommendedModel = model
	}

	// Determine routing decision
	if decisionResult != nil && decisionResult.Decision != nil {
		response.RoutingDecision = decisionResult.Decision.Name
	} else {
		response.RoutingDecision = s.getRoutingDecision(confidence, req.Options)
	}

	// Add signal information
	if signals != nil {
		response.MatchedSignals = &MatchedSignals{
			Keywords:     signals.MatchedKeywordRules,
			Embeddings:   signals.MatchedEmbeddingRules,
			Domains:      signals.MatchedDomainRules,
			FactCheck:    signals.MatchedFactCheckRules,
			UserFeedback: signals.MatchedUserFeedbackRules,
			Preferences:  signals.MatchedPreferenceRules,
			Language:     signals.MatchedLanguageRules,
			Latency:      signals.MatchedLatencyRules,
			Context:      signals.MatchedContextRules,
			Complexity:   signals.MatchedComplexityRules,
		}
	}

	// Add decision result
	if decisionResult != nil && decisionResult.Decision != nil {
		response.DecisionResult = &DecisionResult{
			DecisionName: decisionResult.Decision.Name,
			Confidence:   decisionResult.Confidence,
			MatchedRules: decisionResult.MatchedRules,
		}
	}

	return response
}

// ClassifyIntent performs intent classification using signal-driven architecture
func (s *ClassificationService) ClassifyIntent(req IntentRequest) (*IntentResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
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
	signals := s.classifier.EvaluateAllSignalsWithForceOption(req.Text, forceEvaluateAll)

	// Evaluate decision with engine (if decisions are configured)
	// Pass pre-computed signals to avoid re-evaluation
	var decisionResult *decision.DecisionResult
	var err error
	if s.config != nil && len(s.config.IntelligentRouting.Decisions) > 0 {
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
		category, confidence, _, err = s.classifier.ClassifyCategoryWithEntropy(req.Text)
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

// PIIRequest represents a request for PII detection
type PIIRequest struct {
	Text    string      `json:"text"`
	Options *PIIOptions `json:"options,omitempty"`
}

// PIIOptions contains options for PII detection
type PIIOptions struct {
	EntityTypes         []string `json:"entity_types,omitempty"`
	ConfidenceThreshold float64  `json:"confidence_threshold,omitempty"`
	ReturnPositions     bool     `json:"return_positions,omitempty"`
	MaskEntities        bool     `json:"mask_entities,omitempty"`
}

// PIIResponse represents the response from PII detection
type PIIResponse struct {
	HasPII                 bool        `json:"has_pii"`
	Entities               []PIIEntity `json:"entities"`
	MaskedText             string      `json:"masked_text,omitempty"`
	SecurityRecommendation string      `json:"security_recommendation"`
	ProcessingTimeMs       int64       `json:"processing_time_ms"`
}

// PIIEntity represents a detected PII entity
type PIIEntity struct {
	Type        string  `json:"type"`
	Value       string  `json:"value"`
	Confidence  float64 `json:"confidence"`
	StartPos    int     `json:"start_position,omitempty"`
	EndPos      int     `json:"end_position,omitempty"`
	MaskedValue string  `json:"masked_value,omitempty"`
}

// DetectPII performs PII detection
func (s *ClassificationService) DetectPII(req PIIRequest) (*PIIResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		// Return placeholder response
		processingTime := time.Since(start).Milliseconds()
		return &PIIResponse{
			HasPII:                 false,
			Entities:               []PIIEntity{},
			SecurityRecommendation: "allow",
			ProcessingTimeMs:       processingTime,
		}, nil
	}

	// Perform PII detection using the classifier with full details
	detections, err := s.classifier.ClassifyPIIWithDetails(req.Text)
	if err != nil {
		return nil, fmt.Errorf("PII detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &PIIResponse{
		HasPII:           len(detections) > 0,
		Entities:         []PIIEntity{},
		ProcessingTimeMs: processingTime,
	}

	// Convert PII detections to API entities with actual confidence scores
	for _, detection := range detections {
		entity := PIIEntity{
			Type:       detection.EntityType,
			Value:      "[DETECTED]",                  // Redacted for security
			Confidence: float64(detection.Confidence), // Actual confidence from model
			StartPos:   detection.Start,
			EndPos:     detection.End,
		}
		response.Entities = append(response.Entities, entity)
	}

	// Set security recommendation
	if response.HasPII {
		response.SecurityRecommendation = "block"
	} else {
		response.SecurityRecommendation = "allow"
	}

	return response, nil
}

// SecurityRequest represents a request for security detection
type SecurityRequest struct {
	Text    string           `json:"text"`
	Options *SecurityOptions `json:"options,omitempty"`
}

// SecurityOptions contains options for security detection
type SecurityOptions struct {
	DetectionTypes   []string `json:"detection_types,omitempty"`
	Sensitivity      string   `json:"sensitivity,omitempty"`
	IncludeReasoning bool     `json:"include_reasoning,omitempty"`
}

// SecurityResponse represents the response from security detection
type SecurityResponse struct {
	IsJailbreak      bool     `json:"is_jailbreak"`
	RiskScore        float64  `json:"risk_score"`
	DetectionTypes   []string `json:"detection_types"`
	Confidence       float64  `json:"confidence"`
	Recommendation   string   `json:"recommendation"`
	Reasoning        string   `json:"reasoning,omitempty"`
	PatternsDetected []string `json:"patterns_detected"`
	ProcessingTimeMs int64    `json:"processing_time_ms"`
}

// CheckSecurity performs security detection
func (s *ClassificationService) CheckSecurity(req SecurityRequest) (*SecurityResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		// Return placeholder response
		processingTime := time.Since(start).Milliseconds()
		return &SecurityResponse{
			IsJailbreak:      false,
			RiskScore:        0.1,
			DetectionTypes:   []string{},
			Confidence:       0.9,
			Recommendation:   "allow",
			PatternsDetected: []string{},
			ProcessingTimeMs: processingTime,
		}, nil
	}

	// Perform jailbreak detection using the existing classifier
	isJailbreak, jailbreakType, confidence, err := s.classifier.CheckForJailbreak(req.Text)
	if err != nil {
		return nil, fmt.Errorf("security detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &SecurityResponse{
		IsJailbreak:      isJailbreak,
		RiskScore:        float64(confidence),
		Confidence:       float64(confidence),
		ProcessingTimeMs: processingTime,
		DetectionTypes:   []string{},
		PatternsDetected: []string{},
	}

	if isJailbreak {
		response.DetectionTypes = append(response.DetectionTypes, jailbreakType)
		response.PatternsDetected = append(response.PatternsDetected, jailbreakType)
		response.Recommendation = "block"
		if req.Options != nil && req.Options.IncludeReasoning {
			response.Reasoning = fmt.Sprintf("Detected %s pattern with confidence %.3f", jailbreakType, confidence)
		}
	} else {
		response.Recommendation = "allow"
	}

	return response, nil
}

// Helper methods
func (s *ClassificationService) getRecommendedModel(category string, _ float64) string {
	// Use classifier's existing logic if available
	if s.classifier != nil {
		model := s.classifier.SelectBestModelForCategory(category)
		if model != "" {
			return model
		}
	}

	// Fallback: Access config directly to find decision and model
	if s.config != nil {
		// Find decision by category name (case-insensitive)
		for _, decision := range s.config.IntelligentRouting.Decisions {
			if strings.EqualFold(decision.Name, category) {
				// Get first model from ModelRefs
				if len(decision.ModelRefs) > 0 {
					modelRef := decision.ModelRefs[0]
					// Use LoRA name if specified, otherwise base model
					if modelRef.LoRAName != "" {
						return modelRef.LoRAName
					}
					return modelRef.Model
				}
				break
			}
		}

		// Fallback to default model if no decision found
		if s.config.BackendModels.DefaultModel != "" {
			return s.config.BackendModels.DefaultModel
		}
	}

	// Return empty string if no recommendation available
	return ""
}

func (s *ClassificationService) getRoutingDecision(confidence float64, options *IntentOptions) string {
	threshold := 0.7 // default threshold
	if options != nil && options.ConfidenceThreshold > 0 {
		threshold = options.ConfidenceThreshold
	}

	if confidence >= threshold {
		return "high_confidence_specialized"
	}
	return "low_confidence_general"
}

// UnifiedBatchResponse represents the response from unified batch classification
type UnifiedBatchResponse struct {
	IntentResults    []classification.IntentResult   `json:"intent_results"`
	PIIResults       []classification.PIIResult      `json:"pii_results"`
	SecurityResults  []classification.SecurityResult `json:"security_results"`
	ProcessingTimeMs int64                           `json:"processing_time_ms"`
	TotalTexts       int                             `json:"total_texts"`
}

// ClassifyBatchUnified performs unified batch classification using the new architecture
func (s *ClassificationService) ClassifyBatchUnified(texts []string) (*UnifiedBatchResponse, error) {
	return s.ClassifyBatchUnifiedWithOptions(texts, nil)
}

// ClassifyBatchUnifiedWithOptions performs unified batch classification with options support
func (s *ClassificationService) ClassifyBatchUnifiedWithOptions(texts []string, _ interface{}) (*UnifiedBatchResponse, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("texts cannot be empty")
	}

	// Check if unified classifier is available
	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	start := time.Now()

	// Direct call to unified classifier - no complex scheduling needed!
	results, err := s.unifiedClassifier.ClassifyBatch(texts)
	if err != nil {
		return nil, fmt.Errorf("unified batch classification failed: %w", err)
	}

	// Build response
	response := &UnifiedBatchResponse{
		IntentResults:    results.IntentResults,
		PIIResults:       results.PIIResults,
		SecurityResults:  results.SecurityResults,
		ProcessingTimeMs: time.Since(start).Milliseconds(),
		TotalTexts:       len(texts),
	}

	return response, nil
}

// NOTE: ClassifyIntentUnified removed - ClassifyIntent now always uses signal-driven architecture
// For batch operations, use ClassifyBatchUnifiedWithOptions()

// ClassifyPIIUnified performs PII detection using unified classifier
func (s *ClassificationService) ClassifyPIIUnified(texts []string) ([]classification.PIIResult, error) {
	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	results, err := s.ClassifyBatchUnified(texts)
	if err != nil {
		return nil, err
	}

	return results.PIIResults, nil
}

// ClassifySecurityUnified performs security detection using unified classifier
func (s *ClassificationService) ClassifySecurityUnified(texts []string) ([]classification.SecurityResult, error) {
	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	results, err := s.ClassifyBatchUnified(texts)
	if err != nil {
		return nil, err
	}

	return results.SecurityResults, nil
}

// HasUnifiedClassifier returns true if the service has a unified classifier
func (s *ClassificationService) HasUnifiedClassifier() bool {
	return s.unifiedClassifier != nil && s.unifiedClassifier.IsInitialized()
}

// GetUnifiedClassifierStats returns statistics about the unified classifier
func (s *ClassificationService) GetUnifiedClassifierStats() map[string]interface{} {
	if s.unifiedClassifier == nil {
		return map[string]interface{}{
			"available": false,
		}
	}

	stats := s.unifiedClassifier.GetStats()
	stats["available"] = true
	return stats
}

// GetClassifier returns the classifier instance (for signal-driven methods)
func (s *ClassificationService) GetClassifier() *classification.Classifier {
	return s.classifier
}

// FactCheckRequest represents a request for fact-check classification
type FactCheckRequest struct {
	Text    string            `json:"text"`
	Options *FactCheckOptions `json:"options,omitempty"`
}

// FactCheckOptions contains options for fact-check classification
type FactCheckOptions struct {
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// FactCheckResponse represents the response from fact-check classification
type FactCheckResponse struct {
	NeedsFactCheck   bool    `json:"needs_fact_check"`
	Label            string  `json:"label"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// ClassifyFactCheck performs fact-check classification
func (s *ClassificationService) ClassifyFactCheck(req FactCheckRequest) (*FactCheckResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		processingTime := time.Since(start).Milliseconds()
		return &FactCheckResponse{
			NeedsFactCheck:   false,
			Label:            "unknown",
			Confidence:       0.0,
			ProcessingTimeMs: processingTime,
		}, nil
	}

	// Check if fact-check classifier is enabled
	if !s.classifier.IsFactCheckEnabled() {
		processingTime := time.Since(start).Milliseconds()
		return &FactCheckResponse{
			NeedsFactCheck:   false,
			Label:            "fact_check_disabled",
			Confidence:       0.0,
			ProcessingTimeMs: processingTime,
		}, nil
	}

	// Perform fact-check classification
	result, err := s.classifier.ClassifyFactCheck(req.Text)
	if err != nil {
		return nil, fmt.Errorf("fact-check classification failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	return &FactCheckResponse{
		NeedsFactCheck:   result.NeedsFactCheck,
		Label:            result.Label,
		Confidence:       float64(result.Confidence),
		ProcessingTimeMs: processingTime,
	}, nil
}

// UserFeedbackRequest represents a request for user feedback classification
type UserFeedbackRequest struct {
	Text    string               `json:"text"`
	Options *UserFeedbackOptions `json:"options,omitempty"`
}

// UserFeedbackOptions contains options for user feedback classification
type UserFeedbackOptions struct {
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// UserFeedbackResponse represents the response from user feedback classification
type UserFeedbackResponse struct {
	FeedbackType     string  `json:"feedback_type"`
	Label            string  `json:"label"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// ClassifyUserFeedback performs user feedback classification
func (s *ClassificationService) ClassifyUserFeedback(req UserFeedbackRequest) (*UserFeedbackResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		processingTime := time.Since(start).Milliseconds()
		return &UserFeedbackResponse{
			FeedbackType:     "unknown",
			Label:            "unknown",
			Confidence:       0.0,
			ProcessingTimeMs: processingTime,
		}, nil
	}

	// Check if feedback detector is enabled
	if !s.classifier.IsFeedbackDetectorEnabled() {
		processingTime := time.Since(start).Milliseconds()
		return &UserFeedbackResponse{
			FeedbackType:     "feedback_detector_disabled",
			Label:            "feedback_detector_disabled",
			Confidence:       0.0,
			ProcessingTimeMs: processingTime,
		}, nil
	}

	// Perform user feedback classification
	result, err := s.classifier.ClassifyFeedback(req.Text)
	if err != nil {
		return nil, fmt.Errorf("user feedback classification failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	return &UserFeedbackResponse{
		FeedbackType:     result.FeedbackType,
		Label:            result.FeedbackType, // FeedbackType is the label
		Confidence:       float64(result.Confidence),
		ProcessingTimeMs: processingTime,
	}, nil
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

// ClassifyIntentForEval performs intent classification specifically for evaluation scenarios
// This method forces evaluation of all signals and returns comprehensive signal information
func (s *ClassificationService) ClassifyIntentForEval(req IntentRequest) (*EvalResponse, error) {
	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		// Return placeholder response
		return &EvalResponse{
			OriginalText: req.Text,
			Metrics:      &classification.SignalMetricsCollection{},
		}, nil
	}

	// Force evaluate all signals
	signals := s.classifier.EvaluateAllSignalsWithForceOption(req.Text, true)

	// Evaluate decision with engine (if decisions are configured)
	var decisionResult *decision.DecisionResult
	var err error
	if s.config != nil && len(s.config.IntelligentRouting.Decisions) > 0 {
		decisionResult, err = s.classifier.EvaluateDecisionWithEngine(signals)
		if err != nil {
			// Log error but continue
			if !strings.Contains(err.Error(), "no decisions configured") {
				logging.Warnf("Decision evaluation failed: %v", err)
			}
		}
	}

	// Build eval response
	response := s.buildEvalResponse(req.Text, signals, decisionResult)

	return response, nil
}

// buildEvalResponse builds an EvalResponse from signal results and decision result
func (s *ClassificationService) buildEvalResponse(
	text string,
	signals *classification.SignalResults,
	decisionResult *decision.DecisionResult,
) *EvalResponse {
	response := &EvalResponse{
		OriginalText: text,
		Metrics:      signals.Metrics,
	}

	// Build matched signals
	matchedSignals := &MatchedSignals{
		Keywords:     signals.MatchedKeywordRules,
		Embeddings:   signals.MatchedEmbeddingRules,
		Domains:      signals.MatchedDomainRules,
		FactCheck:    signals.MatchedFactCheckRules,
		UserFeedback: signals.MatchedUserFeedbackRules,
		Preferences:  signals.MatchedPreferenceRules,
		Language:     signals.MatchedLanguageRules,
		Latency:      signals.MatchedLatencyRules,
		Context:      signals.MatchedContextRules,
		Complexity:   signals.MatchedComplexityRules,
	}

	// Build unmatched signals by comparing all configured signals with matched signals
	unmatchedSignals := s.getUnmatchedSignals(signals)

	// Build decision result
	if decisionResult != nil && decisionResult.Decision != nil {
		// Extract used signals from decision's rule configuration (not just matched rules)
		usedSignals := s.extractUsedSignalsFromDecision(decisionResult.Decision)

		response.DecisionResult = &EvalDecisionResult{
			DecisionName:     decisionResult.Decision.Name,
			UsedSignals:      usedSignals,
			MatchedSignals:   matchedSignals,
			UnmatchedSignals: unmatchedSignals,
		}

		// Set recommended models and routing decision
		if len(decisionResult.Decision.ModelRefs) > 0 {
			// Collect all models from modelRefs
			models := make([]string, 0, len(decisionResult.Decision.ModelRefs))
			for _, modelRef := range decisionResult.Decision.ModelRefs {
				models = append(models, modelRef.Model)
			}
			response.RecommendedModels = models
			response.RoutingDecision = decisionResult.Decision.Name
		}
	} else {
		// No decision matched
		response.DecisionResult = &EvalDecisionResult{
			DecisionName:     "",
			UsedSignals:      &MatchedSignals{}, // Empty used signals
			MatchedSignals:   matchedSignals,
			UnmatchedSignals: unmatchedSignals,
		}
	}

	return response
}

// extractUsedSignalsFromDecision extracts all signals used in a decision's rule configuration
// This includes ALL signals defined in the decision rules, not just the ones that matched
func (s *ClassificationService) extractUsedSignalsFromDecision(decision *config.Decision) *MatchedSignals {
	usedSignals := &MatchedSignals{}

	// Recursively extract signals from rule combination
	s.extractSignalsFromRuleCombination(decision.Rules, usedSignals)

	return usedSignals
}

// extractSignalsFromRuleCombination recursively extracts signals from a rule combination
func (s *ClassificationService) extractSignalsFromRuleCombination(rules config.RuleCombination, usedSignals *MatchedSignals) {
	for _, condition := range rules.Conditions {
		signalType := strings.ToLower(strings.TrimSpace(condition.Type))
		signalName := strings.TrimSpace(condition.Name)

		// Add to appropriate field based on type
		switch signalType {
		case "keyword":
			if !contains(usedSignals.Keywords, signalName) {
				usedSignals.Keywords = append(usedSignals.Keywords, signalName)
			}
		case "embedding":
			if !contains(usedSignals.Embeddings, signalName) {
				usedSignals.Embeddings = append(usedSignals.Embeddings, signalName)
			}
		case "domain":
			if !contains(usedSignals.Domains, signalName) {
				usedSignals.Domains = append(usedSignals.Domains, signalName)
			}
		case "fact_check":
			if !contains(usedSignals.FactCheck, signalName) {
				usedSignals.FactCheck = append(usedSignals.FactCheck, signalName)
			}
		case "user_feedback":
			if !contains(usedSignals.UserFeedback, signalName) {
				usedSignals.UserFeedback = append(usedSignals.UserFeedback, signalName)
			}
		case "preference":
			if !contains(usedSignals.Preferences, signalName) {
				usedSignals.Preferences = append(usedSignals.Preferences, signalName)
			}
		case "language":
			if !contains(usedSignals.Language, signalName) {
				usedSignals.Language = append(usedSignals.Language, signalName)
			}
		case "latency":
			if !contains(usedSignals.Latency, signalName) {
				usedSignals.Latency = append(usedSignals.Latency, signalName)
			}
		case "context":
			if !contains(usedSignals.Context, signalName) {
				usedSignals.Context = append(usedSignals.Context, signalName)
			}
		case "complexity":
			if !contains(usedSignals.Complexity, signalName) {
				usedSignals.Complexity = append(usedSignals.Complexity, signalName)
			}
		}
	}
}

// contains checks if a string slice contains a specific string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// getUnmatchedSignals returns all configured signals that were not matched
func (s *ClassificationService) getUnmatchedSignals(signals *classification.SignalResults) *MatchedSignals {
	unmatched := &MatchedSignals{}

	if s.classifier == nil || s.config == nil {
		return unmatched
	}

	// Helper function to check if a signal is matched
	isMatched := func(signalName string, matchedList []string) bool {
		for _, matched := range matchedList {
			if matched == signalName {
				return true
			}
		}
		return false
	}

	// Check keyword rules
	for _, rule := range s.classifier.Config.KeywordRules {
		if !isMatched(rule.Name, signals.MatchedKeywordRules) {
			unmatched.Keywords = append(unmatched.Keywords, rule.Name)
		}
	}

	// Check embedding rules
	for _, rule := range s.classifier.Config.EmbeddingRules {
		if !isMatched(rule.Name, signals.MatchedEmbeddingRules) {
			unmatched.Embeddings = append(unmatched.Embeddings, rule.Name)
		}
	}

	// Check domain rules (categories)
	for _, category := range s.classifier.Config.Categories {
		if !isMatched(category.Name, signals.MatchedDomainRules) {
			unmatched.Domains = append(unmatched.Domains, category.Name)
		}
	}

	// Check fact-check rules
	for _, rule := range s.classifier.Config.FactCheckRules {
		if !isMatched(rule.Name, signals.MatchedFactCheckRules) {
			unmatched.FactCheck = append(unmatched.FactCheck, rule.Name)
		}
	}

	// Check user feedback rules
	for _, rule := range s.classifier.Config.UserFeedbackRules {
		if !isMatched(rule.Name, signals.MatchedUserFeedbackRules) {
			unmatched.UserFeedback = append(unmatched.UserFeedback, rule.Name)
		}
	}

	// Check preference rules
	for _, rule := range s.classifier.Config.PreferenceRules {
		if !isMatched(rule.Name, signals.MatchedPreferenceRules) {
			unmatched.Preferences = append(unmatched.Preferences, rule.Name)
		}
	}

	// Check language rules
	for _, rule := range s.classifier.Config.LanguageRules {
		if !isMatched(rule.Name, signals.MatchedLanguageRules) {
			unmatched.Language = append(unmatched.Language, rule.Name)
		}
	}

	// Check latency rules
	for _, rule := range s.classifier.Config.LatencyRules {
		if !isMatched(rule.Name, signals.MatchedLatencyRules) {
			unmatched.Latency = append(unmatched.Latency, rule.Name)
		}
	}

	// Check context rules
	for _, rule := range s.classifier.Config.ContextRules {
		if !isMatched(rule.Name, signals.MatchedContextRules) {
			unmatched.Context = append(unmatched.Context, rule.Name)
		}
	}

	// Check complexity rules
	for _, rule := range s.classifier.Config.ComplexityRules {
		if !isMatched(rule.Name, signals.MatchedComplexityRules) {
			unmatched.Complexity = append(unmatched.Complexity, rule.Name)
		}
	}

	return unmatched
}
