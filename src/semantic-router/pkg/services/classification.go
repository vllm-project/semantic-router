package services

import (
	"fmt"
	"os"
	"sort"
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
func createLegacyClassifier(cfg *config.RouterConfig) (*classification.Classifier, error) {
	// Load category mapping
	var categoryMapping *classification.CategoryMapping

	// Check if we should load categories from MCP server
	// Note: tool_name is optional and will be auto-discovered if not specified
	useMCPCategories := cfg.CategoryModel.ModelID == "" &&
		cfg.MCPCategoryModel.Enabled

	if useMCPCategories && cfg.UsesSignalTypeInRouting(config.SignalTypeDomain) {
		// Categories will be loaded from MCP server during initialization
		logging.Infof("Category mapping will be loaded from MCP server")
		// Create empty mapping initially - will be populated during initialization
		categoryMapping = nil
	} else if cfg.NeedsCategoryMappingForRouting() {
		// Load from file only when routing actually needs domain classification
		var err error
		categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}

	// Load PII mapping only when routing actually needs PII classification
	var piiMapping *classification.PIIMapping
	if cfg.NeedsPIIMappingForRouting() {
		var err error
		piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}

	// Load jailbreak mapping only when routing actually needs jailbreak classification
	var jailbreakMapping *classification.JailbreakMapping
	if cfg.NeedsJailbreakMappingForRouting() {
		var err error
		jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}

	// Create classifier
	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	return classifier, nil
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
		nil,
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
	RevealEntityText    bool     `json:"reveal_entity_text,omitempty"`
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
	// Use custom confidence threshold if provided
	var detections []classification.PIIDetection
	var err error
	if req.Options != nil && req.Options.ConfidenceThreshold > 0 {
		detections, err = s.classifier.ClassifyPIIWithDetailsAndThreshold(req.Text, float32(req.Options.ConfidenceThreshold))
	} else {
		detections, err = s.classifier.ClassifyPIIWithDetails(req.Text)
	}
	if err != nil {
		return nil, fmt.Errorf("PII detection failed: %w", err)
	}
	processingTime := time.Since(start).Milliseconds()
	response := s.buildPIIResponse(req.Text, detections, req.Options)
	response.ProcessingTimeMs = processingTime
	return response, nil
}

// buildPIIResponse processes raw PII detections into a PIIResponse, applying all options.
func (s *ClassificationService) buildPIIResponse(text string, detections []classification.PIIDetection, options *PIIOptions) *PIIResponse {
	detections = filterPIIDetectionsByType(detections, options)

	returnPositions := options != nil && options.ReturnPositions
	maskEntities := options != nil && options.MaskEntities
	revealEntityText := options != nil && options.RevealEntityText

	var placeholders map[string]string
	if maskEntities {
		placeholders = buildPIIMaskPlaceholders(detections)
	}

	response := &PIIResponse{
		HasPII:   len(detections) > 0,
		Entities: buildPIIEntities(detections, returnPositions, maskEntities, revealEntityText, placeholders),
	}

	if maskEntities && len(detections) > 0 {
		response.MaskedText = buildMaskedPIIText(text, detections, placeholders)
	}
	response.SecurityRecommendation = piiSecurityRecommendation(response.HasPII)

	return response
}

func filterPIIDetectionsByType(detections []classification.PIIDetection, options *PIIOptions) []classification.PIIDetection {
	if options == nil || len(options.EntityTypes) == 0 {
		return detections
	}
	filtered := detections[:0]
	for _, detection := range detections {
		for _, entityType := range options.EntityTypes {
			if strings.EqualFold(detection.EntityType, entityType) {
				filtered = append(filtered, detection)
				break
			}
		}
	}
	return filtered
}

func buildPIIMaskPlaceholders(detections []classification.PIIDetection) map[string]string {
	typeCounters := make(map[string]map[string]int)
	placeholders := make(map[string]string)
	for _, detection := range detections {
		key := detection.EntityType + "\x00" + detection.Text
		if _, exists := placeholders[key]; exists {
			continue
		}
		texts, ok := typeCounters[detection.EntityType]
		if !ok {
			texts = make(map[string]int)
			typeCounters[detection.EntityType] = texts
		}
		idx := len(texts)
		texts[detection.Text] = idx
		placeholders[key] = fmt.Sprintf("[%s_%d]", detection.EntityType, idx)
	}
	return placeholders
}

func buildPIIEntities(
	detections []classification.PIIDetection,
	returnPositions bool,
	maskEntities bool,
	revealEntityText bool,
	placeholders map[string]string,
) []PIIEntity {
	entities := make([]PIIEntity, 0, len(detections))
	for _, detection := range detections {
		entity := PIIEntity{
			Type:       detection.EntityType,
			Value:      buildPIIEntityValue(detection.Text, revealEntityText),
			Confidence: float64(detection.Confidence),
		}
		if returnPositions {
			entity.StartPos = detection.Start
			entity.EndPos = detection.End
		}
		if maskEntities {
			entity.MaskedValue = placeholders[detection.EntityType+"\x00"+detection.Text]
		}
		entities = append(entities, entity)
	}
	return entities
}

func buildPIIEntityValue(text string, revealEntityText bool) string {
	if revealEntityText {
		return text
	}
	return "[DETECTED]"
}

func buildMaskedPIIText(text string, detections []classification.PIIDetection, placeholders map[string]string) string {
	sorted := make([]classification.PIIDetection, len(detections))
	copy(sorted, detections)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Start > sorted[j].Start
	})
	maskedText := text
	for _, detection := range sorted {
		placeholder := placeholders[detection.EntityType+"\x00"+detection.Text]
		if detection.Start >= 0 && detection.End <= len(maskedText) && detection.Start < detection.End {
			maskedText = maskedText[:detection.Start] + placeholder + maskedText[detection.End:]
		}
	}
	return maskedText
}

func piiSecurityRecommendation(hasPII bool) string {
	if hasPII {
		return "block"
	}
	return "allow"
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
	if s.classifier != nil {
		model := s.classifier.SelectBestModelForCategory(category)
		if model != "" {
			return model
		}
	}
	if s.config == nil {
		return ""
	}
	if model := recommendedModelFromDecisions(s.config.Decisions, category); model != "" {
		return model
	}
	return s.config.DefaultModel
}

func recommendedModelFromDecisions(decisions []config.Decision, category string) string {
	for _, decision := range decisions {
		if !strings.EqualFold(decision.Name, category) {
			continue
		}
		if len(decision.ModelRefs) == 0 {
			return ""
		}
		modelRef := decision.ModelRefs[0]
		if modelRef.LoRAName != "" {
			return modelRef.LoRAName
		}
		return modelRef.Model
	}
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

// NLIRequest represents a request for Natural Language Inference classification.
type NLIRequest struct {
	Premise    string `json:"premise"`    // Text to evaluate (the source text or claim)
	Hypothesis string `json:"hypothesis"` // Policy or hypothesis to check against the premise
}

// NLIResponse represents the result of NLI classification.
type NLIResponse struct {
	Label             string  `json:"label"`              // "entailment", "neutral", or "contradiction"
	Confidence        float32 `json:"confidence"`         // Confidence of the predicted label (0.0-1.0)
	EntailmentProb    float32 `json:"entailment_prob"`    // Probability that premise entails hypothesis
	NeutralProb       float32 `json:"neutral_prob"`       // Probability that relationship is neutral
	ContradictionProb float32 `json:"contradiction_prob"` // Probability that premise contradicts hypothesis
	ProcessingTimeMs  int64   `json:"processing_time_ms"`
}

// ClassifyNLI performs Natural Language Inference between a premise and hypothesis.
// Returns ENTAILMENT when the premise supports the hypothesis, NEUTRAL when it
// neither supports nor contradicts, and CONTRADICTION when it conflicts.
func (s *ClassificationService) ClassifyNLI(req NLIRequest) (*NLIResponse, error) {
	start := time.Now()

	if req.Premise == "" || req.Hypothesis == "" {
		return nil, fmt.Errorf("both premise and hypothesis must be provided")
	}

	if s.classifier == nil {
		return nil, fmt.Errorf("classification service not available")
	}

	det := s.classifier.GetHallucinationDetector()
	if det == nil || !det.IsNLIInitialized() {
		return nil, fmt.Errorf("NLI model not initialized — configure hallucination_mitigation.nli_model in your router config")
	}

	result, err := det.ClassifyNLI(req.Premise, req.Hypothesis)
	if err != nil {
		return nil, fmt.Errorf("NLI classification failed: %w", err)
	}

	return &NLIResponse{
		Label:             result.LabelStr,
		Confidence:        result.Confidence,
		EntailmentProb:    result.EntailmentProb,
		NeutralProb:       result.NeutralProb,
		ContradictionProb: result.ContradictProb,
		ProcessingTimeMs:  time.Since(start).Milliseconds(),
	}, nil
}

// IsNLIReady reports whether the NLI model is loaded and ready for inference.
func (s *ClassificationService) IsNLIReady() bool {
	if s.classifier == nil {
		return false
	}
	det := s.classifier.GetHallucinationDetector()
	return det != nil && det.IsNLIInitialized()
}
