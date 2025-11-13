package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
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
	observability.Debugf("Debug: Current working directory: %s", wd)
	observability.Debugf("Debug: Attempting to discover models in: ./models")

	// Always try to auto-discover and initialize unified classifier for batch processing
	// Use model path from config, fallback to "./models" if not specified
	modelsPath := "./models"
	if config != nil && config.Classifier.CategoryModel.ModelID != "" {
		// Extract the models directory from the model path
		// e.g., "models/category_classifier_modernbert-base_model" -> "models"
		if idx := strings.Index(config.Classifier.CategoryModel.ModelID, "/"); idx > 0 {
			modelsPath = config.Classifier.CategoryModel.ModelID[:idx]
		}
	}
	unifiedClassifier, ucErr := classification.AutoInitializeUnifiedClassifier(modelsPath)
	if ucErr != nil {
		observability.Infof("Unified classifier auto-discovery failed: %v", ucErr)
	}
	// create legacy classifier
	legacyClassifier, lcErr := createLegacyClassifier(config)
	if lcErr != nil {
		observability.Warnf("Legacy classifier initialization failed: %v", lcErr)
	}
	if unifiedClassifier == nil && legacyClassifier == nil {
		observability.Warnf("No classifier initialized. Using placeholder service.")
	}
	return NewUnifiedClassificationService(unifiedClassifier, legacyClassifier, config), nil
}

// createLegacyClassifier creates a legacy classifier with proper model loading
func createLegacyClassifier(config *config.RouterConfig) (*classification.Classifier, error) {
	// Load category mapping
	var categoryMapping *classification.CategoryMapping

	// Check if we should load categories from MCP server
	// Note: tool_name is optional and will be auto-discovered if not specified
	useMCPCategories := config.Classifier.CategoryModel.ModelID == "" &&
		config.Classifier.MCPCategoryModel.Enabled

	if useMCPCategories {
		// Categories will be loaded from MCP server during initialization
		observability.Infof("Category mapping will be loaded from MCP server")
		// Create empty mapping initially - will be populated during initialization
		categoryMapping = nil
	} else if config.Classifier.CategoryModel.CategoryMappingPath != "" {
		// Load from file as usual
		var err error
		categoryMapping, err = classification.LoadCategoryMapping(config.Classifier.CategoryModel.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}

	// Load PII mapping
	var piiMapping *classification.PIIMapping
	if config.Classifier.PIIModel.PIIMappingPath != "" {
		var err error
		piiMapping, err = classification.LoadPIIMapping(config.Classifier.PIIModel.PIIMappingPath)
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
}

// IntentResponse represents the response from intent classification
type IntentResponse struct {
	Classification   Classification     `json:"classification"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RecommendedModel string             `json:"recommended_model,omitempty"`
	RoutingDecision  string             `json:"routing_decision,omitempty"`
}

// Classification represents basic classification result
type Classification struct {
	Category         string  `json:"category"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// MultimodalClassificationRequest represents a request for multimodal classification
type MultimodalClassificationRequest struct {
	Text        string                 `json:"text"`
	Images      []MultimodalImage      `json:"images,omitempty"`
	ContentType string                 `json:"content_type"` // "text", "image", "multimodal"
	Options     *ClassificationOptions `json:"options,omitempty"`
}

// MultimodalImage represents an image in a multimodal classification request
type MultimodalImage struct {
	Data        string `json:"data"`                  // Base64 encoded image
	URL         string `json:"url,omitempty"`         // Alternative: image URL
	MimeType    string `json:"mime_type"`             // image/jpeg, image/png, etc.
	Description string `json:"description,omitempty"` // Optional text description
}

// ClassificationOptions contains options for multimodal classification
type ClassificationOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
}

// ClassifyIntent performs intent classification
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

	// Perform classification using the existing classifier
	category, confidence, err := s.classifier.ClassifyCategory(req.Text)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &IntentResponse{
		Classification: Classification{
			Category:         category,
			Confidence:       confidence,
			ProcessingTimeMs: processingTime,
		},
	}

	// Add probabilities if requested
	if req.Options != nil && req.Options.ReturnProbabilities {
		// TODO: Implement probability extraction from classifier
		response.Probabilities = map[string]float64{
			category: confidence,
		}
	}

	// Add recommended model based on category
	if model := s.getRecommendedModel(category, confidence); model != "" {
		response.RecommendedModel = model
	}

	// Determine routing decision
	response.RoutingDecision = s.getRoutingDecision(confidence, req.Options)

	return response, nil
}

// ClassifyMultimodal performs multimodal classification (text + images)
// Basic prototype version - uses Ollama
func (s *ClassificationService) ClassifyMultimodal(req MultimodalClassificationRequest) (*IntentResponse, error) {
	// Validate input
	if req.Text == "" && len(req.Images) == 0 {
		return nil, fmt.Errorf("either text or images must be provided")
	}

	text := req.Text

	var imageDescriptions []string
	if len(req.Images) > 0 {
		observability.Infof("Processing %d image(s) with Ollama", len(req.Images))
		descriptions, err := s.extractImageFeatures(req.Images)
		if err != nil {
			observability.Errorf("Failed: ", err)
		} else if len(descriptions) == 0 {
			observability.Warnf("no error occurred")
		} else {
			imageDescriptions = descriptions
			observability.Infof("Successfully extracted %d", len(descriptions))
		}
	}

	fusedText := s.fuseTextAndImageFeatures(text, imageDescriptions)
	observability.Debugf("Fused text length: %d chars", len(fusedText))

	intentReq := IntentRequest{
		Text:    fusedText,
		Options: convertClassificationOptions(req.Options),
	}

	intentResp, err := s.ClassifyIntent(intentReq)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	observability.Infof("Classification result: category=%s, confidence=%.3f",
		intentResp.Classification.Category, intentResp.Classification.Confidence)

	return intentResp, nil
}

// extractImageFeatures sends images to Ollama for feature extraction
func (s *ClassificationService) extractImageFeatures(images []MultimodalImage) ([]string, error) {
	var descriptions []string
	ollamaURL := "http://127.0.0.1:11434/api/generate" // Hardcoded for prototype
	ollamaModel := "llava:7b"

	for i, img := range images {
		if img.Data == "" {
			observability.Warnf("Skipping image %d with empty data", i+1)
			continue
		}

		cleanData := strings.ReplaceAll(strings.TrimSpace(img.Data), "\n", "")
		observability.Debugf("Sending image %d to Ollama (data length: %d chars, mime_type: %s)", i+1, len(cleanData), img.MimeType)

		body := map[string]interface{}{
			"model":  ollamaModel,
			"prompt": "Describe the image briefly for context.",
			"images": []string{cleanData},
		}

		payload, err := json.Marshal(body)
		if err != nil {
			observability.Errorf("Failed to marshal Ollama request: %v", err)
			return nil, fmt.Errorf("failed to prepare image request: %w", err)
		}

		observability.Debugf("Calling Ollama at %s with model %s", ollamaURL, ollamaModel)
		resp, err := http.Post(ollamaURL, "application/json", bytes.NewReader(payload))
		if err != nil {
			observability.Errorf("Ollama request failed: %v", err)
			return nil, fmt.Errorf("failed to connect to Ollama at %s: %w", ollamaURL, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			observability.Errorf("Ollama returned status %d: %s", resp.StatusCode, string(bodyBytes))
			return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(bodyBytes))
		}

		var fullResponse string
		decoder := json.NewDecoder(resp.Body)
		for decoder.More() {
			var chunk map[string]interface{}
			if err := decoder.Decode(&chunk); err != nil {
				observability.Warnf("Error decoding Ollama chunk: %v", err)
				break
			}
			if part, ok := chunk["response"].(string); ok {
				fullResponse += part
			}
		}

		if fullResponse != "" {
			descriptions = append(descriptions, fullResponse)
			observability.Infof("Image processed successfully, description length: %d chars", len(fullResponse))
		} else {
			observability.Warnf("Ollama returned empty response for image")
		}
	}

	return descriptions, nil
}

func (s *ClassificationService) fuseTextAndImageFeatures(text string, imageDescriptions []string) string {
	if len(imageDescriptions) == 0 {
		return text
	}

	var fused strings.Builder
	if text != "" {
		fused.WriteString(text)
	}

	for i, desc := range imageDescriptions {
		if desc != "" {
			if fused.Len() > 0 {
				fused.WriteString("\n")
			}
			if len(imageDescriptions) > 1 {
				fused.WriteString(fmt.Sprintf("Visual context %d: %s", i+1, desc))
			} else {
				fused.WriteString("Visual context: " + desc)
			}
		}
	}

	return fused.String()
}

func convertClassificationOptions(opts *ClassificationOptions) *IntentOptions {
	if opts == nil {
		return nil
	}
	return &IntentOptions{
		ReturnProbabilities: opts.ReturnProbabilities,
		ConfidenceThreshold: opts.ConfidenceThreshold,
		IncludeExplanation:  opts.IncludeExplanation,
	}
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

	// Perform PII detection using the existing classifier
	piiTypes, err := s.classifier.ClassifyPII(req.Text)
	if err != nil {
		return nil, fmt.Errorf("PII detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &PIIResponse{
		HasPII:           len(piiTypes) > 0,
		Entities:         []PIIEntity{},
		ProcessingTimeMs: processingTime,
	}

	// Convert PII types to entities (simplified for now)
	for _, piiType := range piiTypes {
		entity := PIIEntity{
			Type:       piiType,
			Value:      "[DETECTED]", // Placeholder - would need actual entity extraction
			Confidence: 0.9,          // Placeholder - would need actual confidence
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
	// TODO: Implement model recommendation logic based on category
	return fmt.Sprintf("%s-specialized-model", category)
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

// ClassifyIntent with unified classifier support (backward compatibility)
func (s *ClassificationService) ClassifyIntentUnified(req IntentRequest) (*IntentResponse, error) {
	if s.unifiedClassifier != nil {
		// Use unified classifier for better performance
		results, err := s.ClassifyBatchUnified([]string{req.Text})
		if err != nil {
			return nil, err
		}

		if len(results.IntentResults) == 0 {
			return nil, fmt.Errorf("no classification results")
		}

		// Convert unified result to legacy format
		intentResult := results.IntentResults[0]

		// Build probabilities map if available
		var probabilities map[string]float64
		if len(intentResult.Probabilities) > 0 && req.Options != nil && req.Options.ReturnProbabilities {
			probabilities = make(map[string]float64)
			// For now, just include the main category probability
			probabilities[intentResult.Category] = float64(intentResult.Confidence)
		}

		return &IntentResponse{
			Classification: Classification{
				Category:         intentResult.Category,
				Confidence:       float64(intentResult.Confidence),
				ProcessingTimeMs: results.ProcessingTimeMs,
			},
			Probabilities:    probabilities,
			RecommendedModel: s.getRecommendedModel(intentResult.Category, float64(intentResult.Confidence)),
			RoutingDecision:  s.getRoutingDecision(float64(intentResult.Confidence), req.Options),
		}, nil
	}

	// Fallback to legacy classifier
	return s.ClassifyIntent(req)
}

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
	config.ReplaceGlobalConfig(newConfig)
}
