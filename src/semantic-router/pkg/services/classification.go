package services

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
)

// Global classification service instance
var globalClassificationService *ClassificationService

// ClassificationService provides classification functionality
type ClassificationService struct {
	classifier *classification.Classifier
	config     *config.RouterConfig
}

// NewClassificationService creates a new classification service
func NewClassificationService(classifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	service := &ClassificationService{
		classifier: classifier,
		config:     config,
	}
	// Set as global service for API access
	globalClassificationService = service
	return service
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
func (s *ClassificationService) getRecommendedModel(category string, confidence float64) string {
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
