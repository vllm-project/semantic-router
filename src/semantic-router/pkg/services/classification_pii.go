package services

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

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

	if blankText(req.Text) {
		return nil, ErrEmptyText
	}

	if s.classifier == nil {
		processingTime := time.Since(start).Milliseconds()
		return &PIIResponse{
			HasPII:                 false,
			Entities:               []PIIEntity{},
			SecurityRecommendation: "allow",
			ProcessingTimeMs:       processingTime,
		}, nil
	}

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
