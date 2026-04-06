package services

import (
	"fmt"
	"time"
)

// SecurityRequest represents a request for security detection.
type SecurityRequest struct {
	Text    string           `json:"text"`
	Options *SecurityOptions `json:"options,omitempty"`
}

// SecurityOptions contains options for security detection.
type SecurityOptions struct {
	DetectionTypes   []string `json:"detection_types,omitempty"`
	Sensitivity      string   `json:"sensitivity,omitempty"`
	IncludeReasoning bool     `json:"include_reasoning,omitempty"`
}

// SecurityResponse represents the response from security detection.
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

// CheckSecurity performs security detection.
func (s *ClassificationService) CheckSecurity(req SecurityRequest) (*SecurityResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	if s.classifier == nil {
		return &SecurityResponse{
			IsJailbreak:      false,
			RiskScore:        0.1,
			DetectionTypes:   []string{},
			Confidence:       0.9,
			Recommendation:   "allow",
			PatternsDetected: []string{},
			ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}

	isJailbreak, jailbreakType, confidence, err := s.classifier.CheckForJailbreak(req.Text)
	if err != nil {
		return nil, fmt.Errorf("security detection failed: %w", err)
	}

	response := &SecurityResponse{
		IsJailbreak:      isJailbreak,
		RiskScore:        float64(confidence),
		Confidence:       float64(confidence),
		ProcessingTimeMs: time.Since(start).Milliseconds(),
		DetectionTypes:   []string{},
		PatternsDetected: []string{},
	}

	if isJailbreak {
		response.DetectionTypes = append(response.DetectionTypes, jailbreakType)
		response.PatternsDetected = append(response.PatternsDetected, jailbreakType)
		response.Recommendation = "block"
		if req.Options != nil && req.Options.IncludeReasoning {
			response.Reasoning = fmt.Sprintf(
				"Detected %s pattern with confidence %.3f",
				jailbreakType,
				confidence,
			)
		}
	} else {
		response.Recommendation = "allow"
	}

	return response, nil
}
