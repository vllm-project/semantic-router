package services

import (
	"fmt"
	"time"
)

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

	if blankText(req.Text) {
		return nil, ErrEmptyText
	}

	if s.classifier == nil {
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

	if !s.classifier.IsJailbreakModelReady() {
		return nil, ErrModelNotReady
	}

	isJailbreak, jailbreakType, confidence, err := s.classifier.CheckForJailbreak(req.Text)
	if err != nil {
		return nil, fmt.Errorf("security detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()
	includeReasoning := req.Options != nil && req.Options.IncludeReasoning

	return buildSecurityResponse(isJailbreak, jailbreakType, confidence, riskScore, includeReasoning, processingTime), nil
}

// buildSecurityResponse assembles the security detection response. The risk_score
// reflects the probability of the jailbreak class (P(jailbreak)), which is distinct
// from the classifier confidence: a confident benign prediction yields a low
// risk_score rather than a misleadingly high one (issue #2591).
func buildSecurityResponse(isJailbreak bool, jailbreakType string, confidence, riskScore float32, includeReasoning bool, processingTimeMs int64) *SecurityResponse {
	response := &SecurityResponse{
		IsJailbreak:      isJailbreak,
		RiskScore:        float64(riskScore),
		Confidence:       float64(confidence),
		ProcessingTimeMs: processingTimeMs,
		DetectionTypes:   []string{},
		PatternsDetected: []string{},
	}

	if isJailbreak {
		response.DetectionTypes = append(response.DetectionTypes, jailbreakType)
		response.PatternsDetected = append(response.PatternsDetected, jailbreakType)
		response.Recommendation = "block"
		if includeReasoning {
			response.Reasoning = fmt.Sprintf("Detected %s pattern with confidence %.3f", jailbreakType, confidence)
		}
	} else {
		response.Recommendation = "allow"
	}

	return response
}
