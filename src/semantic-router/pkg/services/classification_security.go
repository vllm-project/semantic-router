package services

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// SecurityRequest represents a request for security detection
type SecurityRequest struct {
	Recipe  string           `json:"recipe,omitempty"`
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
	Recipe           string   `json:"recipe,omitempty"`
	IsJailbreak      bool     `json:"is_jailbreak"`
	RiskScore        *float64 `json:"risk_score"`
	DetectionTypes   []string `json:"detection_types"`
	Confidence       *float64 `json:"confidence"`
	ScoresAvailable  bool     `json:"scores_available"`
	Recommendation   string   `json:"recommendation"`
	Reasoning        string   `json:"reasoning,omitempty"`
	PatternsDetected []string `json:"patterns_detected"`
	ProcessingTimeMs int64    `json:"processing_time_ms"`
}

// CheckSecurity asks the jailbreak preset about the text, at its calibrated
// threshold. ctx is forwarded to the decision model, so the call is cancelled
// when the caller (e.g. the HTTP request) is.
func (s *ClassificationService) CheckSecurity(ctx context.Context, req SecurityRequest) (*SecurityResponse, error) {
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	start := time.Now()

	if blankText(req.Text) {
		return nil, ErrEmptyText
	}

	classifier, _, recipe, scopeErr := s.diagnosticClassifierSnapshot(req.Recipe)
	if scopeErr != nil {
		return nil, scopeErr
	}
	if req.Recipe != "" && classifier == nil {
		return nil, ErrClassifierUnavailable
	}
	if classifier == nil {
		return nil, fmt.Errorf("security detector is unavailable")
	}

	answer, err := classifier.AskPreset(ctx, config.SignalTypeJailbreak, presets.Jailbreak, req.Text)
	if errors.Is(err, modelservice.ErrUnavailable) {
		return nil, fmt.Errorf("%w: security detection failed: %w", ErrClassifierUnavailable, err)
	}
	if err != nil {
		return nil, fmt.Errorf("security detection failed: %w", err)
	}
	canonical, _ := presets.Lookup(presets.Jailbreak, presets.StageRequest)
	attack := canonical.Positive[0]
	risk := answer.Probabilities[attack]
	detected := risk >= classifier.Config.PresetThreshold(config.SignalTypeJailbreak, presets.Jailbreak)

	processingTime := time.Since(start).Milliseconds()
	includeReasoning := req.Options != nil && req.Options.IncludeReasoning
	response := buildSecurityResponse(detected, attack, float32(answer.Probabilities[answer.Choice]), float32(risk), includeReasoning, processingTime)
	response.Recipe = recipe
	return response, nil
}

// buildSecurityResponse assembles the security detection response. The risk_score
// reflects the probability of the jailbreak class (P(jailbreak)), which is distinct
// from the classifier confidence: a confident benign prediction yields a low
// risk_score rather than a misleadingly high one (issue #2591).
func buildSecurityResponse(isJailbreak bool, jailbreakType string, confidence, riskScore float32, includeReasoning bool, processingTimeMs int64) *SecurityResponse {
	confidenceValue, riskValue := float64(confidence), float64(riskScore)
	response := &SecurityResponse{
		IsJailbreak: isJailbreak, Confidence: &confidenceValue, RiskScore: &riskValue, ScoresAvailable: true,
		ProcessingTimeMs: processingTimeMs, DetectionTypes: []string{}, PatternsDetected: []string{}, Recommendation: "allow",
	}
	if isJailbreak {
		response.DetectionTypes = append(response.DetectionTypes, jailbreakType)
		response.PatternsDetected = append(response.PatternsDetected, jailbreakType)
		response.Recommendation = "block"
		if includeReasoning {
			response.Reasoning = fmt.Sprintf("Detected %s pattern with confidence %.3f", jailbreakType, confidence)
		}
	}
	return response
}
