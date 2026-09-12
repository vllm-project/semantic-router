package services

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
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
	IsJailbreak      bool                 `json:"is_jailbreak"`
	RiskScore        *float64             `json:"risk_score"`
	DetectionTypes   []string             `json:"detection_types"`
	Confidence       *float64             `json:"confidence"`
	ScoresAvailable  bool                 `json:"scores_available"`
	Decision         *tasks.LabelDecision `json:"decision,omitempty"`
	Recommendation   string               `json:"recommendation"`
	Reasoning        string               `json:"reasoning,omitempty"`
	PatternsDetected []string             `json:"patterns_detected"`
	ProcessingTimeMs int64                `json:"processing_time_ms"`
}

// CheckSecurity performs security detection. ctx is forwarded to the
// configured jailbreak backend so a remote (http_chat/http_classify) call can
// be cancelled if the caller (e.g. the HTTP request) is cancelled first.
func (s *ClassificationService) CheckSecurity(ctx context.Context, req SecurityRequest) (*SecurityResponse, error) {
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	start := time.Now()

	if blankText(req.Text) {
		return nil, ErrEmptyText
	}

	classifier := s.classifierSnapshot()
	if classifier == nil {
		return nil, fmt.Errorf("security detector is unavailable")
	}

	verdict, err := classifier.CheckForJailbreakVerdict(ctx, req.Text, classifier.Config.PromptGuard.Threshold)
	if err != nil {
		return nil, fmt.Errorf("security detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()
	includeReasoning := req.Options != nil && req.Options.IncludeReasoning

	return buildSecurityVerdictResponse(verdict, includeReasoning, processingTime), nil
}

// buildSecurityResponse assembles the security detection response. The risk_score
// reflects the probability of the jailbreak class (P(jailbreak)), which is distinct
// from the classifier confidence: a confident benign prediction yields a low
// risk_score rather than a misleadingly high one (issue #2591).
func buildSecurityResponse(isJailbreak bool, jailbreakType string, confidence, riskScore float32, includeReasoning bool, processingTimeMs int64) *SecurityResponse {
	return buildSecurityVerdictResponse(classification.JailbreakVerdict{Detected: isJailbreak, Label: jailbreakType, Confidence: &confidence, RiskScore: &riskScore}, includeReasoning, processingTimeMs)
}

func buildSecurityVerdictResponse(verdict classification.JailbreakVerdict, includeReasoning bool, processingTimeMs int64) *SecurityResponse {
	response := &SecurityResponse{IsJailbreak: verdict.Detected, Decision: verdict.Decision, ProcessingTimeMs: processingTimeMs, DetectionTypes: []string{}, PatternsDetected: []string{}}
	if verdict.Confidence != nil {
		value := float64(*verdict.Confidence)
		response.Confidence = &value
	}
	if verdict.RiskScore != nil {
		value := float64(*verdict.RiskScore)
		response.RiskScore = &value
	}
	response.ScoresAvailable = response.Confidence != nil && response.RiskScore != nil
	if verdict.Detected {
		response.DetectionTypes = append(response.DetectionTypes, verdict.Label)
		response.PatternsDetected = append(response.PatternsDetected, verdict.Label)
		response.Recommendation = "block"
		if includeReasoning {
			if verdict.Confidence != nil {
				response.Reasoning = fmt.Sprintf("Detected %s pattern with confidence %.3f", verdict.Label, *verdict.Confidence)
			} else {
				response.Reasoning = fmt.Sprintf("Model selected the %s safety label; probabilities are unavailable", verdict.Label)
			}
		}
	} else {
		response.Recommendation = "allow"
	}
	return response
}
