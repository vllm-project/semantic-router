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

// FactCheckRequest represents a request for fact-check classification
type FactCheckRequest struct {
	Recipe  string            `json:"recipe,omitempty"`
	Text    string            `json:"text"`
	Options *FactCheckOptions `json:"options,omitempty"`
}

// FactCheckOptions contains options for fact-check classification
type FactCheckOptions struct {
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// FactCheckResponse represents the response from fact-check classification
type FactCheckResponse struct {
	Recipe              string  `json:"recipe,omitempty"`
	NeedsFactCheck      bool    `json:"needs_fact_check"`
	Label               string  `json:"label"`
	Confidence          float64 `json:"confidence"`
	ConfidenceAvailable bool    `json:"confidence_available"`
	ProcessingTimeMs    int64   `json:"processing_time_ms"`
}

// ClassifyFactCheck asks the fact_check preset whether answering the text
// needs fact checking: it does when the probability of FACT_CHECK_NEEDED
// reaches the option's threshold, else the one calibrated for the model.
func (s *ClassificationService) ClassifyFactCheck(ctx context.Context, req FactCheckRequest) (*FactCheckResponse, error) {
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
	if classifier == nil {
		return &FactCheckResponse{
			Recipe: recipe, Label: "unknown", ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}
	answer, err := classifier.AskPreset(ctx, config.SignalTypeFactCheck, presets.FactCheck, req.Text)
	if errors.Is(err, modelservice.ErrUnavailable) {
		if req.Recipe == "" {
			return &FactCheckResponse{
				Recipe: recipe, Label: "unknown", ProcessingTimeMs: time.Since(start).Milliseconds(),
			}, nil
		}
		return nil, fmt.Errorf("%w: fact-check classification failed: %w", ErrClassifierUnavailable, err)
	}
	if err != nil {
		return nil, fmt.Errorf("fact-check classification failed: %w", err)
	}
	threshold := classifier.Config.PresetThreshold(config.SignalTypeFactCheck, presets.FactCheck)
	if req.Options != nil && req.Options.ConfidenceThreshold > 0 {
		threshold = req.Options.ConfidenceThreshold
	}
	return &FactCheckResponse{
		Recipe:              recipe,
		NeedsFactCheck:      answer.Probabilities[factCheckNeeded] >= threshold,
		Label:               answer.Choice,
		Confidence:          answer.Probabilities[answer.Choice],
		ConfidenceAvailable: true,
		ProcessingTimeMs:    time.Since(start).Milliseconds(),
	}, nil
}

// factCheckNeeded is the fact_check preset's positive option.
const factCheckNeeded = "FACT_CHECK_NEEDED"

// UserFeedbackRequest represents a request for user feedback classification
type UserFeedbackRequest struct {
	Recipe  string               `json:"recipe,omitempty"`
	Text    string               `json:"text"`
	Options *UserFeedbackOptions `json:"options,omitempty"`
}

// UserFeedbackOptions contains options for user feedback classification
type UserFeedbackOptions struct {
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// UserFeedbackResponse represents the response from user feedback classification
type UserFeedbackResponse struct {
	Recipe              string  `json:"recipe,omitempty"`
	Abstained           bool    `json:"abstained,omitempty"`
	FeedbackType        string  `json:"feedback_type"`
	Label               string  `json:"label"`
	Confidence          float64 `json:"confidence"`
	ConfidenceAvailable bool    `json:"confidence_available"`
	ProcessingTimeMs    int64   `json:"processing_time_ms"`
}

// ClassifyUserFeedback asks the feedback preset what feedback the text
// gives; it abstains when the chosen option's probability is below the
// option's threshold, else the one calibrated for the model.
func (s *ClassificationService) ClassifyUserFeedback(ctx context.Context, req UserFeedbackRequest) (*UserFeedbackResponse, error) {
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
	if classifier == nil {
		return &UserFeedbackResponse{
			Recipe: recipe, FeedbackType: "unknown", Label: "unknown", ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}
	answer, err := classifier.AskPreset(ctx, config.SignalTypeUserFeedback, presets.Feedback, req.Text)
	if errors.Is(err, modelservice.ErrUnavailable) {
		if req.Recipe == "" {
			return &UserFeedbackResponse{
				Recipe: recipe, FeedbackType: "unknown", Label: "unknown", ProcessingTimeMs: time.Since(start).Milliseconds(),
			}, nil
		}
		return nil, fmt.Errorf("%w: user feedback classification failed: %w", ErrClassifierUnavailable, err)
	}
	if err != nil {
		return nil, fmt.Errorf("user feedback classification failed: %w", err)
	}
	threshold := classifier.Config.PresetThreshold(config.SignalTypeUserFeedback, presets.Feedback)
	if req.Options != nil && req.Options.ConfidenceThreshold > 0 {
		threshold = req.Options.ConfidenceThreshold
	}
	confidence := answer.Probabilities[answer.Choice]
	return &UserFeedbackResponse{
		Recipe:              recipe,
		Abstained:           confidence < threshold,
		FeedbackType:        answer.Choice,
		Label:               answer.Choice,
		Confidence:          confidence,
		ConfidenceAvailable: true,
		ProcessingTimeMs:    time.Since(start).Milliseconds(),
	}, nil
}
