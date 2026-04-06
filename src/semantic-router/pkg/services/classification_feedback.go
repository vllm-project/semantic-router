package services

import (
	"fmt"
	"time"
)

// FactCheckRequest represents a request for fact-check classification.
type FactCheckRequest struct {
	Text    string            `json:"text"`
	Options *FactCheckOptions `json:"options,omitempty"`
}

// FactCheckOptions contains options for fact-check classification.
type FactCheckOptions struct {
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// FactCheckResponse represents the response from fact-check classification.
type FactCheckResponse struct {
	NeedsFactCheck   bool    `json:"needs_fact_check"`
	Label            string  `json:"label"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// ClassifyFactCheck performs fact-check classification.
func (s *ClassificationService) ClassifyFactCheck(req FactCheckRequest) (*FactCheckResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	if s.classifier == nil {
		return &FactCheckResponse{
			NeedsFactCheck:   false,
			Label:            "unknown",
			Confidence:       0.0,
			ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}

	if !s.classifier.IsFactCheckEnabled() {
		return &FactCheckResponse{
			NeedsFactCheck:   false,
			Label:            "fact_check_disabled",
			Confidence:       0.0,
			ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}

	result, err := s.classifier.ClassifyFactCheck(req.Text)
	if err != nil {
		return nil, fmt.Errorf("fact-check classification failed: %w", err)
	}

	return &FactCheckResponse{
		NeedsFactCheck:   result.NeedsFactCheck,
		Label:            result.Label,
		Confidence:       float64(result.Confidence),
		ProcessingTimeMs: time.Since(start).Milliseconds(),
	}, nil
}

// UserFeedbackRequest represents a request for user feedback classification.
type UserFeedbackRequest struct {
	Text    string               `json:"text"`
	Options *UserFeedbackOptions `json:"options,omitempty"`
}

// UserFeedbackOptions contains options for user feedback classification.
type UserFeedbackOptions struct {
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
}

// UserFeedbackResponse represents the response from user feedback classification.
type UserFeedbackResponse struct {
	FeedbackType     string  `json:"feedback_type"`
	Label            string  `json:"label"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// ClassifyUserFeedback performs user feedback classification.
func (s *ClassificationService) ClassifyUserFeedback(req UserFeedbackRequest) (*UserFeedbackResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	if s.classifier == nil {
		return &UserFeedbackResponse{
			FeedbackType:     "unknown",
			Label:            "unknown",
			Confidence:       0.0,
			ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}

	if !s.classifier.IsFeedbackDetectorEnabled() {
		return &UserFeedbackResponse{
			FeedbackType:     "feedback_detector_disabled",
			Label:            "feedback_detector_disabled",
			Confidence:       0.0,
			ProcessingTimeMs: time.Since(start).Milliseconds(),
		}, nil
	}

	result, err := s.classifier.ClassifyFeedback(req.Text)
	if err != nil {
		return nil, fmt.Errorf("user feedback classification failed: %w", err)
	}

	return &UserFeedbackResponse{
		FeedbackType:     result.FeedbackType,
		Label:            result.FeedbackType,
		Confidence:       float64(result.Confidence),
		ProcessingTimeMs: time.Since(start).Milliseconds(),
	}, nil
}
