package services

import (
	"fmt"
	"time"
)

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

	if blankText(req.Text) {
		return nil, ErrEmptyText
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

	if blankText(req.Text) {
		return nil, ErrEmptyText
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
