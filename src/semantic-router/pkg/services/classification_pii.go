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

// PIIRequest represents a request for PII detection
type PIIRequest struct {
	Recipe  string      `json:"recipe,omitempty"`
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
	Recipe   string      `json:"recipe,omitempty"`
	HasPII   bool        `json:"has_pii"`
	Entities []PIIEntity `json:"entities"`
	// ScanIncomplete reports that the classifier saw only part of the text,
	// because a remote token_spans.v1 backend declared a truncation. The
	// entities below are real, but has_pii: false then means "nothing found in
	// the part that was read", not "nothing to find". Absent when the scan was
	// complete, which every local backend always is.
	ScanIncomplete         bool   `json:"scan_incomplete,omitempty"`
	MaskedText             string `json:"masked_text,omitempty"`
	SecurityRecommendation string `json:"security_recommendation"`
	ProcessingTimeMs       int64  `json:"processing_time_ms"`
}

// PIIEntity represents a detected PII entity
type PIIEntity struct {
	Type       string  `json:"type"`
	Value      string  `json:"value"`
	Confidence float64 `json:"confidence"`
	// Pointers so that an absent field means the caller did not ask for
	// positions. With a plain int, omitempty also drops an offset of 0.
	StartPos    *int   `json:"start_position,omitempty"`
	EndPos      *int   `json:"end_position,omitempty"`
	MaskedValue string `json:"masked_value,omitempty"`
}

// DetectPII performs PII detection
func (s *ClassificationService) DetectPII(ctx context.Context, req PIIRequest) (*PIIResponse, error) {
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
		processingTime := time.Since(start).Milliseconds()
		return &PIIResponse{
			Recipe:                 recipe,
			HasPII:                 false,
			Entities:               []PIIEntity{},
			SecurityRecommendation: "allow",
			ProcessingTimeMs:       processingTime,
		}, nil
	}

	answer, err := classifier.AskPreset(ctx, config.SignalTypePII, presets.PII, req.Text)
	if errors.Is(err, modelservice.ErrUnavailable) {
		return nil, fmt.Errorf("%w: PII detection failed: %w", ErrClassifierUnavailable, err)
	}
	if err != nil {
		return nil, fmt.Errorf("PII detection failed: %w", err)
	}
	threshold := classifier.Config.PresetThreshold(config.SignalTypePII, presets.PII)
	if req.Options != nil && req.Options.ConfidenceThreshold > 0 {
		threshold = req.Options.ConfidenceThreshold
	}

	processingTime := time.Since(start).Milliseconds()
	response := s.buildPIIResponse(req.Text, piiDetections(req.Text, answer.Spans, threshold), req.Options)
	// The spans of an answer that read only part of the text are valid for
	// that part. It is reported, so a caller cannot read a partial scan as a
	// clean one.
	response.ScanIncomplete = answer.InputCoverage != "complete"
	response.ProcessingTimeMs = processingTime
	response.Recipe = recipe
	return response, nil
}
