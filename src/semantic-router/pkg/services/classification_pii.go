package services

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

// PIIRequest represents a request for PII detection.
type PIIRequest struct {
	Text    string      `json:"text"`
	Options *PIIOptions `json:"options,omitempty"`
}

// PIIOptions contains options for PII detection.
type PIIOptions struct {
	EntityTypes         []string `json:"entity_types,omitempty"`
	ConfidenceThreshold float64  `json:"confidence_threshold,omitempty"`
	ReturnPositions     bool     `json:"return_positions,omitempty"`
	MaskEntities        bool     `json:"mask_entities,omitempty"`
	RevealEntityText    bool     `json:"reveal_entity_text,omitempty"`
}

// PIIResponse represents the response from PII detection.
type PIIResponse struct {
	HasPII                 bool        `json:"has_pii"`
	Entities               []PIIEntity `json:"entities"`
	MaskedText             string      `json:"masked_text,omitempty"`
	SecurityRecommendation string      `json:"security_recommendation"`
	ProcessingTimeMs       int64       `json:"processing_time_ms"`
}

// PIIEntity represents a detected PII entity.
type PIIEntity struct {
	Type        string  `json:"type"`
	Value       string  `json:"value"`
	Confidence  float64 `json:"confidence"`
	StartPos    int     `json:"start_position,omitempty"`
	EndPos      int     `json:"end_position,omitempty"`
	MaskedValue string  `json:"masked_value,omitempty"`
}

// DetectPII performs PII detection.
func (s *ClassificationService) DetectPII(req PIIRequest) (*PIIResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	if s.classifier == nil {
		return &PIIResponse{
			HasPII:                 false,
			Entities:               []PIIEntity{},
			SecurityRecommendation: "allow",
			ProcessingTimeMs:       time.Since(start).Milliseconds(),
		}, nil
	}

	var (
		detections []classification.PIIDetection
		err        error
	)
	if req.Options != nil && req.Options.ConfidenceThreshold > 0 {
		detections, err = s.classifier.ClassifyPIIWithDetailsAndThreshold(
			req.Text,
			float32(req.Options.ConfidenceThreshold),
		)
	} else {
		detections, err = s.classifier.ClassifyPIIWithDetails(req.Text)
	}
	if err != nil {
		return nil, fmt.Errorf("PII detection failed: %w", err)
	}

	response := s.buildPIIResponse(req.Text, detections, req.Options)
	response.ProcessingTimeMs = time.Since(start).Milliseconds()
	return response, nil
}

// buildPIIResponse processes raw PII detections into a PIIResponse, applying all options.
func (s *ClassificationService) buildPIIResponse(text string, detections []classification.PIIDetection, options *PIIOptions) *PIIResponse {
	detections = filterPIIDetectionsByType(detections, options)

	returnPositions := options != nil && options.ReturnPositions
	maskEntities := options != nil && options.MaskEntities
	revealEntityText := options != nil && options.RevealEntityText

	var placeholders map[string]string
	if maskEntities {
		placeholders = buildPIIMaskPlaceholders(detections)
	}

	response := &PIIResponse{
		HasPII:   len(detections) > 0,
		Entities: buildPIIEntities(detections, returnPositions, maskEntities, revealEntityText, placeholders),
	}

	if maskEntities && len(detections) > 0 {
		response.MaskedText = buildMaskedPIIText(text, detections, placeholders)
	}
	response.SecurityRecommendation = piiSecurityRecommendation(response.HasPII)

	return response
}

func filterPIIDetectionsByType(detections []classification.PIIDetection, options *PIIOptions) []classification.PIIDetection {
	if options == nil || len(options.EntityTypes) == 0 {
		return detections
	}
	filtered := detections[:0]
	for _, detection := range detections {
		for _, entityType := range options.EntityTypes {
			if strings.EqualFold(detection.EntityType, entityType) {
				filtered = append(filtered, detection)
				break
			}
		}
	}
	return filtered
}

func buildPIIMaskPlaceholders(detections []classification.PIIDetection) map[string]string {
	typeCounters := make(map[string]map[string]int)
	placeholders := make(map[string]string)
	for _, detection := range detections {
		key := detection.EntityType + "\x00" + detection.Text
		if _, exists := placeholders[key]; exists {
			continue
		}
		texts, ok := typeCounters[detection.EntityType]
		if !ok {
			texts = make(map[string]int)
			typeCounters[detection.EntityType] = texts
		}
		idx := len(texts)
		texts[detection.Text] = idx
		placeholders[key] = fmt.Sprintf("[%s_%d]", detection.EntityType, idx)
	}
	return placeholders
}

func buildPIIEntities(
	detections []classification.PIIDetection,
	returnPositions bool,
	maskEntities bool,
	revealEntityText bool,
	placeholders map[string]string,
) []PIIEntity {
	entities := make([]PIIEntity, 0, len(detections))
	for _, detection := range detections {
		entity := PIIEntity{
			Type:       detection.EntityType,
			Value:      buildPIIEntityValue(detection.Text, revealEntityText),
			Confidence: float64(detection.Confidence),
		}
		if returnPositions {
			entity.StartPos = detection.Start
			entity.EndPos = detection.End
		}
		if maskEntities {
			entity.MaskedValue = placeholders[detection.EntityType+"\x00"+detection.Text]
		}
		entities = append(entities, entity)
	}
	return entities
}

func buildPIIEntityValue(text string, revealEntityText bool) string {
	if revealEntityText {
		return text
	}
	return "[DETECTED]"
}

func buildMaskedPIIText(text string, detections []classification.PIIDetection, placeholders map[string]string) string {
	sorted := make([]classification.PIIDetection, len(detections))
	copy(sorted, detections)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Start > sorted[j].Start
	})
	maskedText := text
	for _, detection := range sorted {
		placeholder := placeholders[detection.EntityType+"\x00"+detection.Text]
		if detection.Start >= 0 && detection.End <= len(maskedText) && detection.Start < detection.End {
			maskedText = maskedText[:detection.Start] + placeholder + maskedText[detection.End:]
		}
	}
	return maskedText
}

func piiSecurityRecommendation(hasPII bool) string {
	if hasPII {
		return "block"
	}
	return "allow"
}
