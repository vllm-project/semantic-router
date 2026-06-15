package services

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

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
