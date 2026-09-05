package services

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

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
		Entities: buildPIIEntities(text, detections, returnPositions, maskEntities, revealEntityText, placeholders),
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

// byteOffsetToRuneOffset converts a byte offset into text to a code-point offset.
//
// The classifier reports byte offsets, which is what buildMaskedPIIText needs
// because it slices a Go string. Clients index by code point, so the API
// converts on the way out.
func byteOffsetToRuneOffset(text string, byteOffset int) int {
	if byteOffset <= 0 {
		return 0
	}
	if byteOffset >= len(text) {
		return utf8.RuneCountInString(text)
	}
	return utf8.RuneCountInString(text[:byteOffset])
}

func buildPIIEntities(
	text string,
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
			startPos := byteOffsetToRuneOffset(text, detection.Start)
			endPos := byteOffsetToRuneOffset(text, detection.End)
			entity.StartPos = &startPos
			entity.EndPos = &endPos
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

// buildMaskedPIIText replaces every detected span with its placeholder.
//
// token_spans.v1 allows overlapping and nested spans (PERSON inside ADDRESS,
// EMAIL inside URL), so spans are first merged into a union of disjoint byte
// ranges and the replacements are applied from the end of the string
// backwards. Replacing one span at a time against an already modified string
// would let a later-starting span change the length and push an earlier,
// overlapping span past the end or onto the wrong boundary, leaving PII in
// the output. A merged range takes the placeholder of its longest span, so
// single, non-overlapping detections mask exactly as before.
func buildMaskedPIIText(text string, detections []classification.PIIDetection, placeholders map[string]string) string {
	type span struct {
		start, end  int
		placeholder string
	}
	spans := make([]span, 0, len(detections))
	for _, detection := range detections {
		if detection.Start < 0 || detection.End > len(text) || detection.Start >= detection.End {
			continue
		}
		spans = append(spans, span{
			start:       detection.Start,
			end:         detection.End,
			placeholder: placeholders[detection.EntityType+"\x00"+detection.Text],
		})
	}
	if len(spans) == 0 {
		return text
	}
	// Longest span first within the same start, so the merged range keeps the
	// outermost placeholder when spans nest.
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start != spans[j].start {
			return spans[i].start < spans[j].start
		}
		return spans[i].end > spans[j].end
	})
	merged := []span{spans[0]}
	for _, s := range spans[1:] {
		last := &merged[len(merged)-1]
		if s.start < last.end {
			if s.end > last.end {
				if s.end-s.start > last.end-last.start {
					last.placeholder = s.placeholder
				}
				last.end = s.end
			}
			continue
		}
		merged = append(merged, s)
	}
	maskedText := text
	for i := len(merged) - 1; i >= 0; i-- {
		m := merged[i]
		maskedText = maskedText[:m.start] + m.placeholder + maskedText[m.end:]
	}
	return maskedText
}

func piiSecurityRecommendation(hasPII bool) string {
	if hasPII {
		return "block"
	}
	return "allow"
}
