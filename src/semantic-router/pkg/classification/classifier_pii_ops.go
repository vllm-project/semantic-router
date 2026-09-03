package classification

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ClassifyPII performs PII token classification on the given text and returns detected PII types
func (c *Classifier) ClassifyPII(text string) ([]string, error) {
	return c.ClassifyPIIWithThreshold(text, c.Config.PIIModel.Threshold)
}

// ClassifyPIIWithThreshold performs PII token classification with a custom threshold
func (c *Classifier) ClassifyPIIWithThreshold(text string, threshold float32) ([]string, error) {
	if !c.IsPIIEnabled() {
		return []string{}, fmt.Errorf("PII detection is not properly configured")
	}

	if text == "" {
		return []string{}, nil
	}

	// Use ModernBERT PII token classifier for entity detection
	tokenResult, err := c.piiInference.ClassifyTokens(text)
	if err != nil {
		return nil, fmt.Errorf("PII token classification error: %w", err)
	}

	if len(tokenResult.Entities) > 0 {
		logging.Infof("PII token classification found %d entities", len(tokenResult.Entities))
	}

	// Extract unique PII types from detected entities
	// Translate class_X format to named types using PII mapping
	piiTypes := make(map[string]bool)
	for _, entity := range tokenResult.Entities {
		if entity.Confidence >= threshold {
			// Translate entity type from class_X format to named type (e.g., class_6 → DATE_TIME)
			translatedType := c.PIIMapping.TranslatePIIType(entity.EntityType)
			piiTypes[translatedType] = true
			logging.Infof("Detected PII entity: %s → %s ('%s') at [%d-%d] with confidence %.3f",
				entity.EntityType, translatedType, entity.Text, entity.Start, entity.End, entity.Confidence)
		}
	}

	// Convert to slice
	var result []string
	for piiType := range piiTypes {
		result = append(result, piiType)
	}

	if len(result) > 0 {
		logging.Infof("Detected PII types: %v", result)
	}

	return result, nil
}

// ClassifyPIIWithDetails performs PII token classification and returns full entity details including confidence scores
func (c *Classifier) ClassifyPIIWithDetails(text string) ([]PIIDetection, error) {
	return c.ClassifyPIIWithDetailsAndThreshold(text, c.Config.PIIModel.Threshold)
}

// ClassifyPIIWithDetailsAndThreshold performs PII token classification with a custom threshold and returns full entity details
func (c *Classifier) ClassifyPIIWithDetailsAndThreshold(text string, threshold float32) ([]PIIDetection, error) {
	if !c.IsPIIEnabled() {
		return []PIIDetection{}, fmt.Errorf("PII detection is not properly configured")
	}

	if text == "" {
		return []PIIDetection{}, nil
	}

	detections, err := c.scanPIIChunks(text, threshold)
	if err != nil {
		return nil, err
	}

	if len(detections) > 0 {
		// Log unique PII types for compatibility with existing logs
		uniqueTypes := make(map[string]bool)
		for _, d := range detections {
			uniqueTypes[d.EntityType] = true
		}
		types := make([]string, 0, len(uniqueTypes))
		for t := range uniqueTypes {
			types = append(types, t)
		}
		logging.Infof("Detected PII types: %v", types)
	}

	return detections, nil
}

// scanPIIChunks runs PII token classification over the whole text.
//
// The classifier truncates at MAX_CLASSIFICATION_SEQ_LEN, so one call over a
// long text only ever scores its start. The PII routing signal already scans in
// bounded overlapping chunks (evaluatePIISignal); this does the same, and maps
// every entity back onto the original text because this surface reports
// positions and the routing signal does not.
func (c *Classifier) scanPIIChunks(text string, threshold float32) ([]PIIDetection, error) {
	var detections []PIIDetection
	seen := make(map[piiDetectionKey]int)
	classified := 0

	for _, span := range piiSignalChunkSpans(text) {
		tokenResult, err := c.piiInference.ClassifyTokens(span.Text)
		if err != nil {
			return nil, fmt.Errorf("PII token classification error: %w", err)
		}

		classified += len(tokenResult.Entities)

		for _, entity := range tokenResult.Entities {
			if entity.Confidence < threshold {
				continue
			}

			// Translate entity type from class_X format to named type (e.g., class_6 → DATE_TIME)
			translatedType := c.PIIMapping.TranslatePIIType(entity.EntityType)
			detection := PIIDetection{
				EntityType: translatedType,
				Start:      span.StartByte + entity.Start,
				End:        span.StartByte + entity.End,
				Text:       entity.Text,
				Confidence: entity.Confidence,
			}

			// Chunks overlap, so an entity inside the overlap window is reported
			// by both chunks. Keep one detection at the higher confidence.
			key := piiDetectionKey{translatedType, detection.Start, detection.End}
			if existing, reported := seen[key]; reported {
				if detection.Confidence > detections[existing].Confidence {
					detections[existing].Confidence = detection.Confidence
				}
				continue
			}
			seen[key] = len(detections)
			detections = append(detections, detection)

			logging.Infof("Detected PII entity: %s → %s ('%s') at [%d-%d] with confidence %.3f",
				entity.EntityType, translatedType, entity.Text, detection.Start, detection.End, entity.Confidence)
		}
	}

	if classified > 0 {
		logging.Infof("PII token classification found %d entities", classified)
	}

	// A single call returned entities in ascending position; keep that contract
	// now that they arrive chunk by chunk.
	sort.SliceStable(detections, func(i, j int) bool {
		if detections[i].Start != detections[j].Start {
			return detections[i].Start < detections[j].Start
		}
		return detections[i].End < detections[j].End
	})

	return detections, nil
}

// piiDetectionKey identifies one entity occurrence in the original text.
type piiDetectionKey struct {
	entityType string
	start      int
	end        int
}

// DetectPIIInContent performs PII classification on all provided content
func (c *Classifier) DetectPIIInContent(allContent []string) []string {
	var detectedPII []string
	seenPII := make(map[string]bool)

	for _, content := range allContent {
		if content == "" {
			continue
		}
		// TODO: classifier may not handle the entire content, so we need to split the content into smaller chunks
		piiTypes, err := c.ClassifyPII(content)
		if err != nil {
			logging.Errorf("PII classification error: %v", err)
			// Continue without PII enforcement on error
			continue
		}
		// Add all detected PII types, avoiding duplicates
		for _, piiType := range piiTypes {
			if seenPII[piiType] {
				continue
			}
			detectedPII = append(detectedPII, piiType)
			seenPII[piiType] = true
			logging.Infof("Detected PII type '%s' in content", piiType)
		}
	}

	return detectedPII
}

// AnalyzeContentForPII performs detailed PII analysis on multiple content pieces
func (c *Classifier) AnalyzeContentForPII(contentList []string) (bool, []PIIAnalysisResult, error) {
	return c.AnalyzeContentForPIIWithThreshold(contentList, c.Config.PIIModel.Threshold)
}

// AnalyzeContentForPIIWithThreshold performs detailed PII analysis with a custom threshold
func (c *Classifier) AnalyzeContentForPIIWithThreshold(contentList []string, threshold float32) (bool, []PIIAnalysisResult, error) {
	if !c.IsPIIEnabled() {
		return false, nil, fmt.Errorf("PII detection is not properly configured")
	}

	var analysisResults []PIIAnalysisResult
	hasPII := false
	failedCount := 0
	var lastErr error

	for i, content := range contentList {
		if content == "" {
			continue
		}

		var result PIIAnalysisResult
		result.Content = content
		result.ContentIndex = i

		// Use ModernBERT PII token classifier for detailed analysis
		tokenResult, err := c.piiInference.ClassifyTokens(content)
		if err != nil {
			logging.Errorf("Error analyzing content %d: %v", i, err)
			failedCount++
			lastErr = err
			continue
		}

		// Convert token entities to PII detections
		for _, entity := range tokenResult.Entities {
			if entity.Confidence >= threshold {
				detection := PIIDetection{
					EntityType: entity.EntityType,
					Start:      entity.Start,
					End:        entity.End,
					Text:       entity.Text,
					Confidence: entity.Confidence,
				}
				result.Entities = append(result.Entities, detection)
				result.HasPII = true
				hasPII = true
			}
		}

		analysisResults = append(analysisResults, result)
	}

	// Fail closed: individual inference failures are tolerated as long as some
	// content was actually classified, but if nothing could be classified the
	// caller must not receive a benign "no PII" verdict it cannot distinguish
	// from a clean scan.
	if failedCount > 0 && len(analysisResults) == 0 {
		return false, nil, fmt.Errorf("PII classification failed for all %d content item(s): %w", failedCount, lastErr)
	}

	return hasPII, analysisResults, nil
}

// collectPIIRuleContents builds the list of text contents to analyze for a PII rule.
func collectPIIRuleContents(piiText string, nonUserMessages []string, includeHistory bool) []string {
	var contents []string
	if piiText != "" {
		contents = append(contents, piiText)
	}
	if includeHistory {
		for _, msg := range nonUserMessages {
			if msg != "" {
				contents = append(contents, msg)
			}
		}
	}
	return contents
}

// collectPIIEntityTypes extracts entity types from cached PII results that meet the threshold.
func (c *Classifier) collectPIIEntityTypes(ruleContents []string, ruleName string, threshold float32, piiCache map[string][]cachedPIIResult) map[string]bool {
	entityTypes := make(map[string]bool)
	for _, content := range ruleContents {
		cachedResults, ok := piiCache[content]
		if !ok {
			continue
		}
		for _, cached := range cachedResults {
			if cached.err != nil {
				logging.Errorf("[Signal Computation] PII rule %q: inference error: %v", ruleName, cached.err)
				continue
			}
			for _, entity := range cached.result.Entities {
				if entity.Confidence >= threshold {
					entityTypes[c.PIIMapping.TranslatePIIType(entity.EntityType)] = true
				}
			}
		}
	}
	return entityTypes
}

// findDeniedEntities returns entity types not covered by the allow-list.
func findDeniedEntities(entityTypes map[string]bool, allowedTypes []string) []string {
	allowSet := make(map[string]bool, len(allowedTypes))
	for _, allowed := range allowedTypes {
		allowSet[strings.ToUpper(allowed)] = true
	}
	var denied []string
	for entityType := range entityTypes {
		if !allowSet[strings.ToUpper(entityType)] {
			denied = append(denied, entityType)
		}
	}
	return denied
}
