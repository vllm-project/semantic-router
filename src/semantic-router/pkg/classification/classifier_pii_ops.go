package classification

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ClassifyPII performs PII token classification on the given text and returns detected PII types
func (c *Classifier) ClassifyPII(ctx context.Context, text string) ([]string, error) {
	return c.ClassifyPIIWithThreshold(ctx, text, c.Config.PIIModel.Threshold)
}

// partialScanError reports a scan that returned valid spans for only part of
// its input. Callers that care get ErrTokenSpansTruncated alongside the
// results, so a partial answer is never mistaken for a complete one; callers
// that ignore the error keep the detections they were always given.
func partialScanError(partial bool) error {
	if partial {
		return ErrTokenSpansTruncated
	}
	return nil
}

// ErrPIIScanRefused marks a scan refused under classifier.pii.on_error: block
// because the provider declared it saw only part of the text. It does not
// unwrap to ErrTokenSpansTruncated on purpose: callers tolerate a declared
// truncation as a partial success, and a refusal must not pass through that
// tolerance as if it were one.
var ErrPIIScanRefused = errors.New("PII scan refused: provider truncated its input and on_error is block")

// piiScanError wraps a token classification failure for callers. A truncation
// under block becomes ErrPIIScanRefused, keeping the provider's message as
// text but not its sentinel in the chain.
func piiScanError(err error, block bool) error {
	if block && errors.Is(err, ErrTokenSpansTruncated) {
		return fmt.Errorf("%w: %s", ErrPIIScanRefused, err.Error())
	}
	return fmt.Errorf("PII token classification error: %w", err)
}

// ClassifyPIIWithThreshold performs PII token classification with a custom threshold
func (c *Classifier) ClassifyPIIWithThreshold(ctx context.Context, text string, threshold float32) ([]string, error) {
	if !c.IsPIIEnabled() {
		return []string{}, fmt.Errorf("PII detection is not properly configured")
	}

	if text == "" {
		return []string{}, nil
	}

	// Use ModernBERT PII token classifier for entity detection
	partial := false
	tokenResult, err := c.classifyPIITokens(ctx, text)
	if err != nil {
		// Same policy as the routing signal and scanPIIChunks: a declared
		// truncation carries valid spans for the part the provider saw, and
		// classifier.pii.on_error decides what the unseen remainder means.
		if !errors.Is(err, ErrTokenSpansTruncated) || c.Config.PIIModel.IsBlock() {
			return nil, piiScanError(err, c.Config.PIIModel.IsBlock())
		}
		logging.Warnf("PII classification: provider truncated its input; reporting the types it did return")
		partial = true
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

	return result, partialScanError(partial)
}

// ClassifyPIIWithDetails performs PII token classification and returns full entity details including confidence scores
func (c *Classifier) ClassifyPIIWithDetails(ctx context.Context, text string) ([]PIIDetection, error) {
	return c.ClassifyPIIWithDetailsAndThreshold(ctx, text, c.Config.PIIModel.Threshold)
}

// ClassifyPIIWithDetailsAndThreshold performs PII token classification with a custom threshold and returns full entity details
func (c *Classifier) ClassifyPIIWithDetailsAndThreshold(ctx context.Context, text string, threshold float32) ([]PIIDetection, error) {
	if !c.IsPIIEnabled() {
		return []PIIDetection{}, fmt.Errorf("PII detection is not properly configured")
	}

	if text == "" {
		return []PIIDetection{}, nil
	}

	detections, err := c.scanPIIChunks(ctx, text, threshold)
	if err != nil && !errors.Is(err, ErrTokenSpansTruncated) {
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

	return detections, err
}

// scanPIIChunks runs PII token classification over the whole text.
//
// The classifier truncates at MAX_CLASSIFICATION_SEQ_LEN, so one call over a
// long text only ever scores its start. The PII routing signal already scans in
// bounded overlapping chunks (evaluatePIISignal); this does the same, and maps
// every entity back onto the original text because this surface reports
// positions and the routing signal does not.
func (c *Classifier) scanPIIChunks(ctx context.Context, text string, threshold float32) ([]PIIDetection, error) {
	var detections []PIIDetection
	seen := make(map[piiDetectionKey]int)
	classified := 0
	partial := false

	for _, span := range c.piiInputSpans(text) {
		tokenResult, err := c.classifyPIITokens(ctx, span.Text)
		if err != nil {
			// A declared truncation carries valid spans for the part the
			// provider saw. classifier.pii.on_error decides what the unseen
			// remainder means here too: block refuses the whole scan, allow
			// reports what was found. Any other error is fatal either way.
			if !errors.Is(err, ErrTokenSpansTruncated) || c.Config.PIIModel.IsBlock() {
				return nil, piiScanError(err, c.Config.PIIModel.IsBlock())
			}
			logging.Warnf("PII scan: provider truncated its input; reporting the spans it did return")
			partial = true
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
	detections, err := coverRepeatedValues(ctx, text, detections)
	if err != nil {
		return nil, err
	}

	// A single call returned entities in ascending position; keep that contract
	// now that they arrive chunk by chunk.
	sort.SliceStable(detections, func(i, j int) bool {
		if detections[i].Start != detections[j].Start {
			return detections[i].Start < detections[j].Start
		}
		return detections[i].End < detections[j].End
	})

	return detections, partialScanError(partial)
}

// coverRepeatedValues adds a detection for every other copy of a detected
// value. A model can label a value once and miss its later copies (Vela 2.0
// scores the second "John Smith" of a request far below the first), and
// masking works from positions, so a missed copy would reach the provider in
// clear text. Only exact, word-aligned copies of at least two characters
// count, and only a copy that a detection already covers whole is skipped:
// a model span over part of a copy still leaves the rest in clear text.
// Each distinct value is searched once, and coverage is a binary search over
// the covered spans, so the work stays linear in the text per value.
func coverRepeatedValues(ctx context.Context, text string, detections []PIIDetection) ([]PIIDetection, error) {
	covered := coveredSpans(detections)
	searched := make(map[string]bool)
	found := len(detections)
	for i := 0; i < found; i++ {
		source := detections[i]
		if source.Start < 0 || source.End > len(text) || source.Start >= source.End {
			continue
		}
		value := text[source.Start:source.End]
		if searched[value] || utf8.RuneCountInString(value) < 2 {
			continue
		}
		searched[value] = true
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var added []piiSpan
		for from := 0; ; {
			index := strings.Index(text[from:], value)
			if index < 0 {
				break
			}
			start := from + index
			end := start + len(value)
			from = end
			if wordAligned(text, start, end) && !containsSpan(covered, start, end) {
				added = append(added, piiSpan{start, end})
				detections = append(detections, PIIDetection{
					EntityType: source.EntityType,
					Start:      start,
					End:        end,
					Text:       value,
					Confidence: source.Confidence,
				})
			}
		}
		covered = mergeSpans(append(covered, added...))
	}
	return detections, nil
}

type piiSpan struct{ start, end int }

// coveredSpans is the union of the detections as sorted, disjoint spans.
func coveredSpans(detections []PIIDetection) []piiSpan {
	spans := make([]piiSpan, 0, len(detections))
	for _, detection := range detections {
		spans = append(spans, piiSpan{detection.Start, detection.End})
	}
	return mergeSpans(spans)
}

// mergeSpans sorts spans and joins the ones that overlap or touch.
func mergeSpans(spans []piiSpan) []piiSpan {
	sort.Slice(spans, func(a, b int) bool { return spans[a].start < spans[b].start })
	merged := spans[:0]
	for _, span := range spans {
		if last := len(merged) - 1; last >= 0 && span.start <= merged[last].end {
			merged[last].end = max(merged[last].end, span.end)
			continue
		}
		merged = append(merged, span)
	}
	return merged
}

// containsSpan reports whether one of the sorted, disjoint spans contains
// [start, end).
func containsSpan(spans []piiSpan, start, end int) bool {
	i := sort.Search(len(spans), func(i int) bool { return spans[i].end > start })
	return i < len(spans) && spans[i].start <= start && spans[i].end >= end
}

// wordAligned reports whether text[start:end] is not part of a longer word.
func wordAligned(text string, start, end int) bool {
	isWord := func(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) }
	if before, _ := utf8.DecodeLastRuneInString(text[:start]); start > 0 && isWord(before) {
		return false
	}
	if after, _ := utf8.DecodeRuneInString(text[end:]); end < len(text) && isWord(after) {
		return false
	}
	return true
}

// piiDetectionKey identifies one entity occurrence in the original text.
type piiDetectionKey struct {
	entityType string
	start      int
	end        int
}

// DetectPIIInContent performs PII classification on all provided content
func (c *Classifier) DetectPIIInContent(ctx context.Context, allContent []string) []string {
	var detectedPII []string
	seenPII := make(map[string]bool)

	for _, content := range allContent {
		if content == "" {
			continue
		}
		// TODO: classifier may not handle the entire content, so we need to split the content into smaller chunks
		piiTypes, err := c.ClassifyPII(ctx, content)
		if err != nil && !errors.Is(err, ErrTokenSpansTruncated) {
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
func (c *Classifier) AnalyzeContentForPII(ctx context.Context, contentList []string) (bool, []PIIAnalysisResult, error) {
	return c.AnalyzeContentForPIIWithThreshold(ctx, contentList, c.Config.PIIModel.Threshold)
}

// AnalyzeContentForPIIWithThreshold performs detailed PII analysis with a custom threshold
func (c *Classifier) AnalyzeContentForPIIWithThreshold(ctx context.Context, contentList []string, threshold float32) (bool, []PIIAnalysisResult, error) {
	if !c.IsPIIEnabled() {
		return false, nil, fmt.Errorf("PII detection is not properly configured")
	}

	var analysisResults []PIIAnalysisResult
	hasPII := false
	failedCount := 0
	partial := false
	var lastErr error

	for i, content := range contentList {
		if content == "" {
			continue
		}

		var result PIIAnalysisResult
		result.Content = content
		result.ContentIndex = i

		// Use ModernBERT PII token classifier for detailed analysis
		tokenResult, err := c.classifyPIITokens(ctx, content)
		if err != nil {
			// As in scanPIIChunks: a truncation still carries valid spans, and
			// on_error decides whether the unseen remainder voids them. Under
			// block the batch is refused outright rather than returning a
			// result set that silently omits the item nobody could verify -
			// the other two entry points already refuse, and an omitted item
			// is indistinguishable from a clean one to the caller.
			if c.Config.PIIModel.IsBlock() {
				return false, nil, fmt.Errorf("PII classification failed for content %d and on_error is block: %w", i, piiScanError(err, true))
			}
			if !errors.Is(err, ErrTokenSpansTruncated) {
				logging.Errorf("Error analyzing content %d: %v", i, err)
				failedCount++
				lastErr = err
				continue
			}
			logging.Warnf("PII analysis of content %d: provider truncated its input; keeping the spans it did return", i)
			partial = true
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

	return hasPII, analysisResults, partialScanError(partial)
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
func (c *Classifier) collectPIIEntityTypes(ruleContents []string, ruleName string, threshold float32, piiCache map[string][]cachedPIIResult) (map[string]bool, bool) {
	entityTypes := make(map[string]bool)
	failed := false
	for _, content := range ruleContents {
		cachedResults, ok := piiCache[content]
		if !ok {
			continue
		}
		for _, cached := range cachedResults {
			if cached.err != nil {
				failed = true
				if !errors.Is(cached.err, ErrTokenSpansTruncated) {
					logging.Errorf("[Signal Computation] PII rule %q: inference error: %v", ruleName, cached.err)
					continue
				}
				// A declared truncation still carries valid spans for the part
				// the provider saw; they count, and the rule is marked as not
				// fully evaluated so on_error decides what the unseen part means.
				logging.Warnf("[Signal Computation] PII rule %q: provider truncated its input, spans are partial", ruleName)
			}
			for _, entity := range cached.result.Entities {
				if entity.Confidence >= threshold {
					entityTypes[c.PIIMapping.TranslatePIIType(entity.EntityType)] = true
				}
			}
		}
	}
	return entityTypes, failed
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
