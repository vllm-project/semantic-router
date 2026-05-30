package classification

import (
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// parseSafetyOutput parses safety model output using the configured parser or auto-detection.
func (v *VLLMJailbreakInference) parseSafetyOutput(output string) (bool, float32, []string) {
	parserType := v.determineParserType()

	switch parserType {
	case "qwen3guard":
		return v.parseQwen3GuardFormat(output)
	case "json":
		isJailbreak, conf := v.parseJSONFormat(output)
		return isJailbreak, conf, nil
	case "simple":
		isJailbreak, conf := v.parseSimpleFormat(output)
		return isJailbreak, conf, nil
	case "auto":
		return v.parseSafetyOutputAuto(output)
	default:
		logging.Warnf("Unknown parser type: %s, using auto", parserType)
		return v.parseSafetyOutputAuto(output)
	}
}

func (v *VLLMJailbreakInference) parseSafetyOutputAuto(output string) (bool, float32, []string) {
	if result, conf, cats := v.parseQwen3GuardFormat(output); conf > 0.1 {
		return result, conf, cats
	}
	if result, conf := v.parseJSONFormat(output); conf > 0.1 {
		return result, conf, nil
	}
	isJailbreak, conf := v.parseSimpleFormat(output)
	return isJailbreak, conf, nil
}

func (v *VLLMJailbreakInference) determineParserType() string {
	if v.parserType != "" && v.parserType != "auto" {
		return v.parserType
	}

	modelLower := strings.ToLower(v.modelName)
	if strings.Contains(modelLower, "qwen3guard") || strings.Contains(modelLower, "qwen_guard") {
		return "qwen3guard"
	}
	if strings.Contains(modelLower, "json") {
		return "json"
	}
	return "auto"
}

func (v *VLLMJailbreakInference) parseQwen3GuardFormat(output string) (bool, float32, []string) {
	if matched, ok := v.parseQwen3GuardSafetyField(output); ok {
		return matched.isJailbreak, matched.confidence, matched.categories
	}
	if matched, ok := v.parseQwen3GuardSeverityField(output); ok {
		return matched.isJailbreak, matched.confidence, matched.categories
	}

	categories := v.extractCategories(output)
	if len(categories) > 0 {
		categoryStr := strings.ToLower(strings.Join(categories, ", "))
		if strings.Contains(categoryStr, "jailbreak") ||
			strings.Contains(categoryStr, "illegal") ||
			strings.Contains(categoryStr, "harmful") ||
			strings.Contains(categoryStr, "violence") ||
			strings.Contains(categoryStr, "hate") {
			logging.Debugf("Qwen3Guard parser (category): Categories=%v, isJailbreak=true, confidence=0.9", categories)
			return true, 0.9, categories
		}
	}

	logging.Warnf("Qwen3Guard parser failed to parse output: %s", output)
	return false, 0.0, nil
}

type qwen3GuardParseMatch struct {
	safety      string
	isJailbreak bool
	confidence  float32
	categories  []string
}

func (v *VLLMJailbreakInference) parseQwen3GuardSafetyField(output string) (qwen3GuardParseMatch, bool) {
	safetyRegex := regexp.MustCompile(`(?i)safety:\s*(safe|unsafe|controversial)`)
	matches := safetyRegex.FindStringSubmatch(output)
	if len(matches) < 2 {
		return qwen3GuardParseMatch{}, false
	}

	match := v.newQwen3GuardParseMatch(matches[1], output)
	logging.Debugf("Qwen3Guard parser: Safety=%s, isJailbreak=%v, confidence=%.3f, categories=%v",
		match.safety, match.isJailbreak, match.confidence, match.categories)
	return match, true
}

func (v *VLLMJailbreakInference) parseQwen3GuardSeverityField(output string) (qwen3GuardParseMatch, bool) {
	severityRegex := regexp.MustCompile(`(?i)severity\s+level:\s*(safe|unsafe|controversial)`)
	matches := severityRegex.FindStringSubmatch(output)
	if len(matches) < 2 {
		return qwen3GuardParseMatch{}, false
	}

	match := v.newQwen3GuardParseMatch(matches[1], output)
	logging.Debugf("Qwen3Guard parser (severity): Severity=%s, isJailbreak=%v, confidence=%.3f, categories=%v",
		match.safety, match.isJailbreak, match.confidence, match.categories)
	return match, true
}

func (v *VLLMJailbreakInference) newQwen3GuardParseMatch(safety string, output string) qwen3GuardParseMatch {
	normalized := strings.ToLower(strings.TrimSpace(safety))
	return qwen3GuardParseMatch{
		safety:      normalized,
		isJailbreak: normalized == "unsafe",
		confidence:  qwen3GuardConfidence(normalized),
		categories:  v.extractCategories(output),
	}
}

func qwen3GuardConfidence(safety string) float32 {
	switch safety {
	case "unsafe":
		return 0.9
	case "controversial":
		return 0.6
	case "safe":
		return 0.1
	default:
		return 0.5
	}
}

func (v *VLLMJailbreakInference) extractCategories(output string) []string {
	categoryRegex := regexp.MustCompile(`(?i)categories?:\s*([^\n]+)`)
	matches := categoryRegex.FindStringSubmatch(output)
	if len(matches) < 2 {
		return nil
	}

	categoryLine := strings.TrimSpace(matches[1])
	if strings.EqualFold(categoryLine, "None") {
		return nil
	}

	parts := strings.Split(categoryLine, ",")
	categories := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" && !strings.EqualFold(trimmed, "None") {
			categories = append(categories, trimmed)
		}
	}

	return categories
}

func (v *VLLMJailbreakInference) parseJSONFormat(output string) (bool, float32) {
	jsonRegex := regexp.MustCompile(`(?i)"safety":\s*"(safe|unsafe|controversial)"`)
	jsonMatches := jsonRegex.FindStringSubmatch(output)
	if len(jsonMatches) >= 2 {
		safety := strings.ToLower(strings.TrimSpace(jsonMatches[1]))
		return safety == "unsafe", qwen3GuardConfidence(safety)
	}

	boolRegex := regexp.MustCompile(`(?i)"(is_jailbreak|is_unsafe)":\s*(true|false)`)
	boolMatches := boolRegex.FindStringSubmatch(output)
	if len(boolMatches) >= 3 {
		if strings.ToLower(boolMatches[2]) == "true" {
			return true, 0.9
		}
		return false, 0.1
	}

	return false, 0.0
}

func (v *VLLMJailbreakInference) parseSimpleFormat(output string) (bool, float32) {
	outputLower := strings.ToLower(output)
	if strings.Contains(outputLower, "unsafe") || strings.Contains(outputLower, "jailbreak") {
		return true, 0.8
	}
	if strings.Contains(outputLower, "controversial") {
		return false, 0.6
	}
	if strings.Contains(outputLower, "safe") {
		return false, 0.1
	}
	return false, 0.5
}
