package classification

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// VLLMJailbreakInference implements JailbreakInference using vLLM REST API
type VLLMJailbreakInference struct {
	client     *VLLMClient
	modelName  string
	threshold  float32
	timeout    time.Duration
	parserType string // Parser type: "qwen3guard", "json", "simple", "auto"
}

// NewVLLMJailbreakInference creates a new vLLM-based jailbreak inference instance
// Takes ExternalModelConfig directly
func NewVLLMJailbreakInference(cfg *config.ExternalModelConfig, defaultThreshold float32) (*VLLMJailbreakInference, error) {
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("vLLM endpoint address is required for guardrail")
	}
	if cfg.ModelName == "" {
		return nil, fmt.Errorf("vLLM model name is required for guardrail")
	}

	// Create client with or without access key
	var client *VLLMClient
	if cfg.AccessKey != "" {
		client = NewVLLMClientWithAuth(&cfg.ModelEndpoint, cfg.AccessKey)
	} else {
		client = NewVLLMClient(&cfg.ModelEndpoint)
	}

	// Use timeout from config, default to 30 seconds
	timeout := 30 * time.Second
	if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	// Use threshold from config, fallback to default
	threshold := defaultThreshold
	if cfg.Threshold > 0 {
		threshold = cfg.Threshold
	}

	// Use parser type from config, default to "auto"
	parserType := cfg.ParserType
	if parserType == "" {
		parserType = "auto"
	}

	return &VLLMJailbreakInference{
		client:     client,
		modelName:  cfg.ModelName,
		threshold:  threshold,
		timeout:    timeout,
		parserType: parserType,
	}, nil
}

// Classify implements the JailbreakInference interface
func (v *VLLMJailbreakInference) Classify(text string) (candle_binding.ClassResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), v.timeout)
	defer cancel()

	// Format prompt - flexible to support different models
	// If custom template is set in endpoint config, it will be used by the client
	// Otherwise, use a generic safety analysis prompt
	prompt := fmt.Sprintf("Analyze the following text for safety and security concerns. Determine if it contains jailbreak attempts, harmful content, or violates safety guidelines:\n\n%s", text)

	// Call vLLM API
	resp, err := v.client.Generate(ctx, v.modelName, prompt, &GenerationOptions{
		MaxTokens:   512,
		Temperature: 0.0, // Deterministic for safety checks
	})
	if err != nil {
		return candle_binding.ClassResult{}, fmt.Errorf("vLLM API call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return candle_binding.ClassResult{}, fmt.Errorf("no choices in vLLM response")
	}

	// Parse model output - flexible to support multiple formats
	output := resp.Choices[0].Message.Content
	logging.Debugf("vLLM jailbreak detection response: %s", output)
	isJailbreak, confidence, categories := v.parseSafetyOutput(output)
	logging.Debugf("Parsed result: isJailbreak=%v, confidence=%.3f, categories=%v",
		isJailbreak, confidence, categories)

	// Map to ClassResult format
	// Class: 0 = safe, 1 = jailbreak/unsafe
	class := 0
	if isJailbreak {
		class = 1
	}

	result := candle_binding.ClassResult{
		Class:      class,
		Confidence: confidence,
	}

	// Only populate categories when content is unsafe or controversial
	// (empty slice for safe content or when categories not available)
	if isJailbreak && len(categories) > 0 {
		result.Categories = categories
	}

	return result, nil
}

// parseSafetyOutput parses safety model output - uses parser type or auto-detection
func (v *VLLMJailbreakInference) parseSafetyOutput(output string) (bool, float32, []string) {
	// Determine parser type based on configuration or model name
	parserType := v.determineParserType()

	switch parserType {
	case "qwen3guard":
		return v.parseQwen3GuardFormat(output)
	case "json":
		isJailbreak, conf := v.parseJSONFormat(output)
		return isJailbreak, conf, nil // JSON parser doesn't support categories yet
	case "simple":
		isJailbreak, conf := v.parseSimpleFormat(output)
		return isJailbreak, conf, nil // Simple parser doesn't support categories yet
	case "auto":
		// Try all parsers (OR logic) until one succeeds
		if result, conf, cats := v.parseQwen3GuardFormat(output); conf > 0.1 {
			return result, conf, cats
		}
		if result, conf := v.parseJSONFormat(output); conf > 0.1 {
			return result, conf, nil
		}
		isJailbreak, conf := v.parseSimpleFormat(output)
		return isJailbreak, conf, nil
	default:
		logging.Warnf("Unknown parser type: %s, using auto", parserType)
		// Fallback to auto mode
		if result, conf, cats := v.parseQwen3GuardFormat(output); conf > 0.1 {
			return result, conf, cats
		}
		if result, conf := v.parseJSONFormat(output); conf > 0.1 {
			return result, conf, nil
		}
		isJailbreak, conf := v.parseSimpleFormat(output)
		return isJailbreak, conf, nil
	}
}

// determineParserType determines which parser to use based on config or model name
func (v *VLLMJailbreakInference) determineParserType() string {
	// If explicitly configured, use that
	if v.parserType != "" && v.parserType != "auto" {
		return v.parserType
	}

	// Auto-detect from model name
	modelLower := strings.ToLower(v.modelName)
	if strings.Contains(modelLower, "qwen3guard") || strings.Contains(modelLower, "qwen_guard") {
		return "qwen3guard"
	}
	if strings.Contains(modelLower, "json") {
		return "json"
	}

	// Default to auto (try all)
	return "auto"
}

// parseQwen3GuardFormat parses Qwen3Guard structured output
func (v *VLLMJailbreakInference) parseQwen3GuardFormat(output string) (bool, float32, []string) {
	// Qwen3Guard actual format: "Safety: Unsafe\nCategories: ..."
	// Try "Safety:" format first (actual Qwen3Guard output)
	safetyRegex := regexp.MustCompile(`(?i)safety:\s*(safe|unsafe|controversial)`)
	matches := safetyRegex.FindStringSubmatch(output)
	if len(matches) >= 2 {
		safety := strings.ToLower(strings.TrimSpace(matches[1]))
		isJailbreak := safety == "unsafe"
		var confidence float32
		switch safety {
		case "unsafe":
			confidence = 0.9
		case "controversial":
			confidence = 0.6
		case "safe":
			confidence = 0.1
		default:
			confidence = 0.5
		}

		// Extract categories from output
		categories := v.extractCategories(output)
		logging.Debugf("Qwen3Guard parser: Safety=%s, isJailbreak=%v, confidence=%.3f, categories=%v",
			safety, isJailbreak, confidence, categories)
		return isJailbreak, confidence, categories
	}

	// Try "Severity level:" format (alternative format)
	severityRegex := regexp.MustCompile(`(?i)severity\s+level:\s*(safe|unsafe|controversial)`)
	severityMatches := severityRegex.FindStringSubmatch(output)
	if len(severityMatches) >= 2 {
		severity := strings.ToLower(strings.TrimSpace(severityMatches[1]))
		isJailbreak := severity == "unsafe"
		var confidence float32
		switch severity {
		case "unsafe":
			confidence = 0.9
		case "controversial":
			confidence = 0.6
		case "safe":
			confidence = 0.1
		default:
			confidence = 0.5
		}

		// Extract categories from output
		categories := v.extractCategories(output)
		logging.Debugf("Qwen3Guard parser (severity): Severity=%s, isJailbreak=%v, confidence=%.3f, categories=%v",
			severity, isJailbreak, confidence, categories)
		return isJailbreak, confidence, categories
	}

	// Try category-based detection (Categories: or Category:)
	// Extract categories even if safety level wasn't found
	categories := v.extractCategories(output)
	if len(categories) > 0 {
		// Check if any category indicates unsafe content
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
	return false, 0.0, nil // Failed to parse
}

// extractCategories extracts violation categories from Qwen3Guard output
// Returns empty slice if no categories found or if "None" is specified
func (v *VLLMJailbreakInference) extractCategories(output string) []string {
	// Pattern matches: "Categories: Violent" or "Categories: Violent, Jailbreak"
	categoryRegex := regexp.MustCompile(`(?i)categories?:\s*([^\n]+)`)
	matches := categoryRegex.FindStringSubmatch(output)
	if len(matches) < 2 {
		return nil
	}

	categoryLine := strings.TrimSpace(matches[1])

	// Handle "None" case
	if strings.EqualFold(categoryLine, "None") {
		return nil
	}

	// Split by comma and trim each category
	parts := strings.Split(categoryLine, ",")
	var categories []string
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" && !strings.EqualFold(trimmed, "None") {
			categories = append(categories, trimmed)
		}
	}

	return categories
}

// parseJSONFormat parses JSON output
func (v *VLLMJailbreakInference) parseJSONFormat(output string) (bool, float32) {
	// Try to find JSON safety field
	jsonRegex := regexp.MustCompile(`(?i)"safety":\s*"(safe|unsafe|controversial)"`)
	jsonMatches := jsonRegex.FindStringSubmatch(output)
	if len(jsonMatches) >= 2 {
		safety := strings.ToLower(strings.TrimSpace(jsonMatches[1]))
		isJailbreak := safety == "unsafe"
		confidence := float32(0.9)
		switch safety {
		case "controversial":
			confidence = 0.6
		case "safe":
			confidence = 0.1
		}
		return isJailbreak, confidence
	}

	// Try is_jailbreak or is_unsafe boolean fields
	boolRegex := regexp.MustCompile(`(?i)"(is_jailbreak|is_unsafe)":\s*(true|false)`)
	boolMatches := boolRegex.FindStringSubmatch(output)
	if len(boolMatches) >= 3 {
		if strings.ToLower(boolMatches[2]) == "true" {
			return true, 0.9
		}
		return false, 0.1
	}

	return false, 0.0 // Failed to parse
}

// parseSimpleFormat parses simple keyword-based output
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
	return false, 0.5 // Default fallback
}
