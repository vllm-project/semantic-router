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
func NewVLLMJailbreakInference(cfg *config.PromptGuardConfig) (*VLLMJailbreakInference, error) {
	// Use dedicated classifier vLLM endpoint from PromptGuardConfig
	// This is separate from backend inference endpoints
	if cfg.ClassifierVLLMEndpoint.Address == "" {
		return nil, fmt.Errorf("classifier_vllm_endpoint.address is required for PromptGuard")
	}

	client := NewVLLMClient(&cfg.ClassifierVLLMEndpoint)
	timeout := 30 * time.Second
	if cfg.VLLMTimeoutSeconds > 0 {
		timeout = time.Duration(cfg.VLLMTimeoutSeconds) * time.Second
	}

	// Determine parser type from config or auto-detect
	parserType := cfg.ResponseParserType
	if parserType == "" {
		parserType = "auto" // Default to auto-detection
	}

	return &VLLMJailbreakInference{
		client:     client,
		modelName:  cfg.VLLMModelName,
		threshold:  cfg.Threshold,
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
	isJailbreak, confidence := v.parseSafetyOutput(output)
	logging.Debugf("Parsed result: isJailbreak=%v, confidence=%.3f", isJailbreak, confidence)

	// Map to ClassResult format
	// Class: 0 = safe, 1 = jailbreak/unsafe
	class := 0
	if isJailbreak {
		class = 1
	}

	return candle_binding.ClassResult{
		Class:      class,
		Confidence: confidence,
	}, nil
}

// parseSafetyOutput parses safety model output - uses parser type or auto-detection
func (v *VLLMJailbreakInference) parseSafetyOutput(output string) (bool, float32) {
	// Determine parser type based on configuration or model name
	parserType := v.determineParserType()

	switch parserType {
	case "qwen3guard":
		return v.parseQwen3GuardFormat(output)
	case "json":
		return v.parseJSONFormat(output)
	case "simple":
		return v.parseSimpleFormat(output)
	case "auto":
		// Try all parsers (OR logic) until one succeeds
		if result, conf := v.parseQwen3GuardFormat(output); conf > 0.1 {
			return result, conf
		}
		if result, conf := v.parseJSONFormat(output); conf > 0.1 {
			return result, conf
		}
		return v.parseSimpleFormat(output)
	default:
		logging.Warnf("Unknown parser type: %s, using auto", parserType)
		// Fallback to auto mode
		if result, conf := v.parseQwen3GuardFormat(output); conf > 0.1 {
			return result, conf
		}
		if result, conf := v.parseJSONFormat(output); conf > 0.1 {
			return result, conf
		}
		return v.parseSimpleFormat(output)
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
func (v *VLLMJailbreakInference) parseQwen3GuardFormat(output string) (bool, float32) {
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
		logging.Debugf("Qwen3Guard parser: Safety=%s, isJailbreak=%v, confidence=%.3f", safety, isJailbreak, confidence)
		return isJailbreak, confidence
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
		logging.Debugf("Qwen3Guard parser (severity): Severity=%s, isJailbreak=%v, confidence=%.3f", severity, isJailbreak, confidence)
		return isJailbreak, confidence
	}

	// Try category-based detection (Categories: or Category:)
	categoryRegex := regexp.MustCompile(`(?i)categories?:\s*([^\n]+)`)
	catMatches := categoryRegex.FindStringSubmatch(output)
	if len(catMatches) >= 2 {
		category := strings.ToLower(strings.TrimSpace(catMatches[1]))
		// Check for jailbreak-related categories
		if strings.Contains(category, "jailbreak") ||
			strings.Contains(category, "illegal") ||
			strings.Contains(category, "harmful") ||
			strings.Contains(category, "violence") ||
			strings.Contains(category, "hate") {
			logging.Debugf("Qwen3Guard parser (category): Category=%s, isJailbreak=true, confidence=0.9", category)
			return true, 0.9
		}
	}

	logging.Warnf("Qwen3Guard parser failed to parse output: %s", output)
	return false, 0.0 // Failed to parse
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
