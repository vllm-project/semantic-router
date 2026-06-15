package classification

import (
	"context"
	"fmt"
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
