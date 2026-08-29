package classification

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// VLLMJailbreakInference implements SequenceClassifierBackend using vLLM REST API
type VLLMJailbreakInference struct {
	client         *VLLMClient
	modelName      string
	threshold      float32
	timeout        time.Duration
	parserType     string // Parser type: "qwen3guard", "json", "simple", "auto"
	mapping        *JailbreakMapping
	positiveLabels []string
	positiveIdx    int
	negativeIdx    int
}

// NewVLLMJailbreakInference creates a new vLLM-based jailbreak inference instance
// Takes ExternalModelConfig directly. mapping must describe exactly two classes:
// this backend only ever produces a binary safe/jailbreak verdict plus a single
// confidence score (the underlying model is generative, not a calibrated
// sequence classifier), so it cannot populate a meaningful distribution over
// more than two classes. The positive class's index is resolved from mapping
// rather than assumed to be a fixed position, so a mapping that orders classes
// differently than {safe: 0, jailbreak: 1} is still scored correctly.
func NewVLLMJailbreakInference(cfg *config.ExternalModelConfig, defaultThreshold float32, mapping *JailbreakMapping, positiveLabels []string) (*VLLMJailbreakInference, error) {
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("vLLM endpoint address is required for guardrail")
	}
	if cfg.ModelName == "" {
		return nil, fmt.Errorf("vLLM model name is required for guardrail")
	}
	if mapping == nil {
		return nil, fmt.Errorf("jailbreak mapping is required for http_chat")
	}
	if mapping.GetJailbreakTypeCount() != 2 {
		return nil, fmt.Errorf("http_chat requires a 2-class (safe/jailbreak) jailbreak_mapping, got %d classes", mapping.GetJailbreakTypeCount())
	}
	positiveIdx, err := resolveSinglePositiveIndex(mapping, positiveLabels)
	if err != nil {
		return nil, err
	}

	client := newVLLMClientFromConfig(cfg)

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
		client:         client,
		modelName:      cfg.ModelName,
		threshold:      threshold,
		timeout:        timeout,
		parserType:     parserType,
		mapping:        mapping,
		positiveLabels: positiveLabels,
		positiveIdx:    positiveIdx,
		negativeIdx:    1 - positiveIdx,
	}, nil
}

// Classify implements the SequenceClassifierBackend interface. Since this
// backend is generative (a chat model asked to judge safety), it cannot
// produce a calibrated softmax distribution; it derives a 2-class
// (safe/jailbreak) distribution from the model's own confidence estimate, so
// callers always get a full distribution regardless of which backend is
// configured, consistent with candle-backed jailbreak backends. The
// distribution is written at whichever indices the configured jailbreak_mapping
// actually assigns to the positive/negative labels (resolved once in the
// constructor), not a hardcoded {0: safe, 1: jailbreak} layout - a mapping
// that orders classes the other way around would otherwise silently invert
// every verdict this backend reports. ctx bounds the vLLM call so the caller's
// cancellation/deadline propagates instead of always running to v.timeout.
func (v *VLLMJailbreakInference) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	ctx, cancel := context.WithTimeout(ctx, v.timeout)
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
		return SequenceClassificationResult{}, fmt.Errorf("vLLM API call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return SequenceClassificationResult{}, fmt.Errorf("no choices in vLLM response")
	}

	// Parse model output - flexible to support multiple formats
	output := resp.Choices[0].Message.Content
	logging.Debugf("vLLM jailbreak detection response: %s", logging.ContentDescriptor(output))
	isJailbreak, confidence, categories := v.parseSafetyOutput(output)
	logging.Debugf("Parsed result: isJailbreak=%v, confidence=%.3f, categories=%v",
		isJailbreak, confidence, categories)

	// Write the distribution at whichever indices the configured
	// jailbreak_mapping assigns to the positive/negative labels. The argmax
	// class/confidence reported to callers is derived centrally from this
	// distribution (deriveArgmax), not decided here from isJailbreak.
	probabilities := make([]float32, 2)
	probabilities[v.positiveIdx] = 1 - confidence
	probabilities[v.negativeIdx] = confidence
	if isJailbreak {
		probabilities[v.positiveIdx] = confidence
		probabilities[v.negativeIdx] = 1 - confidence
	}

	return SequenceClassificationResult{Probabilities: probabilities}, nil
}
