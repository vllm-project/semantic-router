package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// =====================================================================
// NON-STREAMING: parseResponseUsage
// =====================================================================

func TestParseResponseUsage_ValidJSON(t *testing.T) {
	body := buildChatCompletionWithUsage(10, 20)
	usage := parseResponseUsage(body, "test-model")

	assert.Equal(t, 10, usage.promptTokens)
	assert.Equal(t, 20, usage.completionTokens)
}

func TestParseResponseUsage_InvalidJSON(t *testing.T) {
	usage := parseResponseUsage([]byte(`{not valid json`), "test-model")

	assert.Equal(t, 0, usage.promptTokens)
	assert.Equal(t, 0, usage.completionTokens)
}

func TestParseResponseUsage_EmptyBody(t *testing.T) {
	usage := parseResponseUsage([]byte{}, "test-model")

	assert.Equal(t, 0, usage.promptTokens)
	assert.Equal(t, 0, usage.completionTokens)
}

func TestParseResponseUsage_ExtractsUsageFields(t *testing.T) {
	usage := parseResponseUsage([]byte(`{
		"usage": {
			"prompt_tokens": 11,
			"completion_tokens": 7
		}
	}`), "test-model")

	assert.Equal(t, responseUsageMetrics{
		promptTokens:     11,
		completionTokens: 7,
	}, usage)
}

func TestParseResponseUsage_TracksCachedTokenReporting(t *testing.T) {
	usage := parseResponseUsage([]byte(`{
		"usage": {
			"prompt_tokens": 100,
			"completion_tokens": 7,
			"prompt_tokens_details": {
				"cached_tokens": 40
			}
		}
	}`), "test-model")

	assert.Equal(t, responseUsageMetrics{
		promptTokens:               100,
		cachedPromptTokens:         40,
		cachedPromptTokensReported: true,
		completionTokens:           7,
	}, usage)
}

func TestParseResponseUsage_ReturnsZeroForInvalidUsageTypes(t *testing.T) {
	usage := parseResponseUsage([]byte(`{
		"usage": {
			"prompt_tokens": "11",
			"completion_tokens": 7
		}
	}`), "test-model")

	assert.Equal(t, responseUsageMetrics{}, usage)
}

func TestParseResponseUsage_ZeroTokens(t *testing.T) {
	body := buildChatCompletionWithUsage(0, 0)
	usage := parseResponseUsage(body, "test-model")

	assert.Equal(t, 0, usage.promptTokens)
	assert.Equal(t, 0, usage.completionTokens)
}

// =====================================================================
// STREAMING: extractStreamingUsage
// =====================================================================

func TestExtractStreamingUsage_WithAllFields(t *testing.T) {
	ctx := &RequestContext{
		StreamingMetadata: map[string]interface{}{
			"usage": map[string]interface{}{
				"prompt_tokens":     float64(15),
				"completion_tokens": float64(25),
				"total_tokens":      float64(40),
			},
		},
	}

	usage := extractStreamingUsage(ctx)

	assert.Equal(t, int64(15), usage.PromptTokens)
	assert.Equal(t, int64(25), usage.CompletionTokens)
	assert.Equal(t, int64(40), usage.TotalTokens)
}

func TestExtractStreamingUsage_NoUsageInMetadata(t *testing.T) {
	ctx := &RequestContext{
		StreamingMetadata: map[string]interface{}{
			"id":    "chatcmpl-123",
			"model": "test-model",
		},
	}

	usage := extractStreamingUsage(ctx)

	assert.Equal(t, int64(0), usage.PromptTokens)
	assert.Equal(t, int64(0), usage.CompletionTokens)
	assert.Equal(t, int64(0), usage.TotalTokens)
}

func TestExtractStreamingUsage_PartialFields(t *testing.T) {
	ctx := &RequestContext{
		StreamingMetadata: map[string]interface{}{
			"usage": map[string]interface{}{
				"prompt_tokens": float64(10),
			},
		},
	}

	usage := extractStreamingUsage(ctx)

	assert.Equal(t, int64(10), usage.PromptTokens)
	assert.Equal(t, int64(0), usage.CompletionTokens, "missing fields default to 0")
	assert.Equal(t, int64(0), usage.TotalTokens, "missing fields default to 0")
}

func TestCalibrateTokenEstimatorUsesContextTextBytes(t *testing.T) {
	classifier, err := classification.BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "long_context",
					MinTokens: config.TokenCount("0"),
					MaxTokens: config.TokenCount("10K"),
				}},
			},
		},
	}, nil, nil, nil)
	assert.NoError(t, err)

	router := &OpenAIRouter{Classifier: classifier}
	ctx := &RequestContext{
		OriginalRequestBody:     []byte(`{"messages":[{"role":"user","content":"short"}]}`),
		VSRContextTextBytes:     2000,
		VSRMatchedContext:       []string{"long_context"},
		VSRSelectedDecisionName: "fallback_decision",
	}

	for i := 0; i < 20; i++ {
		router.calibrateTokenEstimator(ctx, 1000)
	}

	defaultMean, _, _, defaultCalibrated := classifier.TokenCalibrationRatio("")
	assert.True(t, defaultCalibrated)
	assert.InDelta(t, 2.0, defaultMean, 0.1)

	categoryMean, _, _, categoryCalibrated := classifier.TokenCalibrationRatio("long_context")
	assert.True(t, categoryCalibrated)
	assert.InDelta(t, 2.0, categoryMean, 0.1)
}

// =====================================================================
// Helpers
// =====================================================================

func buildChatCompletionWithUsage(promptTokens, completionTokens int) []byte {
	body := map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1234567890,
		"model":   "test-model",
		"choices": []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "Hello",
			},
			"finish_reason": "stop",
		}},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	b, _ := json.Marshal(body)
	return b
}
