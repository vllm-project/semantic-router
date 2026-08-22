package handlers

import (
	"encoding/json"
	"strings"
)

const estimatedImageTokens int64 = 2048

// estimateChatRequestTokens approximates the tokens the upstream tokenizer will
// see instead of treating every JSON byte as a token. The previous byte-count
// reservation rejected longer, valid conversations even when their live TPM
// meter had ample capacity. Actual usage still reconciles the shared counters
// after every response.
func estimateChatRequestTokens(messages, tools json.RawMessage, maxOutput int64) int64 {
	input := estimateRawJSONTokens(messages) + estimateRawJSONTokens(tools)
	return max(input+max(maxOutput, 0), 1)
}

func estimateRawJSONTokens(raw json.RawMessage) int64 {
	if len(raw) == 0 || string(raw) == "null" {
		return 0
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return estimateTextTokens(string(raw))
	}
	return estimateJSONTokens(value, "")
}

func estimateJSONTokens(value any, field string) int64 {
	switch typed := value.(type) {
	case string:
		if field == "image_url" || strings.HasPrefix(typed, "data:image/") {
			return estimatedImageTokens
		}
		return estimateTextTokens(typed)
	case []any:
		var total int64 = 2
		for _, item := range typed {
			total += estimateJSONTokens(item, field) + 1
		}
		return total
	case map[string]any:
		if field == "image_url" {
			return estimatedImageTokens
		}
		var total int64 = 2
		for key, item := range typed {
			total += estimateTextTokens(key) + estimateJSONTokens(item, key) + 1
		}
		return total
	case nil:
		return 0
	default:
		return 1
	}
}

func estimateTextTokens(value string) int64 {
	if value == "" {
		return 0
	}
	var asciiBytes, nonASCII int64
	for _, char := range value {
		if char <= 0x7f {
			asciiBytes++
		} else {
			nonASCII++
		}
	}
	// Three ASCII bytes per token remains conservative for prose and serialized
	// tool schemas. One token per non-ASCII rune protects multilingual prompts.
	return max((asciiBytes+2)/3+nonASCII, 1)
}
