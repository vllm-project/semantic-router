package looper

import (
	"math"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// remomDefaultBytesPerToken is the bytes-per-token ratio assumed when the
// backend did not report completion usage. It matches English text; CJK and
// other multi-byte scripts pack fewer bytes per token, which is why an
// observed ratio is preferred whenever one is available.
const remomDefaultBytesPerToken = 4.0

// reMoMBytesPerToken returns the bytes-per-token ratio observed for a
// response: the bytes the backend generated divided by the completion tokens
// it billed for them. Completion usage covers content and reasoning together,
// so both are counted. Falls back to remomDefaultBytesPerToken when usage is
// missing, as it is for streaming responses without include_usage.
func reMoMBytesPerToken(resp *ModelResponse) float64 {
	if resp == nil || resp.Usage.CompletionTokens <= 0 {
		return remomDefaultBytesPerToken
	}
	generatedBytes := len(resp.Content) + len(resp.ReasoningContent)
	if generatedBytes == 0 {
		return remomDefaultBytesPerToken
	}
	return float64(generatedBytes) / float64(resp.Usage.CompletionTokens)
}

// compactResponse compacts a response based on strategy. bytesPerToken
// converts the configured token budget into bytes; see reMoMBytesPerToken.
func (l *ReMoMLooper) compactResponse(cfg *config.ReMoMAlgorithmConfig, content string, bytesPerToken float64) string {
	strategy := cfg.CompactionStrategy
	if strategy == "" {
		strategy = "full"
	}

	switch strategy {
	case "last_n_tokens":
		maxTokens := cfg.CompactionTokens
		if maxTokens <= 0 {
			maxTokens = 1000
		}
		return lastNTokens(content, maxTokens, bytesPerToken)
	case "full":
		fallthrough
	default:
		return content
	}
}

// lastNTokens keeps the tail of content that fits in maxTokens at the given
// bytes-per-token ratio. The cut is advanced to the next rune start so the
// result is always valid UTF-8 and never opens with a partial character.
func lastNTokens(content string, maxTokens int, bytesPerToken float64) string {
	if bytesPerToken <= 0 {
		bytesPerToken = remomDefaultBytesPerToken
	}
	maxBytes := int(math.Ceil(float64(maxTokens) * bytesPerToken))
	if maxBytes < 1 {
		maxBytes = 1
	}
	if len(content) <= maxBytes {
		return content
	}
	start := len(content) - maxBytes
	for start < len(content) && !utf8.RuneStart(content[start]) {
		start++
	}
	return content[start:]
}

// estimateTokens estimates the token count of text at the given
// bytes-per-token ratio; see reMoMBytesPerToken.
func estimateTokens(text string, bytesPerToken float64) int {
	if bytesPerToken <= 0 {
		bytesPerToken = remomDefaultBytesPerToken
	}
	return int(math.Round(float64(len(text)) / bytesPerToken))
}
