package contextcompression

import (
	"encoding/json"
	"math"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// currentUserTextBlock never broadens tool, media, citation or instruction
// permissions. Only the final plain user message can opt into losing text.
func (request *RequestIR) currentUserTextBlock(message *MessageIR, block *TextBlockIR) bool {
	if message.Role != "user" || block.JSON || json.Valid([]byte(block.Text)) || block.Source != TargetHistory {
		return false
	}
	last := -1
	for _, candidate := range request.Messages {
		if request.messageStartsTurn(candidate) {
			last = candidate.Index
		}
	}
	if message.Index != last || message.Protection&(ProtectInstructions|ProtectAuthorization|ProtectSafety|ProtectUnknown) != 0 {
		return false
	}
	if request.Semantic == nil {
		_, plain := message.Raw["content"].(string)
		return plain
	}
	position := -1
	for index, candidate := range request.Messages {
		if candidate == message {
			position = index
			break
		}
	}
	if position < 0 || position >= len(request.Semantic.Messages) {
		return false
	}
	for _, content := range request.Semantic.Messages[position].Content {
		if content.Kind != llmprotocol.ContentText || len(content.Citations) > 0 || content.Signature != "" {
			return false
		}
	}
	return true
}

func candidateScore(kind TargetKind, query, text string) float64 {
	if kind == TargetCurrentUser {
		return math.MaxFloat64
	} // Prefer compressible history first.
	return lexicalScore(query, text)
}

func compressCandidateText(model string, counter TokenCounter, candidate plannedCandidate) Result {
	text, target := candidate.block.Text, candidate.plan.TargetTokens
	if candidate.plan.Kind == TargetCurrentUser {
		return truncateCurrentUser(model, counter, text, target)
	}
	// The extractive engine estimates tokens internally; adapt its item budget
	// to the request counter, then verify the result using that same counter.
	original, source := counter.CountText(model, text)
	if source != "utf8_byte_upper_bound" {
		return CompressToolOutput(text, candidate.plan.Query, candidate.plan.OriginalTokens, target)
	}
	targetEstimate := int(float64(target) * float64(EstimateTokens(text)) / float64(max(1, original)))
	return CompressToolOutput(text, candidate.plan.Query, EstimateTokens(text), max(1, targetEstimate))
}

func truncateCurrentUser(model string, counter TokenCounter, text string, target int) Result {
	original, _ := counter.CountText(model, text)
	if original <= target || !utf8.ValidString(text) {
		return unchangedResult(text, original)
	}
	runes := []rune(text)
	// Binary search the retained prefix/suffix together. Whole Unicode code
	// points survive, and every successful edit carries an explicit omission.
	lo, hi, best := 2, len(runes)-1, ""
	for lo <= hi {
		keep := lo + (hi-lo)/2
		head := (keep + 1) / 2
		candidate := string(runes[:head]) + omissionMarker + string(runes[len(runes)-(keep-head):])
		count, _ := counter.CountText(model, candidate)
		if count <= target {
			best = candidate
			lo = keep + 1
		} else {
			hi = keep - 1
		}
	}
	if best == "" {
		return unchangedResult(text, original)
	}
	after, _ := counter.CountText(model, best)
	return Result{Content: best, OriginalTokens: original, CompressedTokens: after, Applied: true, Format: "text", OmittedChunks: 1}
}
