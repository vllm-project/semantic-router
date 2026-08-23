package contextcompression

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRequestTokenCountersUseNeutralSemanticRequest(t *testing.T) {
	request := &RequestIR{Semantic: &llmprotocol.Request{
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "a deliberately non-empty semantic request",
			}},
		}},
	}}

	heuristic, source := (HeuristicTokenCounter{}).CountRequest("model", request)
	if heuristic <= EstimateTokens("null") || source != "heuristic" {
		t.Fatalf("heuristic count = %d source = %q", heuristic, source)
	}

	var estimatedBytes int
	calibrated, source := (CalibratedTokenCounter{Estimate: func(_ string, byteLength int) (int, bool) {
		estimatedBytes = byteLength
		return byteLength, true
	}}).CountRequest("model", request)
	if calibrated <= len("null") || calibrated != estimatedBytes || source != "provider_calibrated" {
		t.Fatalf("calibrated count = %d bytes = %d source = %q", calibrated, estimatedBytes, source)
	}
}
