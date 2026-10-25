package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// TestPrimaryOutputDigestIsTakenBeforeABodyWarning holds the order the buffered
// path has to keep.
//
// A hallucination or unverified-factual rule with the body action prepends
// router text to the same response the digest is read from, and it runs before
// the replay record is attached. Hashing after that credits the warning to the
// model, and a shadow arm, which no response-stage plugin touches, then reports
// a different digest for the same answer.
func TestPrimaryOutputDigestIsTakenBeforeABodyWarning(t *testing.T) {
	server, _ := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "body")
	recorder := startResponseStageReplay(t, router, ctx)
	ctx.TargetFormat = llmprotocol.OpenAIChatV1

	router.handleNonStreamingResponseBody(upstreamAnswerBody(hallucinationAnswer), ctx, 0)

	// Without this the test would pass on a response nothing rewrote.
	delivered := semanticResponseText(*ctx.SemanticResponse)
	if delivered == hallucinationAnswer {
		t.Fatal("no body warning was applied, so this proves nothing about the order")
	}
	if !strings.Contains(delivered, hallucinationAnswer) {
		t.Fatalf("the delivered answer lost the model text: %q", delivered)
	}

	sum := sha256.Sum256([]byte(hallucinationAnswer))
	want := hex.EncodeToString(sum[:])
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		t.Fatalf("replay record %q missing", ctx.RouterReplayID)
	}
	digests := 0
	for _, outcome := range record.Outcomes {
		if outcome.Source != primaryResponseOutcomeSource {
			continue
		}
		digests++
		if got := outcome.Metadata["response_sha256"]; got != want {
			t.Fatalf("primary digest = %q, want the hash of the model answer alone, %q", got, want)
		}
	}
	if digests != 1 {
		t.Fatalf("primary digest outcomes = %d, want exactly one", digests)
	}
}

// upstreamAnswerBody is a Chat Completions response carrying one assistant
// answer, the shape the backend returns before any plugin rewrites it.
func upstreamAnswerBody(answer string) []byte {
	return []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,` +
		`"message":{"id":"output_1","role":"assistant","content":"` + answer + `"},` +
		`"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,` +
		`"total_tokens":3}}`)
}
