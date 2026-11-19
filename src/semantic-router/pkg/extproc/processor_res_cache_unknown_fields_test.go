package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// azureDecoratedChatResponse is the shape Azure OpenAI / AI Foundry returns: a
// canonical chat completion plus members the Chat contract does not declare.
// Reported in #3496.
const azureDecoratedChatResponse = `{"id":"chatcmpl-1","object":"chat.completion","created":1,` +
	`"model":"gpt-4.1-mini",` +
	`"prompt_filter_results":[{"prompt_index":0,"content_filter_results":{}}],` +
	`"choices":[{"index":0,"finish_reason":"stop",` +
	`"message":{"role":"assistant","content":"hi"},` +
	`"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}}}],` +
	`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,` +
	`"latency_checkpoint":{"time_to_first_token_ms":12}}}`

func azureResponseContext() *RequestContext {
	return &RequestContext{
		RequestID:    "req-1",
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
	}
}

// The miss-to-hit path: a same-format response is forwarded verbatim, but the
// bytes persisted on a cache miss are the canonical encode, so one request's
// upstream metadata is never replayed to later clients.
func TestCacheableClientResponseIsReadableByTheCacheReader(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()

	// Miss: decode the decorated upstream response.
	semanticResponse, err := router.decodeClientResponse([]byte(azureDecoratedChatResponse), ctx)
	if err != nil {
		t.Fatalf("decodeClientResponse() error = %v, want nil", err)
	}
	if !ctx.ResponseUnknownFields {
		t.Fatal("ResponseUnknownFields = false, want true for a decorated response")
	}

	// Write: a same-format response is forwarded verbatim, so the cache must
	// receive the canonical re-encode instead of those bytes.
	cached := router.cacheableClientResponse([]byte(azureDecoratedChatResponse), false, *semanticResponse, ctx)
	if len(cached) == 0 {
		t.Fatal("cacheableClientResponse() returned no body, want the canonical encode")
	}
	for _, decoration := range []string{"prompt_filter_results", "content_filter_results", "latency_checkpoint"} {
		if strings.Contains(string(cached), decoration) {
			t.Errorf("cached body still contains the decoration %q", decoration)
		}
	}

	// Hit: the cache reader decodes the canonical entry.
	hitCtx := azureResponseContext()
	if _, err := router.decodeCachedClientResponse(cached, hitCtx); err != nil {
		t.Fatalf("decodeCachedClientResponse() error = %v, want nil on the cached body", err)
	}
}

// An entry that still holds raw upstream bytes, written before the canonical
// write existed, is served instead of failing the hit.
func TestCacheReaderToleratesRawDecoratedUpstreamBody(t *testing.T) {
	router := &OpenAIRouter{}
	hitCtx := azureResponseContext()

	if _, err := router.decodeCachedClientResponse([]byte(azureDecoratedChatResponse), hitCtx); err != nil {
		t.Fatalf("decodeCachedClientResponse() error = %v on a decorated body, want nil", err)
	}
}

// A response with no decorations must be cached byte-for-byte, so the common
// path keeps its existing passthrough fidelity and costs no re-encode.
func TestCacheableClientResponseLeavesUndecoratedBodyUntouched(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)

	semanticResponse, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatalf("decodeClientResponse() error = %v, want nil", err)
	}
	if ctx.ResponseUnknownFields {
		t.Fatal("ResponseUnknownFields = true, want false for a canonical response")
	}

	cached := router.cacheableClientResponse(body, false, *semanticResponse, ctx)
	if string(cached) != string(body) {
		t.Errorf("cached body = %s, want the original bytes unchanged", cached)
	}
}

// A cross-format response is already re-encoded before the cache write, so the
// canonical write must not encode it a second time.
func TestCacheableClientResponseKeepsAlreadyRewrittenBody(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()
	ctx.ResponseUnknownFields = true
	rewritten := []byte(`{"object":"response"}`)

	cached := router.cacheableClientResponse(rewritten, true, llmprotocol.Response{}, ctx)
	if string(cached) != string(rewritten) {
		t.Errorf("cached body = %s, want the already-rewritten body", cached)
	}
}

// A rewrite that only replayed the envelope's bytes, as reported in #6957,
// carries the same decorations a verbatim forward would and must not be
// trusted as canonical just because an encode call ran.
func TestCacheableClientResponseCanonicalizesAReplayedRewrite(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()
	ctx.ResponseUnknownFields = true
	ctx.ResponseClientBodyReplayed = true
	replayed := []byte(azureDecoratedChatResponse)

	cached := router.cacheableClientResponse(replayed, true, llmprotocol.Response{}, ctx)
	for _, decoration := range []string{"prompt_filter_results", "content_filter_results", "latency_checkpoint"} {
		if strings.Contains(string(cached), decoration) {
			t.Errorf("cached body still contains the decoration %q after a replayed rewrite", decoration)
		}
	}
}

// decodeClientResponse and encodeClientResponse are the real pipeline
// functions a request runs. The client gets the trimmed reply with its vendor
// extension intact, and the cache gets the canonical form instead of a replay
// that still carries it, closing the gap reported on #6957.
func TestCacheableClientResponseCanonicalizesAnAnswerSeparatorReplay(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m",` +
		`"vendor_extension":{"trace_id":"trace-1"},` +
		`"choices":[{"index":0,"finish_reason":"stop",` +
		`"message":{"role":"assistant","content":"\n\nRed","reasoning":"thinking it through"}}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)

	semanticResponse, err := router.decodeClientResponse(body, ctx)
	if err != nil {
		t.Fatalf("decodeClientResponse() error = %v, want nil", err)
	}
	if !ctx.ResponseUnknownFields {
		t.Fatal("ResponseUnknownFields = false, want true for the vendor extension")
	}

	clientBody, err := router.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		t.Fatalf("encodeClientResponse() error = %v, want nil", err)
	}
	if !ctx.ResponseClientBodyReplayed {
		t.Fatal("ResponseClientBodyReplayed = false, want true: the trim patches the envelope in place")
	}
	if !strings.Contains(string(clientBody), "vendor_extension") {
		t.Fatalf("client body = %s, want the vendor extension forwarded to the client", clientBody)
	}
	if strings.Contains(string(clientBody), "\n\nRed") {
		t.Fatalf("client body = %s, want the separator trimmed", clientBody)
	}

	cached := router.cacheableClientResponse(clientBody, true, *semanticResponse, ctx)
	if strings.Contains(string(cached), "vendor_extension") {
		t.Errorf("cached body = %s, still contains the vendor extension after the replayed trim", cached)
	}
}
