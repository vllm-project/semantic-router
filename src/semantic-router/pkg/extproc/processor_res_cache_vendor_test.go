package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// azureDecoratedChatResponse is the shape Azure OpenAI / AI Foundry returns: a
// canonical chat completion plus Azure's own decorations. Reported in #3496.
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
		RequestID:      "req-1",
		SourceFormat:   llmprotocol.OpenAIChatV1,
		TargetFormat:   llmprotocol.OpenAIChatV1,
		ResponseVendor: llmprotocol.ResponseVendorAzure,
	}
}

// The miss-to-hit path: the bytes persisted on a cache miss must decode under
// the strict contract the cache reader uses. Before this, a same-format Azure
// response was forwarded verbatim and cached verbatim, so the first success
// poisoned its own partition and every later hit returned 502.
func TestCacheableClientResponseIsReadableByTheCacheReader(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()

	// Miss: decode the decorated upstream response.
	semanticResponse, err := router.decodeClientResponse([]byte(azureDecoratedChatResponse), ctx)
	if err != nil {
		t.Fatalf("decodeClientResponse() error = %v, want nil", err)
	}
	if !ctx.ResponseVendorExtensions {
		t.Fatal("ResponseVendorExtensions = false, want true for a decorated response")
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

	// Hit: the cache reader decodes without any backend vendor allowance.
	hitCtx := azureResponseContext()
	hitCtx.ResponseVendor = ""
	if _, err := router.decodeCachedClientResponse(cached, hitCtx); err != nil {
		t.Fatalf("decodeCachedClientResponse() error = %v, want nil on the cached body", err)
	}
}

// The regression itself: the raw upstream bytes are what used to be cached, and
// the cache reader cannot decode them.
func TestCacheReaderRejectsRawDecoratedUpstreamBody(t *testing.T) {
	router := &OpenAIRouter{}
	hitCtx := azureResponseContext()
	hitCtx.ResponseVendor = ""

	_, err := router.decodeCachedClientResponse([]byte(azureDecoratedChatResponse), hitCtx)
	if err == nil {
		t.Fatal("decodeCachedClientResponse() error = nil on a decorated body, want rejection")
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
	if ctx.ResponseVendorExtensions {
		t.Fatal("ResponseVendorExtensions = true, want false for a canonical response")
	}

	cached := router.cacheableClientResponse(body, false, *semanticResponse, ctx)
	if string(cached) != string(body) {
		t.Errorf("cached body = %s, want the original bytes unchanged", cached)
	}
}

// A cross-format response is already re-encoded before the cache write, so the
// vendor path must not encode it a second time.
func TestCacheableClientResponseKeepsAlreadyRewrittenBody(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := azureResponseContext()
	ctx.ResponseVendorExtensions = true
	rewritten := []byte(`{"object":"response"}`)

	cached := router.cacheableClientResponse(rewritten, true, llmprotocol.Response{}, ctx)
	if string(cached) != string(rewritten) {
		t.Errorf("cached body = %s, want the already-rewritten body", cached)
	}
}
