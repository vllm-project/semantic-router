package extproc

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const streamedImage = "ZmluYWwtaW1hZ2U="

// imageGenerationStream is the protocol codec's golden Responses stream of one
// generated image, with partial images before it, one SSE event per part.
func imageGenerationStream(t *testing.T) [][]byte {
	t.Helper()
	raw, err := os.ReadFile("../protocolcodec/testdata/golden/capability/049-responses-image-generation-stream-in.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Stream struct {
			Chunks []string `json:"chunks"`
		} `json:"stream"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	parts := make([][]byte, 0, len(fixture.Stream.Chunks))
	for _, chunk := range fixture.Stream.Chunks {
		parts = append(parts, []byte(chunk))
	}
	return parts
}

// A streamed answer that is a generated image is rebuilt with the image, so it
// reaches the Responses store, Router Replay, the cache and memory like text.
func TestStreamedGeneratedImageIsRebuilt(t *testing.T) {
	store := NewMockResponseStore()
	router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(store)}
	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: t.Context()}
	streamingClient(ctx, llmprotocol.OpenAIResponsesV1)
	state, err := router.ResponseAPIFilter.PrepareObjectState(t.Context(), *ctx.SemanticRequest, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx.ResponseObjectState = state
	parts := imageGenerationStream(t)
	for i, part := range parts {
		router.handleSemanticStreamingResponseBody(part, i == len(parts)-1, ctx)
	}

	if ctx.StreamingAborted || ctx.SemanticResponse == nil {
		t.Fatalf("aborted = %v, response = %+v; want the image stream rebuilt", ctx.StreamingAborted, ctx.SemanticResponse)
	}
	output := ctx.SemanticResponse.Output
	if len(output) != 1 || len(output[0].Content) != 1 || output[0].ID != "ig_stream_1" {
		t.Fatalf("rebuilt output = %+v, want the one image item", output)
	}
	image := output[0].Content[0].GeneratedImage
	if image == nil || image.Status != llmprotocol.ImageGenerationCompleted || image.Result == nil || *image.Result != streamedImage ||
		image.PartialImage != "" {
		t.Fatalf("rebuilt image = %+v, want the completed image without partial progress", image)
	}
	encoded, err := router.encodeClientResponse(*ctx.SemanticResponse, ctx)
	if err != nil || !strings.Contains(string(encoded), `"result":"`+streamedImage+`"`) {
		t.Fatalf("rebuilt answer for the client = %s, %v; want the image", encoded, err)
	}
	if len(store.responses) != 1 {
		t.Fatalf("stored Responses objects = %d, want the streamed image's", len(store.responses))
	}
}

// A streamed generated image is not cached: a cache hit streams the entry
// back, and the stream cannot carry an image, so every hit would fail.
func TestStreamedGeneratedImageIsNotCached(t *testing.T) {
	responseCache, router, decision := cacheRouterForDecision(config.Decision{
		Name:      "image-cache-decision",
		ModelRefs: []config.ModelRef{{Model: "model-a"}},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    config.ResponseCacheModeExactThenSemantic,
			}),
		}},
	})
	ctx := withSelectedDecision(&RequestContext{
		RequestID:             "req-image-cache",
		CacheExactFingerprint: "fingerprint",
		CacheSemanticSafe:     true,
		CacheQuery:            "draw a cat",
		TraceContext:          t.Context(),
	}, decision)
	streamingClient(ctx, llmprotocol.OpenAIResponsesV1)
	skipped := metrics.CacheWriteSkipped.WithLabelValues("stream_content_unsupported")
	before := testutil.ToFloat64(skipped)
	parts := imageGenerationStream(t)
	for i, part := range parts {
		router.handleSemanticStreamingResponseBody(part, i == len(parts)-1, ctx)
	}

	if ctx.StreamingAborted || ctx.SemanticResponse == nil {
		t.Fatalf("aborted = %v, response = %+v; want the image stream rebuilt", ctx.StreamingAborted, ctx.SemanticResponse)
	}
	if responseCache.addEntryCalled || responseCache.exactAdded {
		t.Fatalf("semantic entry = %v, exact entry = %v; want no cache entry for the image", responseCache.addEntryCalled, responseCache.exactAdded)
	}
	if got := testutil.ToFloat64(skipped); got != before+1 {
		t.Fatalf("cache writes skipped as stream_content_unsupported = %v, want %v", got, before+1)
	}
}

// A held stream whose answer is a generated image is a complete answer: the
// guard reads it, and the client receives the image as the stream the Router
// streams without a guard.
func TestHeldStreamReleasesAGeneratedImage(t *testing.T) {
	router, ctx := guardedStreamingRouter(t, 0.01, config.ResponseGuardBlock, llmprotocol.OpenAIResponsesV1)
	before := bufferedCount(ctx, responseGuardIncomplete)
	parts := imageGenerationStream(t)

	answerHeaders(t, router, ctx, "text/event-stream")
	client, _ := answerBody(t, router, ctx, whole(parts))

	if want := streamedThroughUnguarded(t, llmprotocol.OpenAIResponsesV1, parts); client != want || !strings.Contains(client, streamedImage) {
		t.Fatalf("client stream =\n%s\nwant the image stream as it streams unguarded:\n%s", client, want)
	}
	if got := bufferedCount(ctx, responseGuardIncomplete); got != before || ctx.ResponseGuard == nil {
		t.Fatalf("llm_response_guard_buffered_total{outcome=incomplete} = %v, want %v and the guard to read the image answer", got, before)
	}
}
