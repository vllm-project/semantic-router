package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

func TestRequestDemandSnapshotSeparatesPromptAndOutputReserve(t *testing.T) {
	request := testNeutralRequest("vllm-sr/auto", "one two three four")
	request.Generation = 7
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	promptTokens := extractSemanticRequestSignals(request).ContextTokenFloor

	snapshot := requestDemandSnapshot(
		requestDemandStageOriginal,
		request,
		"",
		promptTokens,
		requestDemandCountingSourceFallback,
	)

	if snapshot.Stage != requestDemandStageOriginal || snapshot.Model != "vllm-sr/auto" ||
		snapshot.Representation != requestDemandRepresentationSemantic {
		t.Fatalf("unexpected snapshot identity: %+v", snapshot)
	}
	if snapshot.PromptTokens != promptTokens || snapshot.ReservedOutputTokens != 256 ||
		snapshot.TotalDemandTokens != promptTokens+256 || !snapshot.TotalDemandKnown {
		t.Fatalf("unexpected request demand: %+v", snapshot)
	}
	if snapshot.CountingSource != requestDemandCountingSourceFallback ||
		snapshot.OutputReserveSource != requestDemandOutputReserveExplicit ||
		snapshot.RequestGeneration != 7 {
		t.Fatalf("unexpected snapshot provenance: %+v", snapshot)
	}
}

func TestRequestDemandSnapshotMarksMissingOutputReserveUnknown(t *testing.T) {
	request := testNeutralRequest("vllm-sr/auto", "hello")
	snapshot := requestDemandSnapshot(
		requestDemandStageOriginal,
		request,
		"",
		extractSemanticRequestSignals(request).ContextTokenFloor,
		requestDemandCountingSourceFallback,
	)

	if snapshot.OutputReserveSource != requestDemandOutputReserveUnknown || snapshot.TotalDemandKnown ||
		snapshot.ReservedOutputTokens != 0 || snapshot.TotalDemandTokens != snapshot.PromptTokens {
		t.Fatalf("missing output reserve was not explicit: %+v", snapshot)
	}
}

func TestUpsertRequestDemandSnapshotIsBoundedAndStageOrdered(t *testing.T) {
	var snapshots []routerreplay.RequestDemandSnapshot
	for _, stage := range []string{
		requestDemandStageProviderBound,
		requestDemandStageOriginal,
		requestDemandStagePostToolPolicy,
		requestDemandStagePostContext,
	} {
		snapshots = upsertRequestDemandSnapshot(snapshots, routerreplay.RequestDemandSnapshot{
			Stage: stage, PromptTokens: 1,
		})
	}
	snapshots = upsertRequestDemandSnapshot(snapshots, routerreplay.RequestDemandSnapshot{
		Stage: requestDemandStagePostContext, PromptTokens: 9,
	})
	snapshots = upsertRequestDemandSnapshot(snapshots, routerreplay.RequestDemandSnapshot{
		Stage: "future_unbounded_stage", PromptTokens: 99,
	})

	wantStages := []string{
		requestDemandStageOriginal,
		requestDemandStagePostContext,
		requestDemandStagePostToolPolicy,
		requestDemandStageProviderBound,
	}
	if len(snapshots) != len(wantStages) {
		t.Fatalf("snapshot count = %d, want %d: %+v", len(snapshots), len(wantStages), snapshots)
	}
	for index, want := range wantStages {
		if snapshots[index].Stage != want {
			t.Fatalf("snapshot[%d].Stage = %q, want %q", index, snapshots[index].Stage, want)
		}
	}
	if snapshots[1].PromptTokens != 9 {
		t.Fatalf("stage upsert did not replace the prior observation: %+v", snapshots[1])
	}
}

func TestConcreteModelDispatchCapturesAllRequestDemandStages(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Cache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})
	body, err := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]interface{}{{
			"role": "user", "content": "stage-aware demand",
		}},
		"max_tokens": 128,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "demand-stages"}

	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody: %v", err)
	}
	if response.GetRequestBody() == nil {
		t.Fatalf("expected provider dispatch response, got %#v", response)
	}

	wantStages := []string{
		requestDemandStageOriginal,
		requestDemandStagePostContext,
		requestDemandStagePostToolPolicy,
		requestDemandStageProviderBound,
	}
	if len(ctx.RequestDemandSnapshots) != len(wantStages) {
		t.Fatalf("request demand snapshots = %+v, want stages %v", ctx.RequestDemandSnapshots, wantStages)
	}
	for index, want := range wantStages {
		snapshot := ctx.RequestDemandSnapshots[index]
		if snapshot.Stage != want {
			t.Fatalf("snapshot[%d].Stage = %q, want %q", index, snapshot.Stage, want)
		}
		if snapshot.CountingSource != string(contextCountProxy) ||
			snapshot.Representation != requestDemandRepresentationSemantic ||
			snapshot.ReservedOutputTokens != 128 ||
			snapshot.OutputReserveSource != requestDemandOutputReserveExplicit {
			t.Fatalf("snapshot[%d] lost demand provenance: %+v", index, snapshot)
		}
	}
}

func TestReplayRouteDiagnosticsCopyRequestDemandSnapshots(t *testing.T) {
	ctx := &RequestContext{RequestDemandSnapshots: []routerreplay.RequestDemandSnapshot{{
		Stage: requestDemandStageOriginal, PromptTokens: 12, CountingSource: requestDemandCountingSourceFallback,
	}}}
	record := buildReplayRoutingRecord(ctx, "entrypoint", "model-a", "route")

	if record.RouteDiagnostics == nil || len(record.RouteDiagnostics.RequestDemandSnapshots) != 1 {
		t.Fatalf("Replay demand snapshots missing: %+v", record.RouteDiagnostics)
	}
	ctx.RequestDemandSnapshots[0].PromptTokens = 99
	if record.RouteDiagnostics.RequestDemandSnapshots[0].PromptTokens != 12 {
		t.Fatalf("Replay demand snapshot aliases request context: %+v", record.RouteDiagnostics.RequestDemandSnapshots)
	}
}

// Each stage counts with the tokenizer the window check uses for its model:
// the proxy for a model that names none, the model's own otherwise.
func TestRequestDemandCountsWithTheStageModelsTokenizer(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"named": {Tokenizer: tokencount.Cl100kBase},
		}},
	}}
	request := testNeutralRequest("vllm-sr/auto", "Count these words with each model's own tokenizer, please.")
	ctx := &RequestContext{SemanticRequest: request}

	router.captureOriginalRequestDemand(ctx, request, extractSemanticRequestSignals(request))
	router.captureRequestDemand(ctx, requestDemandStageProviderBound, request, "named")

	o200k, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	cl100k, err := tokencount.Builtin(tokencount.Cl100kBase)
	require.NoError(t, err)
	require.Len(t, ctx.RequestDemandSnapshots, 2)
	original, provider := ctx.RequestDemandSnapshots[0], ctx.RequestDemandSnapshots[1]
	require.Equal(t, string(contextCountProxy), original.CountingSource)
	require.Equal(t, countPromptTokens(o200k, request), original.PromptTokens)
	require.Equal(t, string(contextCountExact), provider.CountingSource)
	require.Equal(t, countPromptTokens(cl100k, request), provider.PromptTokens)
}

// A stage reuses the count of an unchanged request, the window check's
// included, and a changed request is counted once more, for both.
func TestRequestDemandSharesCountsWithTheWindowCheck(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := testNeutralRequest("model", "share one count per request generation")
	ctx := &RequestContext{SemanticRequest: request, contextCounts: map[contextCountKey]contextCount{
		{tokenizer: tokencount.O200kBase, generation: request.Generation}: {tokens: 4242, ok: true},
	}}

	router.captureRequestDemand(ctx, requestDemandStagePostContext, request, "model")
	require.Equal(t, 4242, ctx.RequestDemandSnapshots[0].PromptTokens, "an unchanged request was counted again")

	request.Generation++
	router.captureRequestDemand(ctx, requestDemandStageProviderBound, request, "model")
	o200k, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	want := countPromptTokens(o200k, request)
	require.Equal(t, want, ctx.RequestDemandSnapshots[1].PromptTokens)
	tokens, kind, ok := router.cachedContextTokens(ctx, nil, "model")
	require.True(t, ok, "the window check does not see the stage's count")
	require.Equal(t, contextCountProxy, kind)
	require.Equal(t, want, tokens)
}

// No text tokenizer counts media, so a request that carries it keeps the
// provider-neutral fallback, taken from its signals.
func TestRequestDemandFallsBackForMedia(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	request := testNeutralRequest("model", "describe")
	request.Messages[0].Content = append(request.Messages[0].Content,
		llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/image"})
	ctx := &RequestContext{SemanticRequest: request}
	facts := extractSemanticRequestSignals(request)

	router.captureOriginalRequestDemand(ctx, request, facts)
	router.captureRequestDemand(ctx, requestDemandStageProviderBound, request, "model")

	require.Len(t, ctx.RequestDemandSnapshots, 2)
	for _, snapshot := range ctx.RequestDemandSnapshots {
		require.Equal(t, requestDemandCountingSourceFallback, snapshot.CountingSource)
		require.Equal(t, facts.ContextTokenFloor, snapshot.PromptTokens)
	}
}
