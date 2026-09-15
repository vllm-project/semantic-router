package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRequestDemandSnapshotSeparatesPromptAndOutputReserve(t *testing.T) {
	request := testNeutralRequest("auto", "one two three four")
	request.Generation = 7
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	promptTokens := extractSemanticRequestSignals(request).ContextTokenFloor

	snapshot := requestDemandSnapshot(
		requestDemandStageOriginal,
		request,
		"",
		promptTokens,
	)

	if snapshot.Stage != requestDemandStageOriginal || snapshot.Model != "auto" ||
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
	request := testNeutralRequest("auto", "hello")
	snapshot := requestDemandSnapshot(
		requestDemandStageOriginal,
		request,
		"",
		extractSemanticRequestSignals(request).ContextTokenFloor,
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
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router.ReplayRecorder = recorder
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
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
	ctx := &RequestContext{
		Headers:                  map[string]string{},
		RequestID:                "demand-stages",
		RouterReplayPluginConfig: &replayConfig,
	}

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
		if snapshot.CountingSource != requestDemandCountingSourceFallback ||
			snapshot.Representation != requestDemandRepresentationSemantic ||
			snapshot.ReservedOutputTokens != 128 ||
			snapshot.OutputReserveSource != requestDemandOutputReserveExplicit {
			t.Fatalf("snapshot[%d] lost demand provenance: %+v", index, snapshot)
		}
	}

	if ctx.RouterReplayID == "" {
		t.Fatal("request demand flow did not start Router Replay")
	}
	startedRecord, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found || startedRecord.RouteDiagnostics == nil {
		t.Fatalf("Replay record %q missing route diagnostics: %+v", ctx.RouterReplayID, startedRecord)
	}
	if got := len(startedRecord.RouteDiagnostics.RequestDemandSnapshots); got != len(wantStages)-1 {
		t.Fatalf("in-progress Replay snapshots = %d, want %d before terminal enrichment", got, len(wantStages)-1)
	}

	router.finalizeRouterReplay(ctx, routerreplay.LifecycleCompleted, "test_complete")
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found || record.RouteDiagnostics == nil {
		t.Fatalf("final Replay record %q missing route diagnostics: %+v", ctx.RouterReplayID, record)
	}
	persisted := record.RouteDiagnostics.RequestDemandSnapshots
	if len(persisted) != len(wantStages) {
		t.Fatalf("Replay snapshots = %+v, want stages %v", persisted, wantStages)
	}
	for index, want := range wantStages {
		if persisted[index].Stage != want {
			t.Fatalf("Replay snapshot[%d].Stage = %q, want %q", index, persisted[index].Stage, want)
		}
	}
}

func TestProviderBoundReplayDemandUpdateFailsOpen(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Cache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})
	router.ReplayRecorder = routerreplay.NewRecorder(&failingRequestDemandStore{
		Storage: store.NewMemoryStore(10, 0),
	})
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	body := []byte(`{"model":"` + model + `","messages":[{"role":"user","content":"fail open"}],"max_tokens":32}`)
	ctx := &RequestContext{
		Headers:                  map[string]string{},
		RequestID:                "demand-update-fail-open",
		RouterReplayPluginConfig: &replayConfig,
	}

	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body},
	}, ctx)
	if err != nil || response.GetRequestBody() == nil {
		t.Fatalf("Replay enrichment changed dispatch: response=%#v err=%v", response, err)
	}

	router.finalizeRouterReplay(ctx, routerreplay.LifecycleCompleted, "test_complete")
	record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !found || record.LifecycleState != routerreplay.LifecycleCompleted {
		t.Fatalf("Replay enrichment failure blocked lifecycle completion: %+v", record)
	}
}

type failingRequestDemandStore struct {
	store.Storage
}

func (f *failingRequestDemandStore) UpdateRequestDemandSnapshots(
	context.Context,
	string,
	[]store.RequestDemandSnapshot,
) error {
	return errors.New("request demand store unavailable")
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
