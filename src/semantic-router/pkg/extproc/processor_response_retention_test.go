package extproc

import (
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

func TestResponsesRetentionPolicyAcrossCompletionPaths(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		name  string
		store *bool
		drop  *bool
	}{
		{name: "default store"},
		{name: "explicit store", store: &yes},
		{name: "client no-store", store: &no},
		{name: "drop default store", drop: &yes},
		{name: "drop explicit store", store: &yes, drop: &yes},
		{name: "drop client no-store", store: &no, drop: &yes},
		{name: "keep default store", drop: &no},
		{name: "keep respects client no-store", store: &no, drop: &no},
	}
	for _, tc := range cases {
		for _, backend := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			for _, path := range []string{"buffered", "streaming", "immediate"} {
				t.Run(tc.name+"/"+string(backend)+"/"+path, func(t *testing.T) {
					router, ctx, responseStore, memoryStore, runner := responseRetentionTestContext(t, tc.store)
					ctx.TargetFormat = backend
					if tc.drop != nil {
						ctx.EmittedRetention = &config.RetentionDirective{Drop: tc.drop}
					}
					completeRetentionTestResponse(t, router, ctx, path)
					if err := runner.RetireAndWait(time.Second); err != nil {
						t.Fatal(err)
					}
					if ctx.SemanticResponse == nil || ctx.SemanticResponse.ID != ctx.ResponseObjectState.GeneratedResponseID {
						t.Fatal("retention policy changed successful response identity")
					}
					dropped := tc.drop != nil && *tc.drop
					wantStored := !dropped && (tc.store == nil || *tc.store)
					_, err := responseStore.GetResponse(t.Context(), ctx.SemanticResponse.ID)
					if wantStored && err != nil || !wantStored && !errors.Is(err, responsestore.ErrNotFound) {
						t.Fatalf("stored response error = %v, wantStored = %v", err, wantStored)
					}
					if _, err := responseStore.GetResponse(t.Context(), "resp_parent"); err != nil {
						t.Fatalf("retention of this turn removed the parent: %v", err)
					}
					// store:false controls response objects, not independent memory opt-in.
					wantMemoryWrites := 1
					if dropped || path == "immediate" {
						wantMemoryWrites = 0
					}
					if memoryStore.writes != wantMemoryWrites {
						t.Fatalf("memory writes = %d, want %d", memoryStore.writes, wantMemoryWrites)
					}
					// Repeated terminal persistence remains idempotent.
					router.persistResponseObject(ctx)
					wantObjects := 1
					if wantStored {
						wantObjects++
					}
					if len(responseStore.responses) != wantObjects {
						t.Fatalf("retained objects = %d, want %d", len(responseStore.responses), wantObjects)
					}
				})
			}
		}
	}
}

func responseRetentionTestContext(t *testing.T, store *bool) (*OpenAIRouter, *RequestContext, *MockResponseStore, *countingPolicyMemoryStore, *memory.PersistenceRunner) {
	t.Helper()
	responseStore := NewMockResponseStore()
	parent := &responseapi.StoredResponse{ID: "resp_parent", OutputText: "The conference starts on Monday."}
	if err := responseStore.StoreResponse(t.Context(), parent); err != nil {
		t.Fatal(err)
	}
	memoryStore := &countingPolicyMemoryStore{}
	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, AutoStore: true}},
		ResponseAPIFilter: NewResponseAPIFilter(responseStore), MemoryExtractor: memory.NewMemoryChunkStore(memoryStore),
		memoryPersistence: runner,
	}
	ctx := memoryPolicyContext(nil)
	ctx.TraceContext = t.Context()
	ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
	ctx.SemanticRequest.Store = store
	ctx.SemanticRequest.PreviousResponseID = parent.ID
	state, err := router.ResponseAPIFilter.PrepareObjectState(t.Context(), *ctx.SemanticRequest, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx.ResponseObjectState = state
	if _, err := router.materializeResponseObjectContext(ctx.SemanticRequest, ctx); err != nil {
		t.Fatal(err)
	}
	if len(ctx.SemanticRequest.Messages) != 2 || state.PreviousResponseID != parent.ID {
		t.Fatal("no-store must still materialize an explicitly requested existing parent")
	}
	return router, ctx, responseStore, memoryStore, runner
}

func completeRetentionTestResponse(t *testing.T, router *OpenAIRouter, ctx *RequestContext, path string) {
	t.Helper()
	switch path {
	case "buffered":
		response := router.handleNonStreamingResponseBody(extProcResponseFixture(ctx.TargetFormat), ctx, 0)
		if response.GetResponseBody() == nil {
			t.Fatal("buffered generation did not complete")
		}
	case "streaming":
		response := router.handleSemanticStreamingResponseBody(extProcStreamFixture(ctx.TargetFormat), true, ctx)
		if response.GetResponseBody() == nil || !ctx.StreamingComplete || ctx.StreamingAborted {
			t.Fatal("streaming generation did not complete")
		}
	case "immediate":
		ctx.SemanticResponse = memoryTestResponse("The conference starts on Monday.")
		ctx.SemanticResponse.ID = ctx.ResponseObjectState.GeneratedResponseID
		router.persistImmediateResponseObject(createImmediateJSONResponse(200, []byte(`{"status":"completed"}`)), ctx)
	}
}
