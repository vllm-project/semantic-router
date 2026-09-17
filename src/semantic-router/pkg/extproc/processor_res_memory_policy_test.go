package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

type countingPolicyMemoryStore struct {
	noopMemoryStore
	writes int
}

func (s *countingPolicyMemoryStore) Store(_ context.Context, _ *memory.Memory) error {
	s.writes++
	return nil
}

func TestResponseMemoryWritesHonorPolicyBeforeClientPreference(t *testing.T) {
	yes, no := true, false
	cases := []struct {
		name       string
		global     config.MemoryConfig
		decision   *config.MemoryPluginConfig
		request    *bool
		drop       *bool
		disable    bool
		wantWrites int
	}{
		{name: "global default", global: config.MemoryConfig{Enabled: true, AutoStore: true}, wantWrites: 1},
		{name: "global disabled", global: config.MemoryConfig{AutoStore: true}, request: &yes},
		{name: "decision disabled", global: config.MemoryConfig{Enabled: true, AutoStore: true}, decision: &config.MemoryPluginConfig{Enabled: false}, request: &yes},
		{name: "decision auto-store disabled overrides global", global: config.MemoryConfig{Enabled: true, AutoStore: true}, decision: &config.MemoryPluginConfig{Enabled: true, AutoStore: &no}},
		{name: "decision auto-store disabled overrides client", global: config.MemoryConfig{Enabled: true, AutoStore: true}, decision: &config.MemoryPluginConfig{Enabled: true, AutoStore: &no}, request: &yes},
		{name: "client opt-out", global: config.MemoryConfig{Enabled: true, AutoStore: true}, request: &no},
		{name: "client opt-out overrides decision opt-in", global: config.MemoryConfig{Enabled: true}, decision: &config.MemoryPluginConfig{Enabled: true, AutoStore: &yes}, request: &no},
		{name: "client opt-in", global: config.MemoryConfig{Enabled: true}, request: &yes, wantWrites: 1},
		{name: "decision opt-in", global: config.MemoryConfig{}, decision: &config.MemoryPluginConfig{Enabled: true, AutoStore: &yes}, wantWrites: 1},
		{name: "retention drop", global: config.MemoryConfig{Enabled: true, AutoStore: true}, request: &yes, drop: &yes},
		{name: "retention keep", global: config.MemoryConfig{Enabled: true, AutoStore: true}, drop: &no, wantWrites: 1},
		{name: "request memory opt-out", global: config.MemoryConfig{Enabled: true, AutoStore: true}, request: &yes, disable: true},
		{name: "route opt-out", global: config.MemoryConfig{Enabled: true, AutoStore: true, DisabledRoutes: []string{"/v1/responses"}}, request: &yes},
		{name: "model opt-out", global: config.MemoryConfig{Enabled: true, AutoStore: true, DisabledModels: []string{"model"}}, request: &yes},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := &countingPolicyMemoryStore{}
			runner := memory.NewPersistenceRunner(time.Second, 1, 1)
			t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })
			router := &OpenAIRouter{
				Config: &config.RouterConfig{Memory: tc.global}, MemoryExtractor: memory.NewMemoryChunkStore(store),
				memoryPersistence: runner,
			}
			ctx := memoryPolicyContext(tc.request)
			if tc.decision != nil {
				payload, err := config.NewStructuredPayload(*tc.decision)
				if err != nil {
					t.Fatal(err)
				}
				ctx.VSRSelectedDecision = &config.Decision{Name: "memory-policy", Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginMemory, Configuration: payload}}}
			}
			if tc.drop != nil {
				ctx.EmittedRetention = &config.RetentionDirective{Drop: tc.drop}
			}
			if tc.disable {
				ctx.Headers[headers.DisableRouterMemory] = "true"
			}
			router.scheduleResponseMemoryStoreText(ctx, "The conference itinerary includes a morning meeting and an afternoon workshop.")
			if err := runner.RetireAndWait(time.Second); err != nil {
				t.Fatal(err)
			}
			if store.writes != tc.wantWrites {
				t.Fatalf("memory writes = %d, want %d", store.writes, tc.wantWrites)
			}
		})
	}
}

func TestResponseMemoryClientOptOutSurvivesProviderMaterialization(t *testing.T) {
	autoStore := false
	ctx := memoryPolicyContext(&autoStore)
	store := &countingPolicyMemoryStore{}
	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, AutoStore: true}},
		MemoryExtractor:   memory.NewMemoryChunkStore(store),
		memoryPersistence: runner,
	}
	state, err := (*ResponseAPIFilter)(nil).PrepareObjectState(t.Context(), *ctx.SemanticRequest, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx.ResponseObjectState = state
	// The retained preference must be a snapshot, not a pointer into the caller.
	autoStore = true
	if _, err := router.materializeResponseObjectContext(ctx.SemanticRequest, ctx); err != nil {
		t.Fatal(err)
	}
	if ctx.SemanticRequest.AutoStore != nil {
		t.Fatal("Router storage control leaked to the provider request")
	}
	router.scheduleResponseMemoryStoreText(ctx, "The conference itinerary includes a morning meeting and an afternoon workshop.")
	if err := runner.RetireAndWait(time.Second); err != nil {
		t.Fatal(err)
	}
	if store.writes != 0 {
		t.Fatalf("client opt-out was lost after materialization: %d writes", store.writes)
	}
}

func memoryPolicyContext(autoStore *bool) *RequestContext {
	return &RequestContext{
		RequestModel: "model",
		Headers:      map[string]string{headers.AuthzUserID: "memory-user", ":path": "/v1/responses"},
		SemanticRequest: &llmprotocol.Request{
			Generation: 1, Model: "model", AutoStore: autoStore,
			Messages: []llmprotocol.Message{neutralTextMessage(llmprotocol.RoleUser, "Please remember the detailed conference itinerary with a morning meeting and an afternoon workshop.")},
		},
	}
}
