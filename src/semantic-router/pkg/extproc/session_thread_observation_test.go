package extproc

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestStatelessResponsesKeepIndependentThreadModels(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := newAgentIdentityRouter(t, "", "x-agent-id")
	const sessionValue = "run-shared"
	const childAgent = "thread-review"

	root := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	prepareStatelessResponse(t, router, root)
	recordObservedModel(t, root, "model-root")

	rootTool := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	prepareStatelessResponse(t, router, rootTool)
	if rootTool.PreviousModel != "model-root" || !rootTool.SessionIdleKnown {
		t.Fatalf("root tool continuation previous=%q idleKnown=%v, want model-root and a known idle time", rootTool.PreviousModel, rootTool.SessionIdleKnown)
	}
	recordObservedModel(t, rootTool, "model-root")

	child := statelessResponseContext(map[string]string{
		"x-session-id": sessionValue,
		"x-agent-id":   childAgent,
	})
	prepareStatelessResponse(t, router, child)
	if child.PreviousModel != "" || child.SessionIdleKnown {
		t.Fatalf("review child resumed the root model: previous=%q idleKnown=%v", child.PreviousModel, child.SessionIdleKnown)
	}
	recordObservedModel(t, child, "model-child")

	childAgain := statelessResponseContext(map[string]string{
		"x-session-id": sessionValue,
		"x-agent-id":   childAgent,
	})
	prepareStatelessResponse(t, router, childAgain)
	if childAgain.PreviousModel != "model-child" || !childAgain.SessionIdleKnown {
		t.Fatalf("child continuation previous=%q idleKnown=%v, want model-child and a known idle time", childAgain.PreviousModel, childAgain.SessionIdleKnown)
	}
	recordObservedModel(t, childAgain, "model-child")

	rootAgain := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	prepareStatelessResponse(t, router, rootAgain)
	if rootAgain.PreviousModel != "model-root" {
		t.Fatalf("resumed root previous=%q, want model-root", rootAgain.PreviousModel)
	}
}

func TestRequestsWithoutAgentHeaderKeepSessionObservationKeys(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := newAgentIdentityRouter(t, "", "x-agent-id")
	const sessionValue = "run-shared"

	root := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	prepareStatelessResponse(t, router, root)
	recordObservedModel(t, root, "model-root")

	child := statelessResponseContext(map[string]string{
		"x-session-id": sessionValue,
		"x-agent-id":   "thread-review",
	})
	prepareStatelessResponse(t, router, child)
	recordObservedModel(t, child, "model-child")

	rootAgain := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	prepareStatelessResponse(t, router, rootAgain)
	if rootAgain.PreviousModel != "model-root" {
		t.Fatalf("root without an agent header resumed %q", rootAgain.PreviousModel)
	}
	if modelObservationKey(rootAgain) != routingSessionStateKey(rootAgain) {
		t.Fatalf("root observation key = %q, session key = %q", modelObservationKey(rootAgain), routingSessionStateKey(rootAgain))
	}
}

func TestAgentObservationLeavesSharedSessionKey(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := newAgentIdentityRouter(t, "", "x-agent-id")
	const sessionValue = "run-shared"
	root := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	child := statelessResponseContext(map[string]string{
		"x-session-id": sessionValue,
		"x-agent-id":   "thread-review",
	})
	prepareStatelessResponse(t, router, root)
	prepareStatelessResponse(t, router, child)

	shared := routingSessionStateKey(root)
	if shared == "" || shared != routingSessionStateKey(child) {
		t.Fatalf("shared session key root=%q child=%q", shared, routingSessionStateKey(child))
	}
	if got := modelObservationKey(child); got == "" || got == shared {
		t.Fatalf("child observation key = %q, shared session key = %q", got, shared)
	}
	if got := modelObservationKey(root); got != shared {
		t.Fatalf("root observation key = %q, shared session key = %q", got, shared)
	}

	selection := router.buildSelectionContext(nil, "route", "", nil, "", nil, child)
	if selection.SessionID != sessionValue || selection.SessionStateKey == shared {
		t.Fatalf("selection session=%q key=%q, want session %q and the child observation key", selection.SessionID, selection.SessionStateKey, sessionValue)
	}
}

func TestSelectionResolvesAgentIdentityWithoutPriorBind(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := newAgentIdentityRouter(t, "", "x-agent-id")
	const sessionValue = "run-shared"
	root := statelessResponseContext(map[string]string{"x-session-id": sessionValue})
	prepareStatelessResponse(t, router, root)
	recordObservedModel(t, root, "model-root")

	child := statelessResponseContext(map[string]string{
		"x-session-id": sessionValue,
		"x-agent-id":   "thread-review",
	})
	child.SessionID = sessionValue
	selection := router.buildSelectionContext(nil, "route", "", nil, "", nil, child)
	if selection.SessionStateKey == "" || selection.SessionStateKey == routingSessionStateKey(child) {
		t.Fatalf("selection key = %q, shared session key = %q", selection.SessionStateKey, routingSessionStateKey(child))
	}
	if selection.AgenticSession == nil || selection.AgenticSession.PreviousModel != "" || selection.AgenticSession.IdleKnown {
		t.Fatalf("selection inherited the root observation: %+v", selection.AgenticSession)
	}
}

func TestReplayRecordsAgentWhenConversationScopeHasNoConversation(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	const (
		sessionHeader = "x-client-session"
		agentHeader   = "x-client-agent"
		sessionValue  = "configured-run"
		childAgent    = "thread-review"
	)
	router := newAgentIdentityRouter(t, sessionHeader, agentHeader)
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(8, 0))
	router.ReplayRecorder = recorder
	router.ReplayStoreShared = true

	child := statelessResponseContext(map[string]string{
		sessionHeader: sessionValue,
		agentHeader:   childAgent,
	})
	prepareStatelessResponse(t, router, child)
	if child.SessionID == "" || child.SessionID == sessionValue {
		t.Fatalf("responses tracking id = %q, want a fresh id distinct from %q", child.SessionID, sessionValue)
	}
	if _, ok := router.protectionIdentity(child, router.Config.RouterLearning.Protection); ok {
		t.Fatal("default conversation scope without a conversation header must not form a protection identity")
	}
	childRecord := startObservedReplay(t, router, recorder, child, "model-child")
	if childRecord.SessionID != sessionValue || childRecord.AgentID != childAgent || childRecord.ConversationID != "" {
		t.Fatalf("child replay session=%q agent=%q conversation=%q", childRecord.SessionID, childRecord.AgentID, childRecord.ConversationID)
	}
	recordObservedModel(t, child, "model-child")

	continued := statelessResponseContext(map[string]string{
		sessionHeader: sessionValue,
		agentHeader:   childAgent,
	})
	prepareStatelessResponse(t, router, continued)
	if continued.PreviousModel != "model-child" {
		t.Fatalf("same agent resumed %q, want model-child", continued.PreviousModel)
	}

	root := statelessResponseContext(map[string]string{sessionHeader: sessionValue})
	prepareStatelessResponse(t, router, root)
	if root.PreviousModel != "" {
		t.Fatalf("root turn without an agent header resumed %q", root.PreviousModel)
	}
	rootRecord := startObservedReplay(t, router, recorder, root, "model-root")
	if rootRecord.SessionID != sessionValue || rootRecord.AgentID != "" {
		t.Fatalf("root replay session=%q agent=%q", rootRecord.SessionID, rootRecord.AgentID)
	}
	if rootRecord.SessionID != childRecord.SessionID {
		t.Fatalf("replay run ids diverged: root=%q child=%q", rootRecord.SessionID, childRecord.SessionID)
	}
}

func newAgentIdentityRouter(t *testing.T, sessionHeader, agentHeader string) *OpenAIRouter {
	t.Helper()
	cfg := &config.RouterConfig{}
	if sessionHeader != "" {
		cfg.RouterLearning.Identity.Headers.Session = &sessionHeader
	}
	if agentHeader != "" {
		cfg.RouterLearning.Identity.Headers.Agent = &agentHeader
	}
	protectionOff := false
	cfg.RouterLearning.Protection.Enabled = &protectionOff
	return newStatelessResponseRouter(t, cfg)
}

func newStatelessResponseRouter(t *testing.T, cfg *config.RouterConfig) *OpenAIRouter {
	t.Helper()
	storage, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := storage.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	return &OpenAIRouter{Config: cfg, ResponseAPIFilter: NewResponseAPIFilter(storage)}
}

var statelessResponseSeq atomic.Uint64

func statelessResponseContext(headers map[string]string) *RequestContext {
	ctx := &RequestContext{
		Headers:      headers,
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TraceContext: context.Background(),
		RequestID:    fmt.Sprintf("req-%d", statelessResponseSeq.Add(1)),
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "coding"})
	return ctx
}

func prepareStatelessResponse(t *testing.T, router *OpenAIRouter, ctx *RequestContext) {
	t.Helper()
	request, immediate := router.prepareProtocolRequest(
		[]byte(`{"model":"public","input":"continue the task","store":false}`),
		ctx,
	)
	if immediate != nil || request == nil {
		t.Fatalf("prepareProtocolRequest failed: %+v", immediate)
	}
	if _, followUp, err := router.prepareRequestForModelRouting(request, "continue the task", ctx); followUp != nil || err != nil {
		t.Fatalf("prepareRequestForModelRouting: err=%v response=%+v", err, followUp)
	}
}

func recordObservedModel(t *testing.T, ctx *RequestContext, model string) {
	t.Helper()
	ctx.RequestModel = model
	recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 8, completionTokens: 4}, sessiontelemetry.TurnPricing{})
}

func startObservedReplay(
	t *testing.T,
	router *OpenAIRouter,
	recorder *routerreplay.Recorder,
	ctx *RequestContext,
	model string,
) routerreplay.RoutingRecord {
	t.Helper()
	ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true}
	router.startRouterReplay(ctx, "public", model, "route")
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	if !found {
		t.Fatal("replay record was not stored")
	}
	return record
}
