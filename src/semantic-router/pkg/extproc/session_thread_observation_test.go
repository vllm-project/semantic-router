package extproc

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestStatelessResponsesKeepThreadObservations(t *testing.T) {
	sessionValue := fmt.Sprintf("run-%d", time.Now().UnixNano())
	sessionHeader, agentHeader := "session-id", "thread-id"
	cfg := &config.RouterConfig{}
	cfg.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
	cfg.RouterLearning.Protection.Identity.Headers.Session = &sessionHeader
	cfg.RouterLearning.Protection.Identity.Headers.Agent = &agentHeader
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(20, 0))
	router := &OpenAIRouter{Config: cfg, ReplayRecorder: recorder, ReplayStoreShared: true}

	steps := []struct {
		name     string
		thread   string
		model    string
		tracking string
		wantPrev string
		wantIdle bool
	}{
		{name: "root", thread: "thread-root", model: "model-root", tracking: "resp-root-1"},
		{name: "root tool continuation", thread: "thread-root", model: "model-root", tracking: "resp-root-2", wantPrev: "model-root", wantIdle: true},
		{name: "resumed root", thread: "thread-root", model: "model-root", tracking: "resp-root-3", wantPrev: "model-root", wantIdle: true},
		{name: "review child", thread: "thread-review", model: "model-child", tracking: "resp-child-1"},
		{name: "child continuation", thread: "thread-review", model: "model-child", tracking: "resp-child-2", wantPrev: "model-child", wantIdle: true},
		{name: "resumed root after child", thread: "thread-root", model: "model-root", tracking: "resp-root-4", wantPrev: "model-root", wantIdle: true},
	}

	records := make([]routerreplay.RoutingRecord, 0, len(steps))
	for _, step := range steps {
		t.Run(step.name, func(t *testing.T) {
			ctx := threadObservationContext(sessionHeader, agentHeader, sessionValue, step.thread, step.tracking)
			router.bindThreadObservationKey(ctx)
			populateSessionTransitionFields(ctx)
			if ctx.SessionID != step.tracking {
				t.Fatalf("session id = %q, want the standalone tracking id %q", ctx.SessionID, step.tracking)
			}
			if ctx.PreviousModel != step.wantPrev || ctx.SessionIdleKnown != step.wantIdle {
				t.Fatalf("observation previous=%q idleKnown=%v, want previous=%q idleKnown=%v", ctx.PreviousModel, ctx.SessionIdleKnown, step.wantPrev, step.wantIdle)
			}

			selectionContext := router.buildSelectionContext(nil, "route", "", nil, "", nil, ctx)
			wantKey := router.configuredAgentObservationKey(ctx)
			if wantKey == "" || selectionContext.SessionStateKey != wantKey {
				t.Fatalf("selection state key = %q, want %q", selectionContext.SessionStateKey, wantKey)
			}
			if selectionContext.SessionID != step.tracking {
				t.Fatalf("selection session id = %q, want tracking id %q", selectionContext.SessionID, step.tracking)
			}
			if selectionContext.AgenticSession == nil || selectionContext.AgenticSession.PreviousModel != step.wantPrev || selectionContext.AgenticSession.IdleKnown != step.wantIdle {
				t.Fatalf("agentic observation = %+v, want previous %q idleKnown %v", selectionContext.AgenticSession, step.wantPrev, step.wantIdle)
			}
			if kvKey := routingSessionStateKey(ctx); kvKey == "" || kvKey == wantKey {
				t.Fatalf("kv key %q must stay on the tracking session and not become %q", kvKey, wantKey)
			}

			identity, ok := router.protectionIdentity(ctx, cfg.RouterLearning.Protection)
			wantMemory := sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, sessionValue)
			if !ok || identity.memoryKey != wantMemory || strings.Contains(identity.memoryKey, step.thread) {
				t.Fatalf("protection memory key = %q ok=%v, want %q without the agent id", identity.memoryKey, ok, wantMemory)
			}

			ctx.RequestModel = step.model
			ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true}
			router.startRouterReplay(ctx, "public", step.model, "route")
			recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 8, completionTokens: 4}, sessiontelemetry.TurnPricing{})
			record, found := recorder.GetRecord(ctx.RouterReplayID)
			if !found {
				t.Fatal("replay record was not stored")
			}
			if record.SessionID != sessionValue || record.AgentID != step.thread || record.SelectedModel != step.model {
				t.Fatalf("replay session=%q agent=%q model=%q", record.SessionID, record.AgentID, record.SelectedModel)
			}
			records = append(records, record)
		})
	}

	if len(records) != len(steps) {
		t.Fatalf("stored %d replay records, want %d", len(records), len(steps))
	}
	for _, record := range records {
		if record.SessionID != sessionValue {
			t.Fatalf("run id %q, want shared session %q", record.SessionID, sessionValue)
		}
	}
	routes := buildTrajectoryRoutes(records)
	if len(routes) != len(steps) || routes[0].AgentID != "thread-root" || routes[3].AgentID != "thread-review" || routes[5].AgentID != "thread-root" {
		t.Fatalf("trajectory lost thread identity: %+v", routes)
	}
	if records[0].SelectedModel == records[3].SelectedModel {
		t.Fatal("mixed-model run needs two selected models under one session id")
	}
}

func TestRequestsWithoutAgentHeaderKeepSessionObservationKeys(t *testing.T) {
	cfg := &config.RouterConfig{}
	router := &OpenAIRouter{Config: cfg}
	pinned := fmt.Sprintf("pinned-%d", time.Now().UnixNano())
	first := pinnedObservationContext(pinned)
	observeAndRecord(t, router, first, "model-a")
	second := pinnedObservationContext(pinned)
	router.bindThreadObservationKey(second)
	populateSessionTransitionFields(second)
	if second.PreviousModel != "model-a" || !second.SessionIdleKnown {
		t.Fatalf("pinned session lost its observation: previous=%q idleKnown=%v", second.PreviousModel, second.SessionIdleKnown)
	}
	if modelObservationKey(second) != routingSessionStateKey(second) {
		t.Fatal("a request without an agent header changed the observation key")
	}

	leftTracking := fmt.Sprintf("resp-left-%d", time.Now().UnixNano())
	rightTracking := fmt.Sprintf("resp-right-%d", time.Now().UnixNano())
	left := standaloneObservationContext(leftTracking)
	observeAndRecord(t, router, left, "model-left")
	right := standaloneObservationContext(rightTracking)
	router.bindThreadObservationKey(right)
	populateSessionTransitionFields(right)
	if right.PreviousModel != "" || right.SessionIdleKnown {
		t.Fatalf("standalone responses shared an observation: previous=%q idleKnown=%v", right.PreviousModel, right.SessionIdleKnown)
	}

	sessionHeader, agentHeader := "session-id", "thread-id"
	sessionValue := fmt.Sprintf("omitted-%d", time.Now().UnixNano())
	configured := &config.RouterConfig{}
	configured.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
	configured.RouterLearning.Protection.Identity.Headers.Session = &sessionHeader
	configured.RouterLearning.Protection.Identity.Headers.Agent = &agentHeader
	configuredRouter := &OpenAIRouter{Config: configured}
	omitted := threadObservationContext(sessionHeader, agentHeader, sessionValue, "", fmt.Sprintf("resp-omit-%d", time.Now().UnixNano()))
	delete(omitted.Headers, agentHeader)
	observeAndRecord(t, configuredRouter, omitted, "model-omitted")
	followed := threadObservationContext(sessionHeader, agentHeader, sessionValue, "", fmt.Sprintf("resp-omit-next-%d", time.Now().UnixNano()))
	delete(followed.Headers, agentHeader)
	configuredRouter.bindThreadObservationKey(followed)
	populateSessionTransitionFields(followed)
	if followed.PreviousModel != "" || followed.threadObservationKey != "" || modelObservationKey(followed) != routingSessionStateKey(followed) {
		t.Fatalf("omitted agent header changed observations: previous=%q key=%q routing=%q", followed.PreviousModel, followed.threadObservationKey, routingSessionStateKey(followed))
	}
	var replay routerreplay.RoutingRecord
	configuredRouter.populateReplayIdentity(&replay, followed)
	if replay.SessionID != sessionValue || replay.AgentID != "" {
		t.Fatalf("replay session=%q agent=%q, want session %q and no agent", replay.SessionID, replay.AgentID, sessionValue)
	}
}

func observeAndRecord(t *testing.T, router *OpenAIRouter, ctx *RequestContext, model string) {
	t.Helper()
	router.bindThreadObservationKey(ctx)
	populateSessionTransitionFields(ctx)
	ctx.RequestModel = model
	recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 8, completionTokens: 4}, sessiontelemetry.TurnPricing{})
}

func threadObservationContext(sessionHeader, agentHeader, sessionValue, thread, tracking string) *RequestContext {
	headers := map[string]string{sessionHeader: sessionValue}
	if thread != "" {
		headers[agentHeader] = thread
	}
	ctx := &RequestContext{
		Headers: headers,
		ResponseObjectState: &ResponseObjectState{
			SessionTrackingID: tracking,
		},
		RequestID: tracking,
		SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "continue"}},
		}}},
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "coding"})
	return ctx
}

func pinnedObservationContext(sessionID string) *RequestContext {
	ctx := &RequestContext{
		Headers: map[string]string{"x-session-id": sessionID},
		SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "continue"}},
		}}},
		RequestID: sessionID,
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "coding"})
	return ctx
}

func standaloneObservationContext(tracking string) *RequestContext {
	ctx := &RequestContext{
		Headers:             map[string]string{},
		ResponseObjectState: &ResponseObjectState{SessionTrackingID: tracking},
		RequestID:           tracking,
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "coding"})
	return ctx
}
