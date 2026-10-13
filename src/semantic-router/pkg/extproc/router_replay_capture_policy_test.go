package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func replayPolicyRequest(t *testing.T) *RequestContext {
	t.Helper()
	return &RequestContext{
		TraceContext: t.Context(), RequestID: "synthetic-replay-policy", SourceFormat: llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public-entry", Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Summarize the meeting agenda."}}},
		}},
	}
}

func replayRuntimeRouter(t *testing.T, cfg *config.RouterConfig) *OpenAIRouter {
	t.Helper()
	recorders, fallback, shared, err := createReplayRuntime(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = closeReplayRecorders(fallback, recorders, shared) })
	return &OpenAIRouter{Config: cfg, ReplayRecorders: recorders, ReplayRecorder: fallback, ReplayStoreShared: shared}
}

func replayDecision(values map[string]interface{}) *config.Decision {
	return &config.Decision{Name: "ordinary", Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginRouterReplay, Configuration: config.MustStructuredPayload(values)}}}
}

func TestReplayGlobalRuntimeCapturesBeforeDecisionAndDirectRequests(t *testing.T) {
	no := false
	for _, enabled := range []bool{false, true} {
		for _, direct := range []bool{false, true} {
			cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: enabled, StoreBackend: "memory", CapturePersonalData: &no}}
			router := replayRuntimeRouter(t, cfg)
			ctx := replayPolicyRequest(t)
			if direct {
				ctx.Routing.SelectPassthrough()
			}
			response := router.respondDecisionUnresolved(ctx, "public-entry", &decision.DecisionUnresolvedError{Decision: "unknown"})
			if response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_ServiceUnavailable {
				t.Fatal("capture changed routing failure")
			}
			if (ctx.RouterReplayID != "") != enabled {
				t.Fatalf("enabled=%v direct=%v id=%q", enabled, direct, ctx.RouterReplayID)
			}
			if !enabled {
				continue
			}
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !ok || record.RequestBody != "" || record.Prompt != "" || record.ResponseBody != "" {
				t.Fatalf("unverified early content leaked: %+v", record)
			}
			if record.LifecycleState != routerreplay.LifecycleFailed {
				t.Fatalf("metadata lifecycle missing: %q", record.LifecycleState)
			}
			if ctx.VSRSelectedDecision != nil {
				t.Fatal("early capture fabricated a decision")
			}
		}
	}
}

func TestReplaySelectedDecisionReplacesCachedGlobalPolicy(t *testing.T) {
	no := false
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, CapturePersonalData: &no}}
	router := replayRuntimeRouter(t, cfg)
	ctx := replayPolicyRequest(t)
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	disabled := replayDecision(map[string]interface{}{"enabled": false})
	router.applyDecisionResultToContext(&decision.DecisionResult{Decision: disabled}, ctx)
	if ctx.RouterReplayPluginConfig != nil {
		t.Fatal("decision disable retained cached global capture")
	}
	router.respondDecisionUnresolved(ctx, "public-entry", &decision.DecisionUnresolvedError{Decision: disabled.Name})
	if ctx.RouterReplayID != "" {
		t.Fatal("post-selection rejection re-enabled global capture")
	}
	// A failed earlier persistence attempt can leave capture state without an ID.
	// The next capture resolves the selected policy rather than keeping that state.
	ctx.RouterReplayContentOmitted = true
	allow := replayDecision(map[string]interface{}{"capture_personal_data": true})
	router.applyDecisionResultToContext(&decision.DecisionResult{Decision: allow}, ctx)
	router.startRouterReplay(ctx, "public-entry", "model", allow.Name)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.Prompt == "" || ctx.RouterReplayContentOmitted {
		t.Fatal("decision override did not restore content eligibility")
	}
}

func TestReplayPersonalDataPolicyKeepsMetadataAndLifecycle(t *testing.T) {
	for _, test := range []struct {
		name                              string
		capture, verified, detected, keep bool
	}{
		{"PII", false, true, true, false},
		{"verified clean", false, true, false, true},
		{"unavailable", false, false, false, false},
		{"default capture", true, false, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory", CapturePersonalData: &test.capture}}
			router := replayRuntimeRouter(t, cfg)
			ctx := replayPolicyRequest(t)
			ctx.PIIContentVerified, ctx.PIIDetected = test.verified, test.detected
			ctx.PIIEvidence = []classification.PrivacyEvidence{classification.NewPrivacyEvidence("request", "Summarize the meeting agenda.", test.verified, !test.detected)}
			ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			router.attachRouterReplayResponse(ctx, []byte(`{"text":"Meeting agenda."}`), true)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !ok {
				t.Fatal("missing metadata record")
			}
			kept := record.RequestBody != "" || record.Prompt != "" || record.ResponseBody != ""
			if kept != test.keep {
				t.Fatalf("content kept=%v want=%v", kept, test.keep)
			}
			if record.Decision != "ordinary" || record.PIIDetected != test.detected || record.LifecycleState != routerreplay.LifecycleCompleted {
				t.Fatalf("lost routing metadata: %+v", record)
			}
			if ctx.SemanticRequest.Messages[0].Content[0].Text != "Summarize the meeting agenda." {
				t.Fatal("capture changed provider request")
			}
		})
	}
}

func TestReplayRuntimeUsesGlobalMemoryCapacity(t *testing.T) {
	max := 1
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory", MaxRecords: &max}}
	router := replayRuntimeRouter(t, cfg)
	for i := 0; i < 2; i++ {
		ctx := replayPolicyRequest(t)
		ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
		router.startRouterReplay(ctx, "public-entry", "model", "")
	}
	if got := len(router.ReplayRecorder.ListAllRecords()); got != 1 {
		t.Fatalf("global memory max_records ignored: %d", got)
	}
}

func TestReplayOmittedContentDropsResponseDetectorExcerpts(t *testing.T) {
	for _, capture := range []bool{false, true} {
		name := "omit"
		if capture {
			name = "allow"
		}
		t.Run(name, func(t *testing.T) {
			cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, CapturePersonalData: &capture}}
			router := replayRuntimeRouter(t, cfg)
			ctx := replayPolicyRequest(t)
			ctx.VSRSelectedDecision = &config.Decision{Name: "ordinary", Plugins: []config.DecisionPlugin{{
				Type: "hallucination", Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true}),
			}}}
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			ctx.HallucinationDetected, ctx.HallucinationScoreAvailable = true, true
			ctx.HallucinationConfidence = 0.9
			ctx.HallucinationSpans = []string{"private@example.invalid"}
			ctx.EnhancedHallucinationInfo = &EnhancedHallucinationInfo{Spans: []EnhancedHallucinationSpan{{
				Text: "private@example.invalid", Explanation: "The private@example.invalid address is unsupported.",
			}}}
			router.updateRouterReplayHallucinationStatus(ctx)
			record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !found || !record.HallucinationDetected || !record.HallucinationScoreAvailable || record.HallucinationConfidence != 0.9 {
				t.Fatal("detector verdict or score metadata was lost")
			}
			if (len(record.HallucinationSpans) > 0) != capture || (len(record.HallucinationSpanDetails) > 0) != capture {
				t.Fatalf("detector response excerpts did not follow capture policy: capture=%t", capture)
			}
			if len(ctx.HallucinationSpans) != 1 || len(ctx.EnhancedHallucinationInfo.Spans) != 1 {
				t.Fatal("Replay changed the response detector result")
			}
		})
	}
}

func TestReplayDirectStartResolvesGlobalPolicyWithoutDecisionPreparation(t *testing.T) {
	no := false
	router := replayRuntimeRouter(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory", CapturePersonalData: &no}})
	ctx := replayPolicyRequest(t)
	ctx.Routing.SelectPassthrough()
	router.startRouterReplay(ctx, "backend", "backend", "")
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.Prompt != "" || record.RequestBody != "" {
		t.Fatal("direct request did not resolve global metadata-only capture")
	}
}

func TestReplayQueriesIncludeGlobalAndDecisionMemoryStores(t *testing.T) {
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"}, IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{*replayDecision(map[string]interface{}{})}}}
	router := replayRuntimeRouter(t, cfg)
	before := replayPolicyRequest(t)
	router.startRouterReplay(before, "direct", "direct", "")
	selected := replayPolicyRequest(t)
	router.applyDecisionResultToContext(&decision.DecisionResult{Decision: &cfg.Decisions[0]}, selected)
	router.startRouterReplay(selected, "public", "backend", "ordinary")
	page, err := router.queryRouterReplayPage(routerReplayListQuery{limit: 10})
	if err != nil || len(page.Data) != 2 {
		t.Fatalf("global/decision records not both queryable: %+v err=%v", page, err)
	}
}

func TestReplayRequestPolicyDoesNotBorrowAnotherFallbackCapture(t *testing.T) {
	no := false
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory", CaptureResponseBody: &no}}
	router := replayRuntimeRouter(t, cfg)
	global := replayPolicyRequest(t)
	router.startRouterReplay(global, "direct", "direct", "")
	selected := replayPolicyRequest(t)
	d := replayDecision(map[string]interface{}{"capture_response_body": true})
	router.applyDecisionResultToContext(&decision.DecisionResult{Decision: d}, selected)
	router.startRouterReplay(selected, "public", "backend", d.Name)
	body := []byte(`{"text":"Synthetic response"}`)
	router.attachRouterReplayResponse(global, body, true)
	router.attachRouterReplayResponse(selected, body, true)
	before, ok := router.ReplayRecorder.GetRecord(global.RouterReplayID)
	if !ok || before.ResponseBody != "" {
		t.Fatal("global request borrowed selected decision capture policy")
	}
	after, ok := router.ReplayRecorder.GetRecord(selected.RouterReplayID)
	if !ok || after.ResponseBody == "" {
		t.Fatal("decision override lost its own response capture policy")
	}
}

func TestReplayCleanPromptDoesNotAuthorizeResponseToolsOrDifferentInput(t *testing.T) {
	no := false
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{Enabled: true, CapturePersonalData: &no}}
	router := replayRuntimeRouter(t, cfg)
	for _, exact := range []bool{false, true} {
		ctx := replayPolicyRequest(t)
		input := "a different request"
		if exact {
			input = "Summarize the meeting agenda."
		}
		ctx.PIIContentVerified = true // This historical aggregate cannot authorize capture.
		ctx.PIIEvidence = []classification.PrivacyEvidence{classification.NewPrivacyEvidence("request", input, true, true)}
		router.startRouterReplay(ctx, "public", "model", "")
		router.attachRouterReplayResponse(ctx, []byte(`{"text":"Contact private@example.invalid"}`), true)
		record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
		if !ok || (record.Prompt != "") != exact {
			t.Fatal("prompt did not require exact input evidence")
		}
		if record.RequestBody != "" || record.ResponseBody != "" || record.ToolTrace != nil || record.ToolDefinitions != "" {
			t.Fatal("request evidence certified unscanned structured or response content")
		}
	}
}
