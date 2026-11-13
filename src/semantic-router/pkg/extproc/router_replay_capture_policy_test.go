package extproc

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
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

// Each prompt capture mode keeps its own part of a request and its answer,
// whatever the capture switches allow, and every mode keeps the routing data.
func TestReplayPromptCaptureModesKeepTheirRecords(t *testing.T) {
	const prompt = "Summarize the meeting agenda."
	for _, test := range []struct {
		name, mode              string
		read, detected          bool
		prompted, bodies, trace bool
	}{
		{"none", config.PromptCaptureNone, true, false, false, false, false},
		{"raw", config.PromptCaptureRaw, false, false, true, true, true},
		{"raw with personal data", config.PromptCaptureRaw, true, true, true, true, true},
		{"verified clean", config.PromptCaptureVerified, true, false, true, false, false},
		{"verified with personal data", config.PromptCaptureVerified, true, true, false, false, false},
		{"verified unread", config.PromptCaptureVerified, false, false, false, false, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(map[string]interface{}{
				"enabled": true, "prompt_capture": test.mode, "capture_request_body": true, "capture_response_body": true,
			}))
			router := replayRuntimeRouter(t, cfg)
			ctx := replayPolicyRequest(t)
			ctx.PIIContentVerified, ctx.PIIDetected = test.read, test.detected
			ctx.PIIEvidence = []classification.PrivacyEvidence{classification.NewPrivacyEvidence("request", prompt, test.read, !test.detected)}
			ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			router.attachRouterReplayResponse(ctx, []byte(`{"text":"Meeting agenda."}`), true)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !ok {
				t.Fatal("missing record")
			}
			if record.Decision != "ordinary" || record.SelectedModel != "model" || record.RequestID != ctx.RequestID ||
				record.PIIDetected != test.detected || record.LifecycleState != routerreplay.LifecycleCompleted {
				t.Fatalf("lost routing data: %+v", record)
			}
			if kept := record.Prompt != ""; kept != test.prompted || (kept && record.Prompt != prompt) {
				t.Fatalf("prompt %q, want kept=%v", record.Prompt, test.prompted)
			}
			if (record.RequestBody != "") != test.bodies || (record.ResponseBody != "") != test.bodies {
				t.Fatalf("request body %q, response body %q, want kept=%v", record.RequestBody, record.ResponseBody, test.bodies)
			}
			if (record.ToolTrace != nil) != test.trace {
				t.Fatalf("tool trace %+v, want kept=%v", record.ToolTrace, test.trace)
			}
			if ctx.SemanticRequest.Messages[0].Content[0].Text != prompt {
				t.Fatal("capture changed the provider request")
			}
		})
	}
}

// none and raw capture never read a prompt for personal data, even where
// another policy of the same configuration keeps prompts only once verified
// and so makes the pii rules ready for it. A verified request still is read.
func TestReplayNoneAndRawCaptureStartNoPromptRead(t *testing.T) {
	const prompt = "Summarize the meeting agenda."
	for _, test := range []struct {
		mode string
		read bool
	}{
		{config.PromptCaptureNone, false},
		{config.PromptCaptureRaw, false},
		{config.PromptCaptureVerified, true},
	} {
		t.Run(test.mode, func(t *testing.T) {
			cfg := &config.RouterConfig{
				RouterReplay:       config.RouterReplayConfig{StoreBackend: "memory"},
				IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}}},
			}
			withGlobalPlugins(t, cfg, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureVerified}))
			if !cfg.ReplayNeedsPIIEvidence() {
				t.Fatal("the global verified capture must make the pii rules ready")
			}
			router := replayRuntimeRouter(t, cfg)
			router.Classifier = &classification.Classifier{Config: cfg}
			var asked, read atomic.Int32
			done := make(chan struct{}, 1)
			savedEvidence, savedReadable := replayPromptEvidence, replayPromptReadable
			t.Cleanup(func() { replayPromptEvidence, replayPromptReadable = savedEvidence, savedReadable })
			replayPromptReadable = func(*classification.Classifier, string, time.Duration, bool) bool {
				asked.Add(1)
				return true
			}
			replayPromptEvidence = func(_ *classification.Classifier, _ context.Context, text string) classification.PrivacyEvidence {
				read.Add(1)
				done <- struct{}{}
				return classification.NewPrivacyEvidence("request", text, true, true)
			}
			ctx := replayPolicyRequest(t)
			ctx.Routing.SelectRecipe(cfg.DefaultRecipe())
			own := replayDecision(map[string]interface{}{"enabled": true, "prompt_capture": test.mode})
			router.applyDecisionResultToContext(&decision.DecisionResult{Decision: own}, ctx)
			router.startRouterReplay(ctx, "public-entry", "model", own.Name)
			if test.read {
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatal("the verified prompt was never read")
				}
			}
			deadline := time.Now().Add(5 * time.Second)
			for len(replayPromptCheckSlots) > 0 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if want := map[bool]int32{true: 1, false: 0}[test.read]; asked.Load() != want || read.Load() != want {
				t.Fatalf("prompt reads asked %d, run %d; want %d", asked.Load(), read.Load(), want)
			}
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if want := map[string]string{config.PromptCaptureNone: "", config.PromptCaptureRaw: prompt, config.PromptCaptureVerified: prompt}[test.mode]; !ok || record.Prompt != want {
				t.Fatalf("prompt %q, want %q", record.Prompt, want)
			}
		})
	}
}

func TestReplayGlobalRuntimeCapturesBeforeDecisionAndDirectRequests(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		for _, direct := range []bool{false, true} {
			cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(map[string]interface{}{"enabled": enabled, "prompt_capture": config.PromptCaptureVerified}))
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
	cfg := withGlobalPlugins(t, &config.RouterConfig{}, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureVerified}))
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
	allow := replayDecision(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureRaw})
	router.applyDecisionResultToContext(&decision.DecisionResult{Decision: allow}, ctx)
	router.startRouterReplay(ctx, "public-entry", "model", allow.Name)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.Prompt == "" || ctx.RouterReplayContentOmitted {
		t.Fatal("decision override did not restore content eligibility")
	}
}

// Verified capture is what capture_personal_data: false did, under either
// spelling: the record is written first, without the prompt, and the prompt
// is attached only once a read after routing finds it clean.
func TestReplayChecksAWithheldPromptAfterRoutingAndAttachesItOnlyWhenClean(t *testing.T) {
	const prompt = "Summarize the meeting agenda."
	for _, verified := range []map[string]interface{}{
		{"enabled": true, "prompt_capture": config.PromptCaptureVerified},
		{"enabled": true, "capture_personal_data": false},
	} {
		for _, clean := range []bool{true, false} {
			cfg := &config.RouterConfig{
				RouterReplay:       config.RouterReplayConfig{StoreBackend: "memory"},
				IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}}},
			}
			withGlobalPlugins(t, cfg, replayTestPlugin(verified))
			router := replayRuntimeRouter(t, cfg)
			router.Classifier = &classification.Classifier{Config: cfg}
			release, read := make(chan struct{}), make(chan string, 1)
			saved := replayPromptEvidence
			replayPromptEvidence = func(_ *classification.Classifier, _ context.Context, text string) classification.PrivacyEvidence {
				<-release
				read <- text
				return classification.NewPrivacyEvidence("request", text, true, clean)
			}
			ctx := replayPolicyRequest(t)
			ctx.Routing.SelectRecipe(cfg.DefaultRecipe())
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !ok || record.Prompt != "" || !ctx.RouterReplayContentOmitted {
				t.Fatalf("%v clean=%v: the record must be written before its check, prompt withheld: %+v", verified, clean, record)
			}
			close(release)
			if got := <-read; got != prompt {
				t.Fatalf("checked %q, want the exact prompt", got)
			}
			deadline := time.Now().Add(5 * time.Second)
			for len(replayPromptCheckSlots) > 0 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			replayPromptEvidence = saved
			record, _ = router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if want := map[bool]string{true: prompt, false: ""}[clean]; record.Prompt != want {
				t.Fatalf("%v clean=%v: prompt=%q want=%q", verified, clean, record.Prompt, want)
			}
			if record.RequestBody != "" || record.ToolDefinitions != "" {
				t.Fatalf("%v clean=%v: a prompt check must not restore other content: %+v", verified, clean, record)
			}
		}
	}
}

// No check starts that could not read the prompt whole in its time: it would
// end incomplete_read and withhold the prompt anyway. Whether it could is the
// classifier's call, told whether the request's own guard scans ran out of
// time on its input; scans that a match ended early did not. A check that
// can finish still runs, and still withholds a prompt with personal data.
func TestReplayStartsNoCheckThatCouldNotReadThePromptInTime(t *testing.T) {
	const prompt = "Summarize the meeting agenda."
	for _, test := range []struct {
		name                          string
		coverage                      map[string]string
		matched                       []string
		readable, guardShort, checked bool
	}{
		{"readable in time", nil, nil, true, false, true},
		{"too long for its time", nil, nil, false, false, false},
		{"guard scan cut by its deadline", map[string]string{"jailbreak:prompt_attack": "partial 43/97"}, nil, false, true, false},
		{"guard scan ended by a match", map[string]string{"jailbreak:prompt_attack": "partial 2/97"}, []string{"prompt_attack"}, true, false, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &config.RouterConfig{
				RouterReplay:       config.RouterReplayConfig{StoreBackend: "memory"},
				IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{PIIRules: []config.PIIRule{{Name: "personal_data"}}}},
			}
			withGlobalPlugins(t, cfg, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureVerified}))
			router := replayRuntimeRouter(t, cfg)
			router.Classifier = &classification.Classifier{Config: cfg}
			read := make(chan string, 1)
			savedEvidence, savedReadable := replayPromptEvidence, replayPromptReadable
			t.Cleanup(func() { replayPromptEvidence, replayPromptReadable = savedEvidence, savedReadable })
			var guardShort bool
			replayPromptReadable = func(_ *classification.Classifier, text string, within time.Duration, short bool) bool {
				if text != prompt || within != replayPromptCheckTimeout {
					t.Errorf("asked whether %q is readable in %v", text, within)
				}
				guardShort = short
				return test.readable
			}
			replayPromptEvidence = func(_ *classification.Classifier, _ context.Context, text string) classification.PrivacyEvidence {
				read <- text
				return classification.NewPrivacyEvidence("request", text, true, false)
			}
			ctx := replayPolicyRequest(t)
			ctx.Routing.SelectRecipe(cfg.DefaultRecipe())
			ctx.VSRScanCoverage, ctx.VSRMatchedJailbreak = test.coverage, test.matched
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			if guardShort != test.guardShort {
				t.Fatalf("told the guard scan ran out of time: %v, want %v", guardShort, test.guardShort)
			}
			select {
			case <-read:
				if !test.checked {
					t.Fatal("a check started that could not read the prompt in time")
				}
			case <-time.After(map[bool]time.Duration{true: 5 * time.Second, false: 50 * time.Millisecond}[test.checked]):
				if test.checked {
					t.Fatal("no check started for a prompt it could read in time")
				}
			}
			deadline := time.Now().Add(5 * time.Second)
			for len(replayPromptCheckSlots) > 0 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if record, _ := router.ReplayRecorder.GetRecord(ctx.RouterReplayID); record.Prompt != "" {
				t.Fatalf("the prompt was attached: %q", record.Prompt)
			}
		})
	}
}

func TestReplayRuntimeUsesGlobalMemoryCapacity(t *testing.T) {
	max := 1
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(map[string]interface{}{"enabled": true, "max_records": max}))
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
	for _, test := range []struct {
		name, mode string
		response   bool
	}{
		{"none", config.PromptCaptureNone, true},
		{"verified", config.PromptCaptureVerified, true},
		{"raw", config.PromptCaptureRaw, true},
		{"response body off", config.PromptCaptureRaw, false},
	} {
		capture := test.mode == config.PromptCaptureRaw && test.response
		t.Run(test.name, func(t *testing.T) {
			cfg := withGlobalPlugins(t, &config.RouterConfig{}, replayTestPlugin(map[string]interface{}{
				"enabled": true, "prompt_capture": test.mode, "capture_response_body": test.response,
			}))
			router := replayRuntimeRouter(t, cfg)
			ctx := replayPolicyRequest(t)
			ctx.VSRSelectedDecision = &config.Decision{Name: "ordinary", Plugins: cfg.DefaultPlugins}
			router.startRouterReplay(ctx, "public-entry", "model", "ordinary")
			key := signalKey(config.SignalTypeHallucination, "ungrounded")
			observed := &classification.ResponseSignal{
				Matched: map[string][]string{config.SignalTypeHallucination: {"ungrounded"}},
				Values:  map[string]float64{key: 0.9},
				Spans:   map[string][]modelservice.Span{key: {{Label: "unsupported", Text: "private@example.invalid", Probability: 0.9}}},
			}
			ctx.ResponseGuard = &responseGuardRun{conditions: observedConditions{observed: observed}}
			router.updateRouterReplayHallucinationStatus(ctx, router.responseGuardReplayRecorder(ctx))
			record, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			if !found || !record.HallucinationDetected || !record.HallucinationScoreAvailable || record.HallucinationConfidence != 0.9 {
				t.Fatal("the hallucination verdict or score metadata was lost")
			}
			if (len(record.HallucinationSpans) > 0) != capture {
				t.Fatalf("response excerpts did not follow capture policy: capture=%t", capture)
			}
			if len(observed.Spans[key]) != 1 {
				t.Fatal("Replay changed the response-stage observation")
			}
		})
	}
}

func TestReplayDirectStartResolvesGlobalPolicyWithoutDecisionPreparation(t *testing.T) {
	router := replayRuntimeRouter(t, withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureVerified})))
	ctx := replayPolicyRequest(t)
	ctx.Routing.SelectPassthrough()
	router.startRouterReplay(ctx, "backend", "backend", "")
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.Prompt != "" || record.RequestBody != "" {
		t.Fatal("direct request did not resolve global metadata-only capture")
	}
}

func TestReplayQueriesIncludeGlobalAndDecisionMemoryStores(t *testing.T) {
	cfg := &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}, IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{*replayDecision(map[string]interface{}{})}}}
	withGlobalPlugins(t, cfg, replayTestPlugin(map[string]interface{}{"enabled": true}))
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
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureRaw, "capture_response_body": no}))
	router := replayRuntimeRouter(t, cfg)
	global := replayPolicyRequest(t)
	router.startRouterReplay(global, "direct", "direct", "")
	selected := replayPolicyRequest(t)
	d := replayDecision(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureRaw, "capture_response_body": true})
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
	cfg := withGlobalPlugins(t, &config.RouterConfig{}, replayTestPlugin(map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureVerified}))
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

// A recipe that enables Replay in its own plugins records its requests that
// are rejected before a decision, on the memory store as on a shared one.
func TestReplayRecipeDefaultsCaptureBeforeDecisionOnMemoryStore(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"},
		Recipes: []config.RoutingRecipe{
			{Name: "plain", Profile: config.RoutingProfile{
				Decisions: []config.Decision{{Name: "route", ModelRefs: []config.ModelRef{{Model: "m"}}}},
			}},
			{Name: "audited", Profile: config.RoutingProfile{
				Plugins:   []config.DecisionPlugin{replayTestPlugin(map[string]interface{}{"enabled": true})},
				Decisions: []config.Decision{{Name: "route", ModelRefs: []config.ModelRef{{Model: "m"}}}},
			}},
		},
	}
	withGlobalPlugins(t, cfg)
	router := replayRuntimeRouter(t, cfg)
	for i, recorded := range []bool{false, true} {
		ctx := replayPolicyRequest(t)
		ctx.Routing.SelectRecipe(&cfg.Recipes[i])
		router.respondDecisionUnresolved(ctx, "public-entry", &decision.DecisionUnresolvedError{Decision: "unknown"})
		if (ctx.RouterReplayID != "") != recorded {
			t.Fatalf("recipe %s: replay id %q, want recorded=%t", cfg.Recipes[i].Name, ctx.RouterReplayID, recorded)
		}
	}
	page, err := router.queryRouterReplayPage(routerReplayListQuery{limit: 10})
	if err != nil || len(page.Data) != 1 {
		t.Fatalf("rejected request not queryable: %+v err=%v", page, err)
	}
	record := page.Data[0]
	if record.Recipe != "audited" || record.Decision != "" || record.ResponseStatus != 503 || record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("record = recipe %q decision %q status %d lifecycle %q", record.Recipe, record.Decision, record.ResponseStatus, record.LifecycleState)
	}
}
