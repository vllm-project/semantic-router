package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestShadowGroupNamedRecipeDoesNotLeakReplayOutcomes(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primary := newShadowTestRouter(t, backend)
	router.Config.ModelConfig["second"] = config.ModelParams{PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI}
	route := config.Decision{
		Name:      shadowTestDecision,
		ModelRefs: []config.ModelRef{{Model: primary}},
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginRouterReplay,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true, "shadow": map[string]interface{}{"models": []string{shadowTestModel, "second"}},
			}),
		}},
	}
	router.Config.RouterReplay = config.RouterReplayConfig{StoreBackend: "memory"}
	router.Config.GlobalPlugins = []config.DecisionPlugin{{Type: config.DecisionPluginRouterReplay, Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true})}}
	router.Config.Recipes = []config.RoutingRecipe{
		{Name: "private", Profile: config.RoutingProfile{Decisions: []config.Decision{route}}},
		{Name: "other", Profile: config.RoutingProfile{Decisions: []config.Decision{route}}},
		{Name: config.DefaultRecipeName},
	}
	withGlobalPlugins(t, router.Config)
	var err error
	router.ReplayRecorders, router.ReplayRecorder, router.ReplayStoreShared, err = createReplayRuntime(router.Config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = router.ShadowDispatcher.Close()
		if err := closeReplayRecorders(router.ReplayRecorder, router.ReplayRecorders, router.ReplayStoreShared); err != nil {
			t.Error(err)
		}
	})

	request := testNeutralRequest("virtual", "please shadow this prompt")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.Routing.SelectRecipe(&router.Config.Recipes[0])
	router.applyDecisionResultToContext(&decision.DecisionResult{Decision: &route}, ctx)
	// Let the routing caller resolve the recorder and assign the replay ID.
	if _, err := router.handleEntrypointModelRouting(
		request, "virtual", route.Name, entropy.ReasoningDecision{}, primary, ctx,
	); err != nil {
		t.Fatal(err)
	}
	waitForShadow(t, router)
	if ctx.RouterReplayID == "" || backend.requestCount() != 2 {
		t.Fatalf("replay ID=%q, backend calls=%d; want an assigned ID and both candidates", ctx.RouterReplayID, backend.requestCount())
	}
	for name, recorder := range map[string]*routerreplay.Recorder{
		"global": router.ReplayRecorder,
		"other":  router.ReplayRecorders["other::shadow-route"],
	} {
		if recorder == nil {
			t.Fatalf("missing %s recorder", name)
		}
		if err := recorder.DrainOutcomes(); err != nil {
			t.Fatal(err)
		}
		if records := recorder.ListAllRecords(); len(records) != 0 {
			t.Errorf("%s recorder received %d records from the private recipe", name, len(records))
		}
	}
	recorder := router.ReplayRecorders["private::shadow-route"]
	if err := recorder.DrainOutcomes(); err != nil {
		t.Fatal(err)
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok {
		t.Fatalf("private recorder missing router-assigned replay ID %q", ctx.RouterReplayID)
	}
	if record.Recipe != "private" || record.RequestID != ctx.RequestID || len(record.Outcomes) != 2 {
		t.Fatalf("unexpected private replay record: %+v", record)
	}
	models := map[string]int{}
	for _, outcome := range record.Outcomes {
		if outcome.Source != shadowDispatchOutcomeSource || outcome.Verdict != shadowVerdictCompleted || outcome.Metadata["recipe"] != "private" {
			t.Fatalf("unexpected shadow outcome: %+v", outcome)
		}
		models[outcome.Metadata["shadow_model"]]++
	}
	if models[shadowTestModel] != 1 || models["second"] != 1 {
		t.Fatalf("shadow outcomes by model=%v, want one per candidate", models)
	}
}

func TestShadowGroupSlowReplayDoesNotHoldExecutionSlots(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primary := newShadowTestRouter(t, backend)
	router.Config.ModelConfig["second"] = config.ModelParams{PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI}
	cfg := shadowTestPluginConfig()
	cfg.Models = append(cfg.Models, "second")
	cfg.Execution.MaxConcurrency = shadowTestInt(1)
	slow := &slowReceiptStore{Storage: store.NewMemoryStore(10, 0), entered: make(chan struct{}, 1), release: make(chan struct{})}
	recorder := routerreplay.NewRecorder(slow)
	var once sync.Once
	release := func() { once.Do(func() { close(slow.release) }) }
	defer func() {
		release()
		_ = router.ShadowDispatcher.Close()
		_ = recorder.DrainOutcomes()
	}()
	baseline := runShadowRequest(t, router, primary, nil, nil)
	run := runShadowRequest(t, router, primary, cfg, func(ctx *RequestContext) {
		id, err := recorder.AddRecord(routerreplay.RoutingRecord{RequestID: ctx.RequestID})
		if err != nil {
			t.Fatal(err)
		}
		ctx.RouterReplayRecorder, ctx.RouterReplayID = recorder, id
	})
	if !bytes.Equal(baseline.body, run.body) {
		t.Fatal("shadow changed the primary request")
	}
	requireReceiptSignal(t, slow.entered)
	if !router.ShadowDispatcher.waitIdle(time.Second) {
		t.Fatalf("slow Replay held candidate execution: %d of 2 model calls completed", backend.requestCount())
	}
	if backend.requestCount() != 2 {
		t.Fatalf("model calls=%d, want 2", backend.requestCount())
	}
	release()
	if err := recorder.DrainOutcomes(); err != nil {
		t.Fatal(err)
	}
	run.recorder = recorder
	if outcomes := shadowOutcomes(t, run); len(outcomes) != 2 {
		t.Fatalf("outcomes=%d, want 2", len(outcomes))
	}
}

func TestShadowGroupReservesReplayCapacityBeforeCallingModels(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primary := newShadowTestRouter(t, backend)
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "occupied"}); err != nil {
		t.Fatal(err)
	}
	var held []*routerreplay.OutcomeReservation
	defer func() {
		for _, reservation := range held {
			reservation.Finish(routerreplay.Outcome{Verdict: "cancelled"})
		}
		_ = recorder.DrainOutcomes()
	}()
	for range routerreplay.DefaultOutcomeQueueCapacity {
		reservation := recorder.TryReserveOutcome("occupied")
		if reservation == nil {
			t.Fatal("could not reserve initial capacity")
		}
		held = append(held, reservation)
	}
	baseline := runShadowRequest(t, router, primary, nil, nil)
	run := runShadowRequest(t, router, primary, shadowTestPluginConfig(), func(ctx *RequestContext) {
		ctx.RouterReplayRecorder = recorder
	})
	waitForShadow(t, router)
	if !bytes.Equal(baseline.body, run.body) || backend.requestCount() != 0 {
		t.Fatalf("full audit queue must leave primary unchanged and skip shadow; model calls=%d", backend.requestCount())
	}
}

func TestShadowGroupQueuesAllModels(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primary := newShadowTestRouter(t, backend)
	router.Config.ModelConfig["second"] = config.ModelParams{PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI}
	cfg := shadowTestPluginConfig()
	cfg.Models = append(cfg.Models, "second")
	cfg.Execution.MaxConcurrency = shadowTestInt(1)
	baseline := runShadowRequest(t, router, primary, nil, nil)
	run := runShadowRequest(t, router, primary, cfg, nil)
	waitForShadow(t, router)
	if !bytes.Equal(baseline.body, run.body) {
		t.Fatal("shadow altered primary request")
	}
	if backend.requestCount() != 2 || len(shadowOutcomes(t, run)) != 2 {
		t.Fatalf("calls=%d outcomes=%d, want both candidates", backend.requestCount(), len(shadowOutcomes(t, run)))
	}
	var bodies [2]map[string]any
	for i := range bodies {
		if err := json.Unmarshal(backend.bodies[i], &bodies[i]); err != nil {
			t.Fatal(err)
		}
		delete(bodies[i], "model")
	}
	a, _ := json.Marshal(bodies[0])
	b, _ := json.Marshal(bodies[1])
	if !bytes.Equal(a, b) {
		t.Fatal("candidate inputs differ")
	}
}

func TestShadowGroupDisconnectCancelsAcceptedWork(t *testing.T) {
	for _, eof := range []bool{false, true} {
		t.Run(map[bool]string{false: "client_canceled", true: "incomplete_503_eof"}[eof], func(t *testing.T) {
			backend := newShadowTestBackend(t)
			started, release := make(chan struct{}), make(chan struct{})
			backend.setHandler(func(w http.ResponseWriter, _ []byte) {
				close(started)
				<-release
				writeShadowChatCompletion(w, shadowTestReply)
			})
			defer close(release)
			router, primary := newShadowTestRouter(t, backend)
			run := runShadowRequest(t, router, primary, shadowTestPluginConfig(), nil)
			select {
			case <-started:
			case <-time.After(time.Second):
				t.Fatal("candidate did not start")
			}
			if eof {
				// A 503 status and a partial body do not constitute a complete response.
				run.ctx.UpstreamStatusCode = http.StatusServiceUnavailable
				router.attachRouterReplayResponse(run.ctx, []byte(`{"error":`), false)
				responseBodyReplySent(run.ctx, nil, false)
				if err := router.handleProcessReceiveError(run.ctx, io.EOF); err != nil {
					t.Fatal(err)
				}
			} else {
				session := &routingSession{router: router, ctx: run.ctx}
				session.Close(context.Canceled)
			}
			finishRequestWork(run.ctx)
			if !router.ShadowDispatcher.waitIdle(time.Second) {
				t.Fatal("abandoned candidate kept running")
			}
			if outcomes := shadowOutcomes(t, run); len(outcomes) != 1 || outcomes[0].Verdict != shadowVerdictFailed || outcomes[0].Reason != "canceled" {
				t.Fatalf("outcomes=%v", outcomes)
			}
		})
	}
}

func TestShadowGroupNormalCompletionSurvivesEOF(t *testing.T) {
	for _, headersOnly := range []bool{false, true} {
		t.Run(map[bool]string{false: "body", true: "headers"}[headersOnly], func(t *testing.T) {
			backend := newShadowTestBackend(t)
			started, release := make(chan struct{}), make(chan struct{})
			defer close(release)
			backend.setHandler(func(w http.ResponseWriter, _ []byte) {
				close(started)
				<-release
				writeShadowChatCompletion(w, shadowTestReply)
			})
			router, primary := newShadowTestRouter(t, backend)
			run := runShadowRequest(t, router, primary, shadowTestPluginConfig(), nil)
			select {
			case <-started:
			case <-time.After(time.Second):
				t.Fatal("candidate did not start")
			}
			// Completion must work even when Replay is disabled for the request.
			run.ctx.RouterReplayID = ""
			if headersOnly {
				responseHeadersReplySent(run.ctx, nil, true)
			} else {
				responseBodyReplySent(run.ctx, nil, true)
			}
			if err := router.handleProcessReceiveError(run.ctx, io.EOF); err != nil {
				t.Fatal(err)
			}
			finishRequestWork(run.ctx)
			if router.ShadowDispatcher.waitIdle(30 * time.Millisecond) {
				t.Fatal("normal EOF canceled an unfinished candidate")
			}
		})
	}
}

func TestShadowGroupSurvivesPrimaryErrorResponse(t *testing.T) {
	for _, trailers := range []bool{false, true} {
		t.Run(map[bool]string{false: "body", true: "trailers"}[trailers], func(t *testing.T) {
			backend := newShadowTestBackend(t)
			started, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			backend.setHandler(func(w http.ResponseWriter, _ []byte) {
				close(started)
				<-release
				writeShadowChatCompletion(w, shadowTestReply)
			})
			router, primary := newShadowTestRouter(t, backend)
			run := runShadowRequest(t, router, primary, shadowTestPluginConfig(), nil)
			select {
			case <-started:
			case <-time.After(time.Second):
				t.Fatal("candidate did not start")
			}
			run.ctx.UpstreamStatusCode = http.StatusServiceUnavailable
			reply, err := router.responseBodyReply(&ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: []byte(`{"error":{"message":"unavailable"}}`), EndOfStream: !trailers},
			}, run.ctx)
			if err != nil {
				t.Fatal(err)
			}
			if router.ShadowDispatcher.waitIdle(30 * time.Millisecond) {
				t.Fatal("a complete primary error response canceled the candidate")
			}
			// A buffered body is complete even when trailers keep EndOfStream false.
			responseBodyReplySent(run.ctx, reply, !trailers)
			if err := router.handleProcessReceiveError(run.ctx, io.EOF); err != nil {
				t.Fatal(err)
			}
			finishRequestWork(run.ctx)
			unblock()
			waitForShadow(t, router)
			if outcomes := shadowOutcomes(t, run); len(outcomes) != 1 || outcomes[0].Verdict != shadowVerdictCompleted {
				t.Fatalf("shadow outcomes after primary 503: %v", outcomes)
			}
		})
	}
}

func TestShadowGroupSnapshotOwnsQueuedInput(t *testing.T) {
	backend := newShadowTestBackend(t)
	started, release := make(chan struct{}, 1), make(chan struct{})
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		select {
		case started <- struct{}{}:
		default:
		}
		<-release
		writeShadowChatCompletion(w, shadowTestReply)
	})
	router, primary := newShadowTestRouter(t, backend)
	cfg := shadowTestPluginConfig()
	cfg.Execution.MaxConcurrency = shadowTestInt(1)
	runShadowRequest(t, router, primary, cfg, nil)
	select {
	case <-started:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("candidate did not start")
	}
	queued := runShadowRequest(t, router, primary, cfg, nil)
	queued.ctx.SemanticRequest.Messages[0].Role = "system"
	close(release)
	waitForShadow(t, router)
	if backend.requestCount() != 2 {
		t.Fatal("queued candidate did not run")
	}
	var wire struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(backend.bodies[1], &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) == 0 || wire.Messages[0].Role != "user" {
		t.Fatalf("snapshot changed: %s", backend.bodies[1])
	}
}

func TestShadowGroupMissingUsageRemainsUnknown(t *testing.T) {
	backend := newShadowTestBackend(t)
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"no-usage","object":"chat.completion","model":"shadow-model","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}`))
	})
	router, primary := newShadowTestRouter(t, backend)
	run := runShadowRequest(t, router, primary, shadowTestPluginConfig(), nil)
	waitForShadow(t, router)
	out := singleShadowOutcome(t, run)
	if _, ok := out.Metadata["input_tokens"]; ok {
		t.Fatal("missing usage was recorded as measured zero")
	}
	if _, ok := out.Metadata["output_tokens"]; ok {
		t.Fatal("missing output usage was recorded as measured zero")
	}
}

func TestShadowGroupSiblingFailureIsIsolated(t *testing.T) {
	backend := newShadowTestBackend(t)
	var calls atomic.Int32
	backend.setHandler(func(w http.ResponseWriter, _ []byte) {
		if calls.Add(1) == 1 {
			http.Error(w, "failed", http.StatusServiceUnavailable)
			return
		}
		writeShadowChatCompletion(w, shadowTestReply)
	})
	router, primary := newShadowTestRouter(t, backend)
	router.Config.ModelConfig["second"] = config.ModelParams{PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI}
	cfg := shadowTestPluginConfig()
	cfg.Models = append(cfg.Models, "second")
	cfg.Execution.MaxConcurrency = shadowTestInt(1)
	run := runShadowRequest(t, router, primary, cfg, nil)
	waitForShadow(t, router)
	outcomes := shadowOutcomes(t, run)
	if len(outcomes) != 2 {
		t.Fatalf("outcomes=%d, want 2", len(outcomes))
	}
	counts := map[string]int{}
	for _, outcome := range outcomes {
		counts[outcome.Verdict]++
	}
	if counts[shadowVerdictFailed] != 1 || counts[shadowVerdictCompleted] != 1 {
		t.Fatalf("outcomes=%v", counts)
	}
}

func TestShadowGroupResponseBudgetRemainsChargedAfterFailure(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusOK} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			backend := newShadowTestBackend(t)
			var calls atomic.Int32
			backend.setHandler(func(w http.ResponseWriter, _ []byte) {
				if calls.Add(1) == 1 {
					w.WriteHeader(status)
					_, _ = w.Write([]byte(strings.Repeat("x", 513)))
					return
				}
				// Fits the whole group's 1024 bytes, but not this candidate's
				// 512-byte share. A failed first read must not refund its share.
				writeShadowChatCompletion(w, strings.Repeat("x", 400))
			})
			router, primary := newShadowTestRouter(t, backend)
			router.Config.ModelConfig["second"] = config.ModelParams{PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI}
			cfg := shadowTestPluginConfig()
			cfg.Models = append(cfg.Models, "second")
			cfg.Execution.MaxConcurrency = shadowTestInt(1)
			cfg.Budget.MaxResponseBytes = shadowTestInt(1024)
			baseline := runShadowRequest(t, router, primary, nil, nil)
			run := runShadowRequest(t, router, primary, cfg, nil)
			waitForShadow(t, router)
			outcomes := shadowOutcomes(t, run)
			if !bytes.Equal(baseline.body, run.body) || backend.requestCount() != 2 || len(outcomes) != 2 {
				t.Fatalf("calls=%d outcomes=%d; primary must remain unchanged", backend.requestCount(), len(outcomes))
			}
			reasons := map[string]int{}
			for _, outcome := range outcomes {
				if outcome.Verdict != shadowVerdictFailed || outcome.Metadata["attempts"] != "1" {
					t.Fatalf("failure or oversized response was retried or accepted: %+v", outcome)
				}
				reasons[outcome.Reason]++
			}
			wantOversized := 2
			if status == http.StatusBadRequest {
				wantOversized = 1
				if reasons[shadowReasonUpstreamStatus] != 1 {
					t.Fatalf("missing status failure: %v", reasons)
				}
			}
			if reasons[shadowReasonResponseTooLarge] != wantOversized {
				t.Fatalf("response budget was not retained: %v", reasons)
			}
		})
	}
}

func TestShadowGroupUsesReplayPolicyForUnmatchedRequests(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		name := "inherited"
		if !enabled {
			name = "recipe-disabled"
		}
		t.Run(name, func(t *testing.T) {
			backend := newShadowTestBackend(t)
			router, primary := newShadowTestRouter(t, backend)
			router.Config.ModelConfig["second"] = config.ModelParams{PreferredEndpoints: []string{"shadow-backend"}, APIFormat: config.APIFormatOpenAI}
			router.Config.RouterReplay = config.RouterReplayConfig{StoreBackend: "memory"}
			router.Config.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}}
			if !enabled {
				router.Config.Recipes[0].Profile.Plugins = []config.DecisionPlugin{replayTestPlugin(map[string]interface{}{"enabled": false})}
			}
			withGlobalPlugins(t, router.Config, replayTestPlugin(map[string]interface{}{"enabled": true, "shadow": map[string]interface{}{"models": []string{shadowTestModel, "second"}}}))
			var err error
			router.ReplayRecorders, router.ReplayRecorder, router.ReplayStoreShared, err = createReplayRuntime(router.Config)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				_ = router.ShadowDispatcher.Close()
				if err := closeReplayRecorders(router.ReplayRecorder, router.ReplayRecorders, router.ReplayStoreShared); err != nil {
					t.Error(err)
				}
			})
			request := testNeutralRequest("virtual", "unmatched request")
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.Routing.SelectRecipe(&router.Config.Recipes[0])
			if _, err := router.handleEntrypointModelRouting(request, "virtual", "", entropy.ReasoningDecision{}, primary, ctx); err != nil {
				t.Fatal(err)
			}
			waitForShadow(t, router)
			want := 0
			if enabled {
				want = 2
			}
			if got := backend.requestCount(); got != want {
				t.Fatalf("shadow calls=%d, want %d", got, want)
			}
			if enabled {
				outcomes := shadowOutcomes(t, shadowRun{ctx: ctx, recorder: ctx.RouterReplayRecorder, replayID: ctx.RouterReplayID})
				if len(outcomes) != 2 {
					t.Fatalf("shadow outcomes=%d, want 2", len(outcomes))
				}
			} else if ctx.RouterReplayID != "" {
				t.Fatal("disabled Replay recorded the request")
			}
		})
	}
}
