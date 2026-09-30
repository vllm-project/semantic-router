package extproc

import (
	"strconv"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestProtectionOnlyResponseFailuresReachRescueBeforeMinimumTurns(t *testing.T) {
	for _, boundary := range []string{"portable", "tool_loop", "nonportable"} {
		t.Run(boundary, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			router := &OpenAIRouter{Config: routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)}
			router.Config.RouterLearning.Protection.Tuning.MinTurnsBeforeSwitch = extprocIntPtr(4)
			refs := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			decision := &config.Decision{Name: "reliability"}
			ctx := routerLearningRequestContext("reliability-session", "conversation")
			ctx.VSRSelectedDecision = decision
			ctx.VSRSelectedDecisionName = decision.Name
			selCtx := &selection.SelectionContext{CandidateModels: refs, DecisionName: decision.Name}
			base := &selection.SelectionResult{
				SelectedModel: "cheap", Score: .9, Method: selection.MethodStatic,
				AllScores: map[string]float64{"cheap": .9, "frontier": .8},
			}
			initialCtx, initialResult, initialRef, _, initialErr := router.applyRouterLearning(selCtx, base, &refs[0], ctx)
			if initialErr != nil {
				t.Fatal(initialErr)
			}
			stageAgenticSessionDecision(initialCtx, initialResult, initialRef, ctx)
			if err := commitAgenticSessionDecision(ctx); err != nil {
				t.Fatal(err)
			}
			ctx.RequestModel = initialRef.Model
			base.SelectedModel = "frontier"
			base.AllScores = map[string]float64{"cheap": .8, "frontier": .9}

			for index, status := range []int{429, 503} {
				processLearningFailureHeaders(t, router, ctx, status)
				experience := router.routerLearningRuntimeState().experienceSnapshot(decision.Name, 0, "cheap")
				if experience.FailedCount != index+1 {
					t.Fatalf("actual response %d: failures=%d, want %d", status, experience.FailedCount, index+1)
				}
				if index == 0 {
					_, result, selected, _, stepErr := router.applyRouterLearning(selCtx, base, &refs[1], ctx)
					if stepErr != nil {
						t.Fatal(stepErr)
					}
					if selected.Model != "cheap" || result.SessionPolicy == nil || !result.SessionPolicy.HardLocked {
						t.Fatalf("one failure must not manufacture rescue before minimum turns: %+v", result)
					}
				}
			}
			// Protection observations do not turn on adaptive quality or usage updates.
			router.observeRouterLearningUsageTelemetry(ctx, time.Second, responseUsageMetrics{promptTokens: 10}, routerreplay.UsageCost{})
			experience := router.routerLearningRuntimeState().experienceSnapshot(decision.Name, 0, "cheap")
			if experience.GoodFitCount != 0 || experience.UnderpoweredCount != 0 || experience.OverprovisionedCount != 0 ||
				experience.LatencyEWMA != 0 || experience.CacheWriteEWMA != 0 {
				t.Fatalf("protection-only failure observation mutated adaptive evidence: %+v", experience)
			}
			switch boundary {
			case "tool_loop":
				ctx.VSRConversationFacts = classification.ConversationFacts{LastAssistantToolCall: true}
			case "nonportable":
				ctx.PreviousResponseID = "opaque-provider-state"
			}
			_, result, selected, _, learningErr := router.applyRouterLearning(selCtx, base, &refs[1], ctx)
			if learningErr != nil {
				t.Fatal(learningErr)
			}
			policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection)
			if !ok {
				t.Fatal("missing protection diagnostics")
			}
			if boundary == "portable" {
				if selected.Model != "frontier" || policy.Action != routerLearningActionRescueSwitch || result.SessionPolicy.MemoryTurnCount != 1 {
					t.Fatalf("two backend failures must reach portable rescue before min turns: %+v", policy.ToMap())
				}
			} else if selected.Model != "cheap" || !result.SessionPolicy.HardLocked || policy.Action == routerLearningActionRescueSwitch {
				t.Fatalf("failure evidence transferred hard request ownership: %+v", policy.ToMap())
			}
			if _, exists := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation); exists {
				t.Fatal("failure observations enabled adaptation")
			}
		})
	}
}

func TestProviderFailureObservationRespectsComponentModes(t *testing.T) {
	for _, test := range []struct {
		name           string
		masterOff      bool
		protectionOff  bool
		adaptationOn   bool
		adaptationMode string
		protectionMode string
		missingID      bool
		status         int
		want           int
	}{
		{name: "protection only", status: 503, want: 1},
		{name: "adaptation bypass protection apply", adaptationOn: true, adaptationMode: "bypass", status: 503, want: 1},
		{name: "observe protection", protectionMode: "observe", status: 429, want: 1},
		{name: "protection bypass", protectionMode: "bypass", status: 503},
		{name: "missing conversation identity", missingID: true, status: 503},
		{name: "both components off", protectionOff: true, status: 503},
		{name: "master off", masterOff: true, status: 503},
		{name: "ordinary client error", status: 400},
		{name: "successful response", status: 200},
		{name: "existing adaptation without identity", adaptationOn: true, protectionOff: true, missingID: true, status: 503, want: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)
			cfg.RouterLearning.Enabled = !test.masterOff
			cfg.RouterLearning.Adaptation.Enabled = extprocBoolPtr(test.adaptationOn)
			cfg.RouterLearning.Protection.Enabled = extprocBoolPtr(!test.protectionOff)
			router := &OpenAIRouter{Config: cfg}
			ctx := routerLearningRequestContext("failure-session", "conversation")
			if test.missingID {
				delete(ctx.Headers, "x-conversation-id")
			}
			ctx.RequestModel = "cheap"
			ctx.VSRSelectedDecisionName = "reliability"
			ctx.VSRSelectedDecision = &config.Decision{Name: "reliability", Adaptations: config.DecisionAdaptationsConfig{
				Adaptation: &config.DecisionLearningAdaptationConfig{Mode: test.adaptationMode},
				Protection: &config.DecisionLearningProtectionConfig{Mode: test.protectionMode},
			}}
			processLearningFailureHeaders(t, router, ctx, test.status)
			if got := router.routerLearningRuntimeState().experienceSnapshot("reliability", 0, "cheap").FailedCount; got != test.want {
				t.Fatalf("failures=%d, want %d", got, test.want)
			}
		})
	}
}

func processLearningFailureHeaders(t *testing.T, router *OpenAIRouter, ctx *RequestContext, status int) {
	t.Helper()
	response, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", RawValue: []byte(strconv.Itoa(status))},
		}}},
	}, ctx)
	if err != nil || response.GetResponseHeaders() == nil || ctx.UpstreamStatusCode != status {
		t.Fatalf("response-header processing failed: status=%d response=%v err=%v", ctx.UpstreamStatusCode, response, err)
	}
}
