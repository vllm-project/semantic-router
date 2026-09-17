package extproc

import (
	"encoding/json"
	"net/url"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/redaction"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestReplaySessionProtectionActualAndObservedRouting(t *testing.T) {
	for _, mode := range []string{config.DecisionAdaptationModeApply, config.DecisionAdaptationModeObserve, config.DecisionAdaptationModeBypass} {
		t.Run(mode, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			sessiontelemetry.ResetLastModelForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			t.Cleanup(sessiontelemetry.ResetLastModelForTesting)
			disabled := false
			cfg := &config.RouterConfig{}
			cfg.RouterLearning = config.RouterLearningConfig{
				Enabled:    true,
				Adaptation: config.RouterLearningAdaptationConfig{Enabled: &disabled},
				Protection: config.RouterLearningProtectionConfig{}, // default conversation scope
			}
			r := &OpenAIRouter{
				Config: cfg, ReplayStoreShared: true,
				ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
			}
			ctx := &RequestContext{
				SessionID: "synthetic-session", TurnIndex: 1,
				Headers:              map[string]string{"x-session-id": "synthetic-session", "x-conversation-id": "synthetic-conversation"},
				VSRSelectedDecision:  &config.Decision{Name: "route", Adaptations: config.DecisionAdaptationsConfig{Mode: mode}},
				VSRConversationFacts: classification.ConversationFacts{LastAssistantToolCall: true},
			}
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
				SessionID:     config.RoutingNamespaceKey("", "synthetic-session/synthetic-conversation"),
				SelectedModel: "alpha", Timestamp: time.Now(),
			})
			refs := []config.ModelRef{{Model: "alpha"}, {Model: "beta"}}
			selCtx := &selection.SelectionContext{CandidateModels: refs, DecisionName: "route", SessionID: ctx.SessionID}
			base := &selection.SelectionResult{
				SelectedModel: "beta", Method: selection.MethodStatic,
				AllScores: map[string]float64{"alpha": 0.2, "beta": 0.9},
			}
			_, _, selected, _ := r.applyRouterLearning(selCtx, base, &refs[1], ctx)
			wantModel, wantAction := "beta", replaySessionActionNone
			switch mode {
			case config.DecisionAdaptationModeApply:
				wantModel, wantAction = "alpha", replaySessionActionHardLock
			case config.DecisionAdaptationModeObserve:
				wantAction = replaySessionActionSwitch
			}
			if selected == nil || selected.Model != wantModel {
				t.Fatalf("actual selection = %+v, want %s", selected, wantModel)
			}
			record := buildReplayRoutingRecord(ctx, "public-model", selected.Model, "route")
			record.Timestamp = time.Now()
			id, err := r.ReplayRecorder.AddRecord(record)
			if err != nil {
				t.Fatal(err)
			}
			response := r.handleRouterReplayRecordAPI("GET", id).GetImmediateResponse()
			var persisted routerreplay.RoutingRecord
			if err := json.Unmarshal(response.Body, &persisted); err != nil {
				t.Fatal(err)
			}
			diagnostics := persisted.RouteDiagnostics
			if persisted.SelectedModel != wantModel || diagnostics.SelectedModel != wantModel || diagnostics.SessionAction != wantAction {
				t.Fatalf("replay misreports actual route: %+v", diagnostics)
			}
			if diagnostics.SessionPolicyApplied != (mode == config.DecisionAdaptationModeApply) {
				t.Fatalf("incorrect application flag: %+v", diagnostics)
			}
			if mode == config.DecisionAdaptationModeObserve {
				if diagnostics.HardLockReason != "" || diagnostics.SessionReason != "observe_only" {
					t.Fatalf("counterfactual hold attributed to actual route: %+v", diagnostics)
				}
				if persisted.Learning.Protection.SelectedModel != "alpha" || !persisted.Learning.Protection.HardLocked {
					t.Fatal("observed hold explanation must remain available")
				}
			}
			if mode == config.DecisionAdaptationModeApply {
				final := buildReplayRouteDiagnostics(ctx, "public-model", "beta", "route", 0, 0)
				if final.SelectedModel != "beta" || final.SessionAction != replaySessionActionSwitch || final.HardLockReason != "" {
					t.Fatalf("prior protection trace overwrote a later dispatch result: %+v", final)
				}
			}
		})
	}
}

func TestReplayTrajectorySeparatesRecipeSessionsAndKeepsEachHop(t *testing.T) {
	r := &OpenAIRouter{ReplayStoreShared: true, ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0))}
	now := time.Now()
	for index, recipe := range []string{"", "first", "other", "first"} {
		record := routerreplay.RoutingRecord{
			SessionID: "same-id", Recipe: recipe, TurnIndex: 1,
			Timestamp: now.Add(time.Duration(index) * time.Second), Decision: recipe + "-decision", SelectedModel: recipe + "-model",
			DurationMS: 17, LifecycleState: routerreplay.LifecycleCompleted, ResponseStatus: 200,
			ToolTrace: &routerreplay.ToolTrace{Steps: []routerreplay.ToolTraceStep{{Type: replayToolStepAssistantFinalResponse, Text: recipe + " synthetic output"}}},
		}
		if _, err := r.ReplayRecorder.AddRecord(record); err != nil {
			t.Fatal(err)
		}
	}
	for _, test := range []struct {
		query, recipe   string
		status, records int
	}{
		{"session_id=same-id", "", 400, 0},
		{"session_id=same-id&recipe=first", "first", 200, 2},
		{"session_id=same-id&recipe=", "", 200, 1},
		{"session_id=same-id&recipe=absent", "absent", 200, 0},
	} {
		response := r.handleRouterReplayTrajectoryAPI("GET", test.query).GetImmediateResponse()
		if int(response.Status.Code) != test.status {
			t.Fatalf("%s: status %d", test.query, response.Status.Code)
		}
		if test.status != 200 {
			continue
		}
		var trajectory routerReplayTrajectoryResponse
		if err := json.Unmarshal(response.Body, &trajectory); err != nil {
			t.Fatal(err)
		}
		if trajectory.Recipe != test.recipe || trajectory.RecordCount != test.records || len(trajectory.Routes) != test.records {
			t.Fatalf("wrong scoped trajectory: %+v", trajectory)
		}
		for _, route := range trajectory.Routes {
			if route.Decision != test.recipe+"-decision" || route.DurationMS != 17 {
				t.Fatalf("mixed route: %+v", route)
			}
		}
		for _, message := range trajectory.Messages {
			if message.Content != test.recipe+" synthetic output" {
				t.Fatalf("mixed message in recipe %q", test.recipe)
			}
		}
		redacted, _, err := redaction.RedactResponseBody(response.Body)
		if err != nil {
			t.Fatal(err)
		}
		var readonly routerReplayTrajectoryResponse
		if err := json.Unmarshal(redacted, &readonly); err != nil {
			t.Fatal(err)
		}
		if len(readonly.Routes) != test.records {
			t.Fatal("read-only response lost routing metadata")
		}
		for _, message := range readonly.Messages {
			if message.Content != "" {
				t.Fatal("read-only response leaked captured content")
			}
		}
	}
	query := url.Values{"session_id": {"unique"}}.Encode()
	if _, err := r.ReplayRecorder.AddRecord(routerreplay.RoutingRecord{SessionID: "unique", Recipe: "only"}); err != nil {
		t.Fatal(err)
	}
	if response := r.handleRouterReplayTrajectoryAPI("GET", query).GetImmediateResponse(); int(response.Status.Code) != 200 {
		t.Fatal("unambiguous legacy trajectory request should remain supported")
	}
}

func TestReplayProtectionDefaultConversationRequiresConfiguredIdentity(t *testing.T) {
	disabled := false
	cfg := &config.RouterConfig{}
	cfg.RouterLearning = config.RouterLearningConfig{
		Enabled:    true,
		Adaptation: config.RouterLearningAdaptationConfig{Enabled: &disabled},
	}
	r := &OpenAIRouter{Config: cfg}
	ctx := &RequestContext{
		SessionID: "synthetic", PreviousModel: "alpha",
		Headers:              map[string]string{"x-session-id": "synthetic"},
		VSRSelectedDecision:  &config.Decision{Name: "route"},
		VSRConversationFacts: classification.ConversationFacts{LastAssistantToolCall: true},
	}
	refs := []config.ModelRef{{Model: "alpha"}, {Model: "beta"}}
	selCtx := &selection.SelectionContext{CandidateModels: refs, DecisionName: "route"}
	base := &selection.SelectionResult{SelectedModel: "beta", Method: selection.MethodStatic}
	_, _, selected, _ := r.applyRouterLearning(selCtx, base, &refs[1], ctx)
	diagnostics := buildReplayRouteDiagnostics(ctx, "public-model", selected.Model, "route", 0, 0)
	if selected.Model != "beta" || diagnostics.SessionPolicyApplied || diagnostics.SessionAction != replaySessionActionNone || diagnostics.SessionReason != "missing_identity" {
		t.Fatalf("missing conversation header must not claim protection: %+v", diagnostics)
	}
}
