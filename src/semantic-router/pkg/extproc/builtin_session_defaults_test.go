package extproc

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Read the shipped defaults rather than enabling protection in a test helper.
// Synthetic selector scores isolate continuity from model-quality measurement.
func TestBuiltinSessionDefaultsProtectPortableBoundaries(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "built-in", "latest", "mom-v1", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, conversation, want                     string
		tool, masterOff, protectionOff, excludeOwner bool
	}{
		{name: "active tool keeps owner", conversation: "task-a", tool: true, want: "frontier"},
		{name: "completed cycle allows better proposal", conversation: "task-a", want: "cheap"},
		{name: "new conversation has its own baseline", conversation: "task-b", want: "cheap"},
		{name: "missing conversation leaves base choice", tool: true, want: "cheap"},
		{name: "master false disables hold", conversation: "task-a", tool: true, masterOff: true, want: "cheap"},
		{name: "protection false disables hold", conversation: "task-a", tool: true, protectionOff: true, want: "cheap"},
		{name: "hard owner outside eligible pool rejects", conversation: "task-a", tool: true, excludeOwner: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			cfg, parseErr := config.ParseYAMLBytes(data)
			if parseErr != nil {
				t.Fatal(parseErr)
			}
			recipe, ok := cfg.RecipeByName("speed")
			if !ok {
				t.Fatal("speed recipe is missing")
			}
			cfg = cfg.ConfigForRecipe(recipe)
			if cfg.RouterLearning.Adaptation.EffectiveEnabled() {
				t.Fatal("builtin default must not enable online adaptation")
			}
			defaults := protectionSelectionConfig(cfg.RouterLearning.Protection)
			if defaults.MinTurnsBeforeSwitch != 1 || defaults.IdleTimeoutSeconds != 300 {
				t.Fatalf("unexpected builtin tuning defaults: %+v", defaults)
			}
			if test.masterOff {
				cfg.RouterLearning.Enabled = false
			}
			if test.protectionOff {
				cfg.RouterLearning.Protection.Enabled = extprocBoolPtr(false)
			}
			cfg.ModelConfig = map[string]config.ModelParams{"cheap": {}, "frontier": {}}
			decision := &cfg.Decisions[0] // Speed's declared tools path.
			ctx := routerLearningRequestContext("session-a", test.conversation)
			ctx.Routing.SelectRecipe(recipe)
			ctx.VSRSelectedDecision = decision
			ctx.TurnIndex = 1
			ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: test.tool}
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
				SessionID:     config.RoutingNamespaceKey("speed", "session-a/task-a"),
				SelectedModel: "frontier", DecisionName: config.RoutingNamespaceKey("speed", decision.Name),
				TurnIndex: 0, Timestamp: time.Now(),
			})
			registry := selection.NewRegistry()
			registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
				SelectedModel: "cheap", Score: 1, Confidence: 1, Method: selection.MethodStatic,
				AllScores: map[string]float64{"cheap": 1, "frontier": 0},
			}})
			router := &OpenAIRouter{Config: cfg, ModelSelector: registry}
			candidates := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			if test.excludeOwner {
				candidates = []config.ModelRef{{Model: "cheap"}, {Model: "backup"}}
			}
			selected, _, selectErr := router.selectModelFromCandidates(&selection.SelectionContext{
				SessionID: "session-a", RecipeName: "speed", DecisionName: decision.Name, CandidateModels: candidates,
			}, nil, ctx)
			if test.want == "" {
				if selected != nil || !errors.Is(selectErr, selection.ErrNoEligibleCandidates) {
					t.Fatalf("ownership must not bypass eligibility: selected=%+v err=%v", selected, selectErr)
				}
				return
			}
			if selectErr != nil || selected == nil || selected.Model != test.want {
				t.Fatalf("selected=%+v err=%v, want %s", selected, selectErr, test.want)
			}
			if test.tool && test.want == "frontier" && (ctx.VSRLearningPolicy == nil || !ctx.VSRLearningPolicy.toReplayProtection().ActiveToolLoop) {
				t.Fatalf("missing actual tool-loop protection trace: %+v", ctx.VSRLearningPolicy)
			}
		})
	}
}
