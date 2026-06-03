package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessionbudget"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func budgetRouter(enabled bool, budget int64) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			SessionTokenBudget: config.SessionTokenBudgetConfig{
				Enabled:      enabled,
				BudgetTokens: budget,
			},
		},
	}
}

func seedCumulative(sessionID string, prompt, completion int) {
	sessiontelemetry.RecordSessionUsage(sessiontelemetry.SessionUsageParams{
		SessionID:        sessionID,
		Model:            "test-model",
		PromptTokens:     prompt,
		CompletionTokens: completion,
	})
}

func TestEvaluateSessionBudgetDisabledIsNoOp(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	seedCumulative("s-disabled", 9000, 9000)

	r := budgetRouter(false, 1000)
	ctx := &RequestContext{SessionID: "s-disabled"}
	r.evaluateSessionBudget(ctx)

	if ctx.SessionBudget != nil {
		t.Fatalf("disabled budget must be a no-op, got %+v", ctx.SessionBudget)
	}
}

func TestEvaluateSessionBudgetZeroBudgetIsNoOp(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	seedCumulative("s-zero", 9000, 9000)

	r := budgetRouter(true, 0)
	ctx := &RequestContext{SessionID: "s-zero"}
	r.evaluateSessionBudget(ctx)

	if ctx.SessionBudget != nil {
		t.Fatalf("zero budget must be a no-op, got %+v", ctx.SessionBudget)
	}
}

func TestEvaluateSessionBudgetEmptySessionIsNoOp(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)

	r := budgetRouter(true, 1000)
	ctx := &RequestContext{SessionID: ""}
	r.evaluateSessionBudget(ctx)

	if ctx.SessionBudget != nil {
		t.Fatalf("empty session id must be a no-op, got %+v", ctx.SessionBudget)
	}
}

func TestEvaluateSessionBudgetUnderBudget(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	seedCumulative("s-under", 300, 200) // cumulative 500 < 1000

	r := budgetRouter(true, 1000)
	ctx := &RequestContext{SessionID: "s-under"}
	r.evaluateSessionBudget(ctx)

	if ctx.SessionBudget == nil {
		t.Fatal("enabled budget must populate SessionBudget even when under budget")
	}
	if ctx.SessionBudget.Stage != sessionbudget.StageNone {
		t.Fatalf("stage = %v, want StageNone", ctx.SessionBudget.Stage)
	}
	if ctx.SessionBudget.Cumulative != 500 {
		t.Fatalf("cumulative = %d, want 500", ctx.SessionBudget.Cumulative)
	}
}

func TestEvaluateSessionBudgetTerminate(t *testing.T) {
	sessiontelemetry.ResetForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	seedCumulative("s-over", 2000, 1500) // cumulative 3500, budget 1000 => ratio 3.5

	r := budgetRouter(true, 1000)
	ctx := &RequestContext{SessionID: "s-over"}
	r.evaluateSessionBudget(ctx)

	if ctx.SessionBudget == nil {
		t.Fatal("expected SessionBudget to be populated")
	}
	if ctx.SessionBudget.Stage != sessionbudget.StageTerminate {
		t.Fatalf("stage = %v, want StageTerminate", ctx.SessionBudget.Stage)
	}
	if ctx.SessionBudget.Ratio != 3.5 {
		t.Fatalf("ratio = %v, want 3.5", ctx.SessionBudget.Ratio)
	}
}
