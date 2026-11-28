package turnselection

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type stubScorer struct {
	fn    func(ctx context.Context, query string, texts []string) ([]float64, error)
	calls int
	texts []string
}

func (s *stubScorer) Score(ctx context.Context, query string, texts []string) ([]float64, error) {
	s.calls++
	s.texts = append([]string(nil), texts...)
	return s.fn(ctx, query, texts)
}

func survivingTurnTexts(req *llmprotocol.Request) []string {
	out := make([]string, 0, len(req.Messages))
	for _, msg := range req.Messages {
		for _, c := range msg.Content {
			if c.Kind == llmprotocol.ContentText && c.Text != "" {
				out = append(out, c.Text)
			}
		}
	}
	return out
}

func containsSubstring(items []string, needle string) bool {
	for _, item := range items {
		if strings.Contains(item, needle) {
			return true
		}
	}
	return false
}

func runSelectionOnCopy(
	t *testing.T,
	base *llmprotocol.Request,
	prov contextcompression.Provenance,
	policy Policy,
	input RequestInput,
) (*llmprotocol.Request, Diagnostics, error) {
	t.Helper()
	cloned := cloneRequest(t, base)
	ir := contextcompression.ParseSemanticRequest(cloned, prov)
	if input.CostEstimator == nil && input.TokenCounter == nil {
		input.CostEstimator = NewSemanticCostEstimator(cloned, nil)
	}
	action := NewAction(policy, input)
	err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()})
	diag := action.Reconcile(ir.Transformations.Receipts())
	return cloned, diag, err
}

func cloneRequest(t *testing.T, req *llmprotocol.Request) *llmprotocol.Request {
	t.Helper()
	snap := contextcompression.CaptureHistory(req)
	conv := snap.Conversation()
	out := *req
	out.Instructions = conv.Instructions
	out.Messages = conv.Messages
	if req.Tools != nil {
		out.Tools = append([]llmprotocol.Tool(nil), req.Tools...)
	}
	if req.Metadata != nil {
		out.Metadata = make(map[string]string, len(req.Metadata))
		for k, v := range req.Metadata {
			out.Metadata[k] = v
		}
	}
	return &out
}

func TestBaselineComparisonDistantFactVsRecentChatter(t *testing.T) {
	base := &llmprotocol.Request{
		Model: "sql-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "Our PostgreSQL database schema has orders(id, customer_id, total_cents) with 2GB RAM."),
			textMsg(llmprotocol.RoleAssistant, "Noted: orders schema with customer_id foreign key and 2GB memory limit."),
			textMsg(llmprotocol.RoleUser, "What is the weather forecast for Seattle this weekend?"),
			textMsg(llmprotocol.RoleAssistant, "Expect light rain on Saturday and sunny skies on Sunday."),
			textMsg(llmprotocol.RoleUser, "Can you recommend a movie for Friday evening?"),
			textMsg(llmprotocol.RoleAssistant, "Try a classic mystery thriller with short runtime."),
			textMsg(llmprotocol.RoleUser, "Also remind me to pick up groceries tomorrow morning."),
			textMsg(llmprotocol.RoleAssistant, "Sure, remember to grab milk, bread, and fresh fruit."),
			textMsg(llmprotocol.RoleUser, "Write an indexed SQL query over orders grouped by customer_id within our 2GB RAM limit."),
		},
	}

	// Allow at most 2 prior turns so the selector must choose between distant
	// database facts and recent unrelated chatter.
	makePolicy := func(strategy Strategy) Policy {
		p := DefaultPolicy()
		p.Strategy = strategy
		p.Budget.MaxPriorTurns = 2
		p.Budget.MinRecentTurns = 1
		return p
	}

	fullReq, fullDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, makePolicy(StrategyFull), RequestInput{})
	if err != nil {
		t.Fatalf("full baseline failed: %v", err)
	}
	if len(fullReq.Messages) != len(base.Messages) || fullDiag.BudgetStatus != BudgetUnsatisfiable {
		t.Fatalf("full baseline = %d msgs, diag=%+v", len(fullReq.Messages), fullDiag)
	}

	recReq, recDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, makePolicy(StrategyRecency), RequestInput{})
	if err != nil {
		t.Fatalf("recency baseline failed: %v", err)
	}
	recTexts := survivingTurnTexts(recReq)
	if recDiag.BudgetStatus != BudgetSatisfied || recDiag.RemovedTurns != 2 {
		t.Fatalf("recency diag = %+v", recDiag)
	}
	if containsSubstring(recTexts, "orders(id, customer_id") {
		t.Fatalf("recency baseline unexpectedly kept oldest turn: %v", recTexts)
	}

	relReq, relDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, makePolicy(StrategyRelevance), RequestInput{})
	if err != nil {
		t.Fatalf("relevance selector failed: %v", err)
	}
	relTexts := survivingTurnTexts(relReq)
	if relDiag.BudgetStatus != BudgetSatisfied || relDiag.RemovedTurns != 2 || relDiag.MinRecentRetained != 1 {
		t.Fatalf("relevance diag = %+v", relDiag)
	}
	if !containsSubstring(relTexts, "orders(id, customer_id") {
		t.Fatalf("relevance selector lost distant database schema turn: %v", relTexts)
	}
	if !containsSubstring(relTexts, "pick up groceries") {
		t.Fatalf("relevance selector lost minimum recent turn: %v", relTexts)
	}
	if containsSubstring(relTexts, "weather forecast") || containsSubstring(relTexts, "recommend a movie") {
		t.Fatalf("relevance selector failed to drop unrelated chatter: %v", relTexts)
	}
	// Chronological order: distant DB turn must still precede recent turn and live turn.
	if !strings.Contains(relReq.Messages[0].Content[0].Text, "PostgreSQL") ||
		!strings.Contains(relReq.Messages[2].Content[0].Text, "groceries") ||
		!strings.Contains(relReq.Messages[4].Content[0].Text, "indexed SQL") {
		t.Fatalf("chronological order broken: %v", relTexts)
	}
}

// TestPairedOptionCorrectionContinuityRegression implements the contrast fixture
// suggested on issue #3344: an older turn introduces two options, a later turn
// corrects or rejects one, and the live turn refers to "the second option"
// without repeating its name. Two paired histories differ ONLY in which option
// is corrected, keeping option names and nearly all tokens identical.
//
// The test verifies that:
//   - full-history exceeds the budget;
//   - recency-only drops the older turn that defines the options;
//   - lexical-only (continuity disabled) retains the option definitions but
//     drops the superseding correction turn (restoring a stale interpretation);
//   - relevance+continuity preserves BOTH the option definitions AND the
//     superseding correction turn in chronological order across both variants.
func TestPairedOptionCorrectionContinuityRegression(t *testing.T) {
	variants := []struct {
		name           string
		correctionUser string
		correctionAsst string
		marker         string
	}{
		{
			name:           "reject_second_option_logical_replication",
			correctionUser: "Correction: reject logical replication because the standby lacks superuser privileges; switch to blue-green cutover instead.",
			correctionAsst: "Understood: logical replication is ruled out; blue-green cutover is active instead.",
			marker:         "standby lacks superuser",
		},
		{
			name:           "reject_first_option_blue_green_cutover",
			correctionUser: "Correction: reject blue-green cutover because the maintenance window is blocked; switch to logical replication instead.",
			correctionAsst: "Understood: blue-green cutover is ruled out; logical replication is active instead.",
			marker:         "maintenance window is blocked",
		},
	}

	for _, variant := range variants {
		t.Run(variant.name, func(t *testing.T) {
			base := &llmprotocol.Request{
				Model: "planner-model",
				Messages: []llmprotocol.Message{
					// T0: older turn introducing two named options.
					textMsg(llmprotocol.RoleUser, "Propose two database migration options for the billing cluster."),
					textMsg(llmprotocol.RoleAssistant, "Option 1: blue-green cutover with snapshot restore. Option 2: logical replication with staged sync."),
					// T1: unrelated older chatter.
					textMsg(llmprotocol.RoleUser, "By the way, what is the weather forecast for Portland next weekend?"),
					textMsg(llmprotocol.RoleAssistant, "Portland will be cloudy with mild temperatures around sixty degrees."),
					// T2: superseding correction turn (does NOT repeat "second option").
					textMsg(llmprotocol.RoleUser, variant.correctionUser),
					textMsg(llmprotocol.RoleAssistant, variant.correctionAsst),
					// T3: unrelated newer chatter (more recent than T2!).
					textMsg(llmprotocol.RoleUser, "Also recommend a quiet coffee shop near downtown for Friday afternoon."),
					textMsg(llmprotocol.RoleAssistant, "Try the corner roastery on Third Avenue; it has plenty of seating."),
					// T4: brief recent turn guaranteed by MinRecentTurns=1.
					textMsg(llmprotocol.RoleUser, "Thanks, let's get back to engineering now."),
					textMsg(llmprotocol.RoleAssistant, "Ready whenever you are."),
					// T5: live turn referring to "the second option" without naming it.
					textMsg(llmprotocol.RoleUser, "Continue with the second option and outline the rollout checklist."),
				},
			}

			// Budget allows 3 prior turns out of 5: T4 (min recent) + 2 ranked turns.
			newPolicy := func(strategy Strategy, continuity ContinuityMode) Policy {
				p := DefaultPolicy()
				p.Strategy = strategy
				p.Budget.MaxPriorTurns = 3
				p.Budget.MinRecentTurns = 1
				p.Scoring.Continuity = continuity
				return p
			}

			// 1. Full-history baseline: retains everything, exceeds turn budget.
			_, fullDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, newPolicy(StrategyFull, ContinuityEnabled), RequestInput{})
			if err != nil || fullDiag.BudgetStatus != BudgetUnsatisfiable {
				t.Fatalf("full baseline err=%v diag=%+v", err, fullDiag)
			}

			// 2. Recency-only baseline: keeps T2, T3, T4 -> loses T0 option definitions.
			recReq, recDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, newPolicy(StrategyRecency, ContinuityEnabled), RequestInput{})
			if err != nil || recDiag.BudgetStatus != BudgetSatisfied {
				t.Fatalf("recency baseline err=%v diag=%+v", err, recDiag)
			}
			if containsSubstring(survivingTurnTexts(recReq), "Option 1: blue-green cutover") {
				t.Fatalf("recency-only should drop T0 under MaxPriorTurns=3, kept=%v", survivingTurnTexts(recReq))
			}

			// 3. Lexical-only ablation (continuity disabled): keeps T0 (matches "option")
			//    + T4 (min recent) + T3 (more recent chatter than T2) -> drops T2 correction!
			lexReq, lexDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, newPolicy(StrategyRelevance, ContinuityDisabled), RequestInput{})
			if err != nil || lexDiag.BudgetStatus != BudgetSatisfied {
				t.Fatalf("lexical ablation err=%v diag=%+v", err, lexDiag)
			}
			lexTexts := survivingTurnTexts(lexReq)
			if !containsSubstring(lexTexts, "Option 1: blue-green cutover") {
				t.Fatalf("lexical ablation should keep T0, got %v", lexTexts)
			}
			if containsSubstring(lexTexts, variant.marker) {
				t.Fatalf("lexical ablation unexpectedly kept non-overlapping correction T2: %v", lexTexts)
			}

			// 4. Relevance + Continuity selector: preserves BOTH T0 (options) AND
			//    T2 (the variant-specific correction) alongside T4 in chronological order!
			contReq, contDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, newPolicy(StrategyRelevance, ContinuityEnabled), RequestInput{})
			if err != nil || contDiag.BudgetStatus != BudgetSatisfied || contDiag.RemovedTurns != 2 {
				t.Fatalf("continuity selector err=%v diag=%+v", err, contDiag)
			}
			if contDiag.ContinuityBoostedTurns < 1 {
				t.Fatalf("expected continuity-boosted turns > 0, diag=%+v", contDiag)
			}
			contTexts := survivingTurnTexts(contReq)
			if !containsSubstring(contTexts, "Option 1: blue-green cutover") {
				t.Fatalf("continuity selector dropped option-definition turn T0: %v", contTexts)
			}
			if !containsSubstring(contTexts, variant.marker) {
				t.Fatalf("continuity selector dropped superseding correction turn T2 (%s): %v", variant.marker, contTexts)
			}
			if containsSubstring(contTexts, "Portland") || containsSubstring(contTexts, "coffee shop") {
				t.Fatalf("continuity selector retained unrelated chatter: %v", contTexts)
			}
			// Chronological reconstruction: T0 -> T2 -> T4 -> T5.
			if len(contReq.Messages) != 7 ||
				!strings.Contains(contReq.Messages[0].Content[0].Text, "billing cluster") ||
				!strings.Contains(contReq.Messages[2].Content[0].Text, variant.marker) ||
				!strings.Contains(contReq.Messages[4].Content[0].Text, "back to engineering") ||
				!strings.Contains(contReq.Messages[6].Content[0].Text, "second option") {
				t.Fatalf("unexpected chronological survivors: %v", contTexts)
			}
		})
	}
}

func TestKnowledgeUpdateContinuityAndNegativeControl(t *testing.T) {
	base := &llmprotocol.Request{
		Model: "db-model",
		Messages: []llmprotocol.Message{
			// T0: original version fact.
			textMsg(llmprotocol.RoleUser, "We run PostgreSQL 14 on our primary analytics cluster."),
			textMsg(llmprotocol.RoleAssistant, "Recorded PostgreSQL 14 for analytics."),
			// T1: later version correction linked to PostgreSQL.
			textMsg(llmprotocol.RoleUser, "Update: we actually upgraded PostgreSQL to 16 instead."),
			textMsg(llmprotocol.RoleAssistant, "Got it, analytics now runs PostgreSQL 16."),
			// T2: negative control — later turn with revision words but UNRELATED topic!
			textMsg(llmprotocol.RoleUser, "Actually, cancel the pizza delivery and switch to sushi instead."),
			textMsg(llmprotocol.RoleAssistant, "Switched lunch order to sushi."),
			// T3: live question.
			textMsg(llmprotocol.RoleUser, "Recommend PostgreSQL shared_buffers and checkpoint tuning."),
		},
	}

	policy := DefaultPolicy()
	policy.Budget.MaxPriorTurns = 2
	policy.Budget.MinRecentTurns = 0

	gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, policy, RequestInput{})
	if err != nil {
		t.Fatalf("ApplySteps failed: %v", err)
	}
	texts := survivingTurnTexts(gotReq)
	if !containsSubstring(texts, "PostgreSQL 14") || !containsSubstring(texts, "PostgreSQL to 16") {
		t.Fatalf("expected both original fact and PostgreSQL 16 update to survive: %v", texts)
	}
	if containsSubstring(texts, "sushi") || containsSubstring(texts, "pizza") {
		t.Fatalf("unrelated revision turn falsely got kept: %v", texts)
	}
	if diag.RemovedTurns != 1 {
		t.Fatalf("expected 1 removed turn, diag=%+v", diag)
	}
}

func TestCJKAndExactErrorCodeLexicalSelection(t *testing.T) {
	base := &llmprotocol.Request{
		Model: "cjk-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "数据库报错 TS-999，连接池配置上限只有 50。"),
			textMsg(llmprotocol.RoleAssistant, "收到，错误码 TS-999 通常与数据库连接池耗尽有关。"),
			textMsg(llmprotocol.RoleUser, "周末打算去公园散步，天气怎么样？"),
			textMsg(llmprotocol.RoleAssistant, "周末晴朗微风，适合户外散步。"),
			textMsg(llmprotocol.RoleUser, "帮我顺便推荐一本小说。"),
			textMsg(llmprotocol.RoleAssistant, "推荐阅读科幻短篇集。"),
			textMsg(llmprotocol.RoleUser, "请针对 TS-999 给出数据库连接池扩容方案。"),
		},
	}
	policy := DefaultPolicy()
	policy.Budget.MaxPriorTurns = 2
	policy.Budget.MinRecentTurns = 1

	gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, policy, RequestInput{})
	if err != nil || diag.BudgetStatus != BudgetSatisfied {
		t.Fatalf("CJK selection err=%v diag=%+v", err, diag)
	}
	texts := survivingTurnTexts(gotReq)
	if !containsSubstring(texts, "TS-999") {
		t.Fatalf("failed to retain CJK/error-code turn: %v", texts)
	}
	if containsSubstring(texts, "公园散步") {
		t.Fatalf("failed to evict unrelated CJK chatter: %v", texts)
	}
}

func TestModelAwareBudgetAndSemanticCostEstimator(t *testing.T) {
	base := &llmprotocol.Request{
		Model: "model-small",
		Instructions: []llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: strings.Repeat("strict system policy instructions ", 12)}},
		}},
		Tools: []llmprotocol.Tool{{
			Name:        "run_sql",
			Description: "Execute a read-only SQL query against analytics.",
			InputSchema: []byte(`{"type":"object","properties":{"query":{"type":"string"},"timeout_ms":{"type":"integer"}},"required":["query"]}`),
		}},
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "SQL schema fact: table invoices(id, region, amount_usd). "+strings.Repeat("detail ", 30)),
			textMsg(llmprotocol.RoleAssistant, "Recorded invoices table schema. "+strings.Repeat("note ", 30)),
			textMsg(llmprotocol.RoleUser, "Weather chat about clouds and rain. "+strings.Repeat("chat ", 30)),
			textMsg(llmprotocol.RoleAssistant, "Sunny tomorrow. "+strings.Repeat("chat ", 30)),
			textMsg(llmprotocol.RoleUser, "Brief recent check-in. "+strings.Repeat("check ", 20)),
			textMsg(llmprotocol.RoleAssistant, "Ready to proceed. "+strings.Repeat("ready ", 20)),
			textMsg(llmprotocol.RoleUser, "Query invoices grouped by region."),
		},
	}

	estimator := NewSemanticCostEstimator(base, nil)
	fixed, src := estimator.FixedOverheadTokens(base.Model)
	if fixed <= 0 || src != "neutral_estimate" {
		t.Fatalf("expected positive fixed overhead for instructions+tools, got fixed=%d src=%q", fixed, src)
	}

	// Calculate exact turn costs to construct tight vs generous model windows.
	irProbe := contextcompression.ParseSemanticRequest(cloneRequest(t, base), contextcompression.Provenance{})
	turns := GroupTurns(irProbe.TransformationView(), DefaultMaxTurnTextBytes)
	turnCost := func(turn Turn) int {
		sum := 0
		for _, m := range turn.Messages {
			c, _ := estimator.MessageTokens(base.Model, m)
			sum += c
		}
		return sum
	}
	t0Cost := turnCost(turns[0])
	t1Cost := turnCost(turns[1])
	t2Cost := turnCost(turns[2])
	liveCost := turnCost(turns[3])

	// Generous model window fits all 3 prior turns + fixed + output reserve.
	reserve := 256
	generousWindow := fixed + liveCost + t0Cost + t1Cost + t2Cost + reserve + 10
	policy := DefaultPolicy()
	policy.Budget.MinRecentTurns = 1

	generousReq, generousDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, policy, RequestInput{
		ModelBudget: ModelBudget{
			Model: "model-large",
			Capabilities: contextcompression.ModelContextCapabilities{
				ContextWindow:   generousWindow,
				RequestedOutput: reserve,
			},
		},
		CostEstimator: estimator,
	})
	if err != nil || generousDiag.Outcome != OutcomeSkipped || generousDiag.Reason != ReasonWithinBudget {
		t.Fatalf("generous model should keep all turns: err=%v diag=%+v", err, generousDiag)
	}
	if len(generousReq.Messages) != len(base.Messages) {
		t.Fatalf("generous model removed messages: %d != %d", len(generousReq.Messages), len(base.Messages))
	}

	// Smaller model window (or higher output demand) fits T2 (min recent) + T0 (SQL)
	// but NOT T1 (weather), proving instructions + tools + output reserve are accounted for.
	tightWindow := fixed + liveCost + t0Cost + t2Cost + reserve
	tightReq, tightDiag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, policy, RequestInput{
		ModelBudget: ModelBudget{
			Model: "model-small",
			Capabilities: contextcompression.ModelContextCapabilities{
				ContextWindow:   tightWindow,
				RequestedOutput: reserve,
			},
		},
		CostEstimator: estimator,
	})
	if err != nil || tightDiag.Outcome != OutcomeApplied || tightDiag.BudgetStatus != BudgetSatisfied {
		t.Fatalf("tight model failed: err=%v diag=%+v", err, tightDiag)
	}
	if tightDiag.ReservedOutputTokens != reserve || tightDiag.FixedOverheadTokens != fixed+liveCost {
		t.Fatalf("budget accounting mismatch: diag=%+v want reserve=%d fixed=%d", tightDiag, reserve, fixed+liveCost)
	}
	if tightDiag.RetainedHistoryTokens > tightDiag.AvailableHistoryBudget {
		t.Fatalf("retained history %d exceeded available budget %d", tightDiag.RetainedHistoryTokens, tightDiag.AvailableHistoryBudget)
	}
	texts := survivingTurnTexts(tightReq)
	if !containsSubstring(texts, "table invoices") || !containsSubstring(texts, "recent check-in") || containsSubstring(texts, "Weather chat") {
		t.Fatalf("unexpected tight-budget survivors: %v", texts)
	}
}

func TestBudgetConflictsAndMinRecencyPolicies(t *testing.T) {
	base := &llmprotocol.Request{
		Model: "tight-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "small older fact"),
			textMsg(llmprotocol.RoleAssistant, "ok"),
			textMsg(llmprotocol.RoleUser, "large recent turn "+strings.Repeat("payload ", 80)),
			textMsg(llmprotocol.RoleAssistant, "large recent reply "+strings.Repeat("payload ", 80)),
			textMsg(llmprotocol.RoleUser, "live question"),
		},
	}

	t.Run("invalid_min_recent_exceeds_max_prior", func(t *testing.T) {
		p := DefaultPolicy()
		p.Budget.MaxPriorTurns = 1
		p.Budget.MinRecentTurns = 2
		if err := p.Validate(); !errors.Is(err, ErrInvalidPolicy) {
			t.Fatalf("Validate() = %v, want ErrInvalidPolicy", err)
		}
	})

	t.Run("min_recency_conflict_strict_fails_closed", func(t *testing.T) {
		p := DefaultPolicy()
		p.FailureMode = contextcompression.FailureClosed
		p.Budget.MinRecentTurns = 1
		p.Budget.MaxHistoryTokens = 40 // fits small T0, does NOT fit large recent T1
		p.Budget.ConflictPolicy = ConflictStrict

		gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{})
		if err == nil || diag.Outcome != OutcomeFailed || diag.Reason != ReasonMinRecencyConflict {
			t.Fatalf("expected min_recency_conflict failure, err=%v diag=%+v", err, diag)
		}
		if len(gotReq.Messages) != len(base.Messages) {
			t.Fatal("failed strict plan mutated request")
		}
	})

	t.Run("min_recency_conflict_relaxed_stays_within_budget", func(t *testing.T) {
		p := DefaultPolicy()
		p.FailureMode = contextcompression.FailureClosed
		p.Budget.MinRecentTurns = 1
		p.Budget.MaxHistoryTokens = 40 // fits small T0, does NOT fit large recent T1
		p.Budget.ConflictPolicy = ConflictRelaxMinRecency

		gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{})
		if err != nil {
			t.Fatalf("relaxed min recency failed: %v", err)
		}
		if diag.BudgetStatus != BudgetMinRecencyRelaxed || diag.MinRecentRetained != 0 {
			t.Fatalf("expected relaxed min recency status, diag=%+v", diag)
		}
		if diag.RetainedHistoryTokens > 40 {
			t.Fatalf("retained history %d exceeded cap 40", diag.RetainedHistoryTokens)
		}
		if containsSubstring(survivingTurnTexts(gotReq), "large recent turn") {
			t.Fatal("oversized recent turn should not exceed token budget")
		}
	})

	t.Run("protected_overflow_strict_fails_without_dropping_protected", func(t *testing.T) {
		p := DefaultPolicy()
		p.FailureMode = contextcompression.FailureClosed
		p.Budget.MaxInputTokens = 5 // smaller than live turn alone
		p.Budget.ConflictPolicy = ConflictStrict

		gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{})
		if err == nil || diag.Outcome != OutcomeFailed || diag.Reason != ReasonBudgetUnsatisfiable {
			t.Fatalf("expected budget_unsatisfiable failure, err=%v diag=%+v", err, diag)
		}
		if len(gotReq.Messages) != len(base.Messages) {
			t.Fatal("protected overflow strict mode mutated request")
		}
	})

	t.Run("protected_overflow_drop_eligible_preserves_protected_and_flags_overflow", func(t *testing.T) {
		p := DefaultPolicy()
		p.FailureMode = contextcompression.FailureClosed
		p.Budget.MaxInputTokens = 5 // smaller than live turn alone
		p.Budget.ConflictPolicy = ConflictDropEligible

		gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{})
		if err != nil {
			t.Fatalf("drop_eligible failed: %v", err)
		}
		if diag.BudgetStatus != BudgetProtectedOverflow || diag.Reason != ReasonProtectedOverflow {
			t.Fatalf("expected explicit protected_overflow status, diag=%+v", diag)
		}
		if len(gotReq.Messages) != 1 || gotReq.Messages[0].Content[0].Text != "live question" {
			t.Fatalf("expected only protected live turn to survive, got %v", survivingTurnTexts(gotReq))
		}
	})
}

func TestExternalScorerBoundsUnknownScoresAndFallback(t *testing.T) {
	base := &llmprotocol.Request{
		Model: "scorer-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "alpha topic one"),
			textMsg(llmprotocol.RoleAssistant, "alpha reply"),
			textMsg(llmprotocol.RoleUser, "beta topic two"),
			textMsg(llmprotocol.RoleAssistant, "beta reply"),
			textMsg(llmprotocol.RoleUser, "gamma topic three"),
			textMsg(llmprotocol.RoleAssistant, "gamma reply"),
			textMsg(llmprotocol.RoleUser, "recent turn"),
			textMsg(llmprotocol.RoleAssistant, "recent reply"),
			textMsg(llmprotocol.RoleUser, "ask about alpha topic"),
		},
	}

	t.Run("valid_external_scorer_guides_ranking", func(t *testing.T) {
		p := DefaultPolicy()
		p.Budget.MaxPriorTurns = 2
		p.Budget.MinRecentTurns = 1
		// Scorer prefers "beta" even though lexical query mentions "alpha".
		scorer := &stubScorer{fn: func(_ context.Context, _ string, texts []string) ([]float64, error) {
			scores := make([]float64, len(texts))
			for i, text := range texts {
				if strings.Contains(text, "beta") {
					scores[i] = 10.0
				} else {
					scores[i] = 0.1
				}
			}
			return scores, nil
		}}

		gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{Scorer: scorer})
		if err != nil || diag.ScoringSource != ScoringExternal || diag.FallbackReason != FallbackNone {
			t.Fatalf("external scorer err=%v diag=%+v", err, diag)
		}
		if !containsSubstring(survivingTurnTexts(gotReq), "beta topic") {
			t.Fatalf("external scorer preference not honored: %v", survivingTurnTexts(gotReq))
		}
	})

	for _, tc := range []struct {
		name         string
		fn           func(context.Context, string, []string) ([]float64, error)
		wantFallback string
	}{
		{
			name: "scorer_error_falls_back_to_lexical",
			fn: func(context.Context, string, []string) ([]float64, error) {
				return nil, errors.New("secret upstream bearer token sk-12345 failed")
			},
			wantFallback: FallbackScorerError,
		},
		{
			name: "nan_score_falls_back_deterministically",
			fn: func(_ context.Context, _ string, texts []string) ([]float64, error) {
				scores := make([]float64, len(texts))
				scores[0] = math.NaN()
				return scores, nil
			},
			wantFallback: FallbackInvalidScore,
		},
		{
			name: "inf_score_falls_back_deterministically",
			fn: func(_ context.Context, _ string, texts []string) ([]float64, error) {
				scores := make([]float64, len(texts))
				scores[0] = math.Inf(1)
				return scores, nil
			},
			wantFallback: FallbackInvalidScore,
		},
		{
			name: "length_mismatch_falls_back",
			fn: func(context.Context, string, []string) ([]float64, error) {
				return []float64{1.0}, nil
			},
			wantFallback: FallbackLengthMismatch,
		},
		{
			name: "scoring_timeout_falls_back",
			fn: func(ctx context.Context, _ string, _ []string) ([]float64, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			},
			wantFallback: FallbackScorerTimeout,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := DefaultPolicy()
			p.Budget.MaxPriorTurns = 2
			p.Budget.MinRecentTurns = 1
			p.Scoring.ScoringTimeout = 5 * time.Millisecond
			p.Scoring.Timeout = 200 * time.Millisecond
			p.Scoring.ScorerFallback = ScorerFallbackLexical

			gotReq, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{Scorer: &stubScorer{fn: tc.fn}})
			if err != nil {
				t.Fatalf("expected lexical fallback, got err=%v", err)
			}
			if diag.ScoringSource != ScoringLexicalFallback || diag.FallbackReason != tc.wantFallback {
				t.Fatalf("diag scoring=%q fallback=%q, want lexical_fallback/%q", diag.ScoringSource, diag.FallbackReason, tc.wantFallback)
			}
			if !containsSubstring(survivingTurnTexts(gotReq), "alpha topic") {
				t.Fatalf("lexical fallback should still retain alpha turn: %v", survivingTurnTexts(gotReq))
			}
			// Raw error strings must never leak into Diagnostics.
			if strings.Contains(mustMarshalJSON(t, diag), "sk-12345") {
				t.Fatal("diagnostics leaked raw scorer error")
			}
		})
	}

	t.Run("scorer_fail_closed_mode_rejects_invalid_score", func(t *testing.T) {
		p := DefaultPolicy()
		p.FailureMode = contextcompression.FailureClosed
		p.Budget.MaxPriorTurns = 2
		p.Budget.MinRecentTurns = 1
		p.Scoring.ScorerFallback = ScorerFallbackFail

		_, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{
			Scorer: &stubScorer{fn: func(_ context.Context, _ string, texts []string) ([]float64, error) {
				return make([]float64, len(texts)+1), nil
			}},
		})
		if err == nil || diag.Outcome != OutcomeFailed || diag.Reason != ReasonScoringFailed {
			t.Fatalf("expected scoring_failed error, err=%v diag=%+v", err, diag)
		}
	})

	t.Run("scorer_candidate_and_scan_bounds_enforced", func(t *testing.T) {
		p := DefaultPolicy()
		p.Budget.MaxPriorTurns = 2
		p.Budget.MinRecentTurns = 1
		p.Scoring.MaxCandidateTurns = 3
		p.Scoring.MaxScoredTurns = 2

		scorer := &stubScorer{fn: func(_ context.Context, _ string, texts []string) ([]float64, error) {
			scores := make([]float64, len(texts))
			for i := range scores {
				scores[i] = float64(i + 1)
			}
			return scores, nil
		}}
		_, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{Scorer: scorer})
		if err != nil {
			t.Fatalf("bounded scoring failed: %v", err)
		}
		if len(scorer.texts) != 2 {
			t.Fatalf("external scorer saw %d texts, want capped at 2", len(scorer.texts))
		}
		if !diag.CandidatesTruncated || diag.ScoredTurns != 3 || diag.FallbackReason != FallbackCandidateCap {
			t.Fatalf("bounds metadata mismatch: diag=%+v", diag)
		}
	})

	t.Run("hard_history_turn_scan_limit", func(t *testing.T) {
		p := DefaultPolicy()
		p.FailureMode = contextcompression.FailureClosed
		p.Budget.MaxPriorTurns = 2
		p.Scoring.MaxHistoryTurns = 2

		_, diag, err := runSelectionOnCopy(t, base, contextcompression.Provenance{}, p, RequestInput{})
		if err == nil || diag.Outcome != OutcomeFailed || diag.Reason != ReasonScanLimitExceeded {
			t.Fatalf("expected scan_limit_exceeded, err=%v diag=%+v", err, diag)
		}
	})
}

func TestDeterminismTieBreakingReplayAndContentMinimization(t *testing.T) {
	secretText := "CONFIDENTIAL_CUSTOMER_SECRET_PAYLOAD_99182"
	base := &llmprotocol.Request{
		Model: "det-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "neutral alpha "+secretText),
			textMsg(llmprotocol.RoleAssistant, "neutral reply one"),
			textMsg(llmprotocol.RoleUser, "neutral beta "+secretText),
			textMsg(llmprotocol.RoleAssistant, "neutral reply two"),
			textMsg(llmprotocol.RoleUser, "neutral gamma "+secretText),
			textMsg(llmprotocol.RoleAssistant, "neutral reply three"),
			textMsg(llmprotocol.RoleUser, "unrelated live query xyz"),
		},
	}

	policy := DefaultPolicy()
	policy.Budget.MaxPriorTurns = 2
	policy.Budget.MinRecentTurns = 0

	// Equal lexical scores (all 0) must break deterministically in favor of recency (T2, T1 kept; T0 dropped).
	reqA, diagA, errA := runSelectionOnCopy(t, base, contextcompression.Provenance{}, policy, RequestInput{})
	reqB, diagB, errB := runSelectionOnCopy(t, base, contextcompression.Provenance{}, policy, RequestInput{})
	if errA != nil || errB != nil {
		t.Fatalf("unexpected errors: %v / %v", errA, errB)
	}
	if mustMarshalJSON(t, reqA) != mustMarshalJSON(t, reqB) {
		t.Fatal("non-deterministic turn selection output")
	}
	if !reflect.DeepEqual(diagA, diagB) {
		t.Fatalf("non-deterministic diagnostics: %+v vs %+v", diagA, diagB)
	}
	if containsSubstring(survivingTurnTexts(reqA), "neutral alpha") {
		t.Fatalf("tie-breaking failed to drop oldest turn: %v", survivingTurnTexts(reqA))
	}

	// Content minimization: Diagnostics must never contain conversation text or secrets.
	encodedDiag := mustMarshalJSON(t, diagA)
	for _, forbidden := range []string{secretText, "neutral alpha", "neutral beta", "unrelated live"} {
		if strings.Contains(encodedDiag, forbidden) {
			t.Fatalf("diagnostics leaked conversation content %q: %s", forbidden, encodedDiag)
		}
	}

	// Idempotent replay on the same RequestIR must be a no-op.
	replayReq := cloneRequest(t, base)
	ir := contextcompression.ParseSemanticRequest(replayReq, contextcompression.Provenance{})
	action := NewAction(policy, RequestInput{CostEstimator: NewSemanticCostEstimator(replayReq, nil)})
	steps := []contextcompression.TransformationStep{action.Step()}
	if err := ir.ApplySteps(context.Background(), steps); err != nil {
		t.Fatalf("first ApplySteps: %v", err)
	}
	afterFirst := mustMarshalJSON(t, replayReq)
	if err := ir.ApplySteps(context.Background(), steps); err != nil {
		t.Fatalf("second ApplySteps: %v", err)
	}
	if mustMarshalJSON(t, replayReq) != afterFirst {
		t.Fatal("replayed ApplySteps modified request")
	}
	if got := action.Reconcile(ir.Transformations.Receipts()); got.Outcome != OutcomeApplied || got.RemovedTurns != 1 {
		t.Fatalf("unexpected reconciled receipt after replay: %+v", got)
	}

	// FailureOpen continues without error; Reconcile reports failed with zero removals.
	openPolicy := DefaultPolicy()
	openPolicy.FailureMode = contextcompression.FailureOpen
	openPolicy.Budget.MaxInputTokens = 2
	openPolicy.Budget.ConflictPolicy = ConflictStrict
	openReq, openDiag, openErr := runSelectionOnCopy(t, base, contextcompression.Provenance{}, openPolicy, RequestInput{})
	if openErr != nil {
		t.Fatalf("FailureOpen should not abort ApplySteps: %v", openErr)
	}
	if openDiag.Outcome != OutcomeFailed || openDiag.RemovedMessages != 0 || len(openReq.Messages) != len(base.Messages) {
		t.Fatalf("FailureOpen reconcile mismatch: diag=%+v msgs=%d", openDiag, len(openReq.Messages))
	}

	// Disabled step reconciled cleanly as skipped/disabled.
	disabledReq := cloneRequest(t, base)
	disabledIR := contextcompression.ParseSemanticRequest(disabledReq, contextcompression.Provenance{})
	disabledAction := NewAction(policy, RequestInput{})
	disabledStep := disabledAction.Step()
	disabledStep.Enabled = false
	if err := disabledIR.ApplySteps(context.Background(), []contextcompression.TransformationStep{disabledStep}); err != nil {
		t.Fatalf("disabled step failed: %v", err)
	}
	disabledDiag := disabledAction.Reconcile(disabledIR.Transformations.Receipts())
	if disabledDiag.Outcome != OutcomeSkipped || disabledDiag.Reason != "disabled" || disabledDiag.RemovedMessages != 0 {
		t.Fatalf("disabled reconcile = %+v", disabledDiag)
	}

	// Upstream FailureClosed abort prevents selection from running; Reconcile reports not_evaluated.
	abortedReq := cloneRequest(t, base)
	abortedIR := contextcompression.ParseSemanticRequest(abortedReq, contextcompression.Provenance{})
	abortedAction := NewAction(policy, RequestInput{})
	failingReset := contextcompression.TransformationStep{
		Kind:        contextcompression.TransformReset,
		Enabled:     true,
		FailureMode: contextcompression.FailureClosed,
		Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
			return contextcompression.TransformationEdits{}, errors.New("upstream abort")
		},
	}
	if err := abortedIR.ApplySteps(context.Background(), []contextcompression.TransformationStep{failingReset, abortedAction.Step()}); err == nil {
		t.Fatal("expected upstream fail-closed error")
	}
	abortedDiag := abortedAction.Reconcile(abortedIR.Transformations.Receipts())
	if abortedDiag.Outcome != OutcomeSkipped || abortedDiag.Reason != ReasonNotEvaluated || abortedDiag.RemovedMessages != 0 {
		t.Fatalf("aborted reconcile = %+v", abortedDiag)
	}

	// Cancelled parent context fails without mutating request.
	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel()
	cancelReq := cloneRequest(t, base)
	cancelIR := contextcompression.ParseSemanticRequest(cancelReq, contextcompression.Provenance{})
	cancelPolicy := policy
	cancelPolicy.FailureMode = contextcompression.FailureClosed
	cancelAction := NewAction(cancelPolicy, RequestInput{})
	if err := cancelIR.ApplySteps(cancelCtx, []contextcompression.TransformationStep{cancelAction.Step()}); err == nil {
		t.Fatal("expected cancellation error")
	}
	if len(cancelReq.Messages) != len(base.Messages) {
		t.Fatal("cancelled selection mutated request")
	}
}

func TestCostEstimatorsOpeningTurnAndPolicyValidation(t *testing.T) {
	t.Run("view_cost_estimator_and_calibrated_semantic_estimator_preserve_image_reserves", func(t *testing.T) {
		req := &llmprotocol.Request{
			Model: "vision-model",
			Instructions: []llmprotocol.InstructionBlock{{
				Role:    llmprotocol.RoleSystem,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "system rules"}},
			}},
			Tools: []llmprotocol.Tool{{
				Name:        "search",
				Description: "search docs",
				InputSchema: []byte(`{"type":"object"}`),
			}},
			Messages: []llmprotocol.Message{
				{
					Role: llmprotocol.RoleUser,
					Content: []llmprotocol.Content{
						{Kind: llmprotocol.ContentText, Text: "check this image"},
						{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/pic.png"},
					},
				},
				textMsg(llmprotocol.RoleAssistant, "looks clear"),
				textMsg(llmprotocol.RoleUser, "live question"),
			},
		}
		calibrated := contextcompression.CalibratedTokenCounter{
			Estimate: func(_ string, byteLen int) (int, bool) {
				if byteLen == 0 {
					return 0, true
				}
				return max(1, byteLen/2), true
			},
		}
		semEst := NewSemanticCostEstimator(req, calibrated)
		fixed, fixedSrc := semEst.FixedOverheadTokens(req.Model)
		if fixed <= 0 || fixedSrc != "provider_calibrated" {
			t.Fatalf("calibrated semantic fixed = %d, source = %q", fixed, fixedSrc)
		}
		ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
		view := ir.TransformationView()
		imgTokens, _ := semEst.MessageTokens(req.Model, view.Messages[0])
		if imgTokens < llmprotocol.InputImageTokens {
			t.Fatalf("calibrated estimator dropped image reserve: got %d, want >= %d", imgTokens, llmprotocol.InputImageTokens)
		}

		// Default ViewCostEstimator path (no CostEstimator injected).
		viewEst := ViewCostEstimator{}
		if vFixed, vSrc := viewEst.FixedOverheadTokens(req.Model); vFixed != 0 || vSrc != "heuristic" {
			t.Fatalf("ViewCostEstimator fixed = %d, source = %q", vFixed, vSrc)
		}
		if vMsg, _ := viewEst.MessageTokens(req.Model, view.Messages[2]); vMsg <= 0 {
			t.Fatalf("ViewCostEstimator message tokens = %d, want > 0", vMsg)
		}
	})

	t.Run("opening_turn_no_eligible_history_and_protected_overflow", func(t *testing.T) {
		opening := &llmprotocol.Request{
			Model:    "model",
			Messages: []llmprotocol.Message{textMsg(llmprotocol.RoleUser, "brand new opening turn")},
		}
		// Zero-value Policy{} normalizes cleanly; within generous budget -> no_eligible_history.
		p := Policy{Budget: BudgetPolicy{MaxInputTokens: 200}}
		_, diag, err := runSelectionOnCopy(t, opening, contextcompression.Provenance{}, p, RequestInput{})
		if err != nil || diag.Outcome != OutcomeSkipped || diag.Reason != ReasonNoEligibleHistory || diag.BudgetStatus != BudgetSatisfied {
			t.Fatalf("opening turn within budget: err=%v diag=%+v", err, diag)
		}

		// Protected overflow with no eligible history: strict fails, drop_eligible skips with protected_overflow.
		strict := DefaultPolicy()
		strict.FailureMode = contextcompression.FailureClosed
		strict.Budget.MaxInputTokens = 1
		strict.Budget.ConflictPolicy = ConflictStrict
		if _, sDiag, sErr := runSelectionOnCopy(t, opening, contextcompression.Provenance{}, strict, RequestInput{}); sErr == nil || sDiag.Reason != ReasonBudgetUnsatisfiable {
			t.Fatalf("opening strict overflow: err=%v diag=%+v", sErr, sDiag)
		}

		drop := strict
		drop.Budget.ConflictPolicy = ConflictDropEligible
		if _, dDiag, dErr := runSelectionOnCopy(t, opening, contextcompression.Provenance{}, drop, RequestInput{}); dErr != nil || dDiag.Reason != ReasonProtectedOverflow || dDiag.BudgetStatus != BudgetProtectedOverflow {
			t.Fatalf("opening drop_eligible overflow: err=%v diag=%+v", dErr, dDiag)
		}
	})

	t.Run("policy_validation_rejects_invalid_configs", func(t *testing.T) {
		invalid := []Policy{
			{Strategy: Strategy("unknown")},
			{Budget: BudgetPolicy{ConflictPolicy: BudgetConflictPolicy("unknown")}},
			{FailureMode: contextcompression.FailureMode("unknown")},
			{Scoring: ScoringPolicy{Continuity: ContinuityMode("unknown")}},
			{Scoring: ScoringPolicy{ScorerFallback: ScorerFallbackMode("unknown")}},
			{Budget: BudgetPolicy{MaxInputTokens: -1}},
			{Budget: BudgetPolicy{MaxHistoryTokens: -1}},
			{Scoring: ScoringPolicy{RelevanceWeight: math.NaN()}},
			{Scoring: ScoringPolicy{RecencyWeight: -0.5}},
			{Scoring: ScoringPolicy{MaxCandidateTurns: -1}},
		}
		for i, p := range invalid {
			if err := p.Validate(); !errors.Is(err, ErrInvalidPolicy) {
				t.Fatalf("case %d: Validate() = %v, want ErrInvalidPolicy", i, err)
			}
		}
	})
}

func FuzzTurnSelectionInvariants(f *testing.F) {
	f.Add(uint8(0), uint8(2), uint8(1), uint8(80), uint8(0))
	f.Add(uint8(1), uint8(1), uint8(1), uint8(10), uint8(2))
	f.Add(uint8(2), uint8(0), uint8(0), uint8(255), uint8(1))

	strategies := []Strategy{StrategyRelevance, StrategyRecency, StrategyFull}
	conflicts := []BudgetConflictPolicy{ConflictStrict, ConflictRelaxMinRecency, ConflictDropEligible}

	f.Fuzz(func(t *testing.T, strategySel, maxPrior, minRecent, histBudget, conflictSel uint8) {
		req := &llmprotocol.Request{
			Model: "fuzz-model",
			Messages: []llmprotocol.Message{
				textMsg(llmprotocol.RoleUser, "database schema orders table"),
				textMsg(llmprotocol.RoleAssistant, "schema recorded"),
				textMsg(llmprotocol.RoleUser, "run tool lookup"),
				toolCallMsg(llmprotocol.ToolCall{ID: "c1", Name: "lookup", Arguments: `{"k":"v"}`}),
				toolResultMsg(llmprotocol.RoleTool, "c1", "lookup output"),
				textMsg(llmprotocol.RoleAssistant, "lookup summary"),
				textMsg(llmprotocol.RoleUser, "authorized secret rule"),
				textMsg(llmprotocol.RoleAssistant, "auth noted"),
				textMsg(llmprotocol.RoleUser, "live query about orders table"),
			},
		}
		prov := contextcompression.Provenance{
			ProtectedMessages: map[int]contextcompression.Protection{6: contextcompression.ProtectAuthorization},
		}

		p := DefaultPolicy()
		p.Strategy = strategies[int(strategySel)%len(strategies)]
		p.Budget.MaxPriorTurns = int(maxPrior % 5)
		p.Budget.MinRecentTurns = int(minRecent % 4)
		p.Budget.MaxHistoryTokens = int(histBudget)
		p.Budget.ConflictPolicy = conflicts[int(conflictSel)%len(conflicts)]
		p.FailureMode = contextcompression.FailureClosed

		ir := contextcompression.ParseSemanticRequest(req, prov)
		action := NewAction(p, RequestInput{CostEstimator: NewSemanticCostEstimator(req, nil)})
		err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()})
		diag := action.Reconcile(ir.Transformations.Receipts())

		// Protected turns (authorization at original IDs 6,7 and live at 8) must ALWAYS survive.
		texts := survivingTurnTexts(req)
		if !containsSubstring(texts, "authorized secret rule") || !containsSubstring(texts, "live query") {
			t.Fatalf("protected content lost: %v", texts)
		}
		if err != nil {
			if diag.RemovedMessages != 0 {
				t.Fatalf("failed plan reported removals: %+v", diag)
			}
			return
		}
		// Shared executor already validates turn and tool-exchange atomicity; also
		// assert token and turn-count bounds whenever BudgetSatisfied is claimed.
		if diag.BudgetStatus == BudgetSatisfied {
			if p.Budget.MaxHistoryTokens > 0 && diag.RetainedHistoryTokens > diag.AvailableHistoryBudget {
				t.Fatalf("retained history %d exceeded available %d", diag.RetainedHistoryTokens, diag.AvailableHistoryBudget)
			}
			retainedEligible := diag.EligibleTurns - diag.RemovedTurns
			if p.Budget.MaxPriorTurns > 0 && retainedEligible > p.Budget.MaxPriorTurns {
				t.Fatalf("retained eligible turns %d exceeded cap %d", retainedEligible, p.Budget.MaxPriorTurns)
			}
		}
	})
}
