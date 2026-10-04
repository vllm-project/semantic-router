package topiccontinuity

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func manyBlocks(n int) llmprotocol.Message {
	content := make([]llmprotocol.Content, n)
	for i := range content {
		content[i] = text("x")
	}
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: content}
}

func TestEvidencePhaseBlockCap(t *testing.T) {
	messages := conversation(manyBlocks(maxContentBlocksPerPrepare), assistant("ok"),
		user("What is the capital city of Peru and its current population?"))
	result := evaluate(messages, defaultPolicy)
	assertInvariants(t, result)
	if result.Coverage != CoveragePartial || !result.Scope.FeatureCapReached || result.Class == ClassChange {
		t.Fatalf("got %s/%s coverage=%s scope=%+v", result.Class, result.Reason, result.Coverage, result.Scope)
	}
}

func TestToolNameCountCap(t *testing.T) {
	for _, count := range []int{maxToolNamesPerTurn - 1, maxToolNamesPerTurn, maxToolNamesPerTurn + 1} {
		calls := make([]llmprotocol.Content, count)
		for i := range calls {
			calls[i] = toolCall(fmt.Sprintf("c%d", i), fmt.Sprintf("tool_%02d", i), "")
		}
		messages := []llmprotocol.Message{user("Run the batch")}
		messages = append(messages, assistantCalls(calls...))
		for i := range calls {
			messages = append(messages, toolResult(fmt.Sprintf("c%d", i), "ok"))
		}
		messages = append(messages, assistant("All done."),
			user("What is the capital city of Peru and its current population?"))
		result := evaluate(messages, defaultPolicy)
		assertInvariants(t, result)
		reached := count >= maxToolNamesPerTurn
		if result.Scope.FeatureCapReached != reached || (reached && result.Coverage != CoveragePartial) {
			t.Fatalf("count=%d: cap=%v coverage=%s", count, result.Scope.FeatureCapReached, result.Coverage)
		}
		if reached && result.Class == ClassChange {
			t.Fatalf("count=%d: change despite cap", count)
		}
	}
}

func TestEntityCapForcesPartialCoverage(t *testing.T) {
	var identifiers strings.Builder
	for i := 0; i <= maxEntitiesPerSegment+50; i++ {
		fmt.Fprintf(&identifiers, "var_%04d ", i)
	}
	result := evaluate(conversation(exchange(identifiers.String(), "noted"),
		user("Please rename var_1070 to something clearer for readers")), defaultPolicy)
	assertInvariants(t, result)
	if result.Coverage != CoveragePartial || !result.Scope.FeatureCapReached || result.Class == ClassChange {
		t.Fatalf("got %s/%s coverage=%s scope=%+v", result.Class, result.Reason, result.Coverage, result.Scope)
	}
}

func TestBoundedOpaqueCheck(t *testing.T) {
	live := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
		text(strings.Repeat(" ", 4<<20)), image(),
	}}
	prepared := prepare(context.Background(), conversation(unrelatedHistory(1), live), true, defaultPolicy)
	if prepared.InputBytes > defaultPolicy.Limits.MaxInputBytes || turnBytes(prepared.Live) > defaultPolicy.Limits.MaxTurnBytes {
		t.Fatalf("read beyond budget: input=%d live=%d", prepared.InputBytes, turnBytes(prepared.Live))
	}
	result := classify(defaultConfig(defaultPolicy), prepared, extract(context.Background(), prepared))
	if result.Reason != ReasonInsufficientText || result.Coverage != CoveragePartial {
		t.Fatalf("got %s coverage=%s", result.Reason, result.Coverage)
	}
}

func TestBudgetsHold(t *testing.T) {
	policy := HistoryPolicy{Limits: Limits{MaxPriorTurns: 3, MaxTurnBytes: 300, MaxInputBytes: 1100}, IncludeAssistant: true}
	long := strings.Repeat("ąbc ", 500)
	messages := conversation(exchange(long, long), exchange(long, long), exchange(long, long),
		exchange(long, long), user(long))
	prepared := prepare(context.Background(), messages, true, policy)
	if len(prepared.Prior) > policy.Limits.MaxPriorTurns || prepared.InputBytes > policy.Limits.MaxInputBytes {
		t.Fatalf("prior=%d input=%d", len(prepared.Prior), prepared.InputBytes)
	}
	for _, turn := range append([]evidenceTurn{prepared.Live}, prepared.Prior...) {
		if turnBytes(turn) > policy.Limits.MaxTurnBytes {
			t.Fatalf("turn bytes %d", turnBytes(turn))
		}
		for _, segment := range append(append([]textSegment{}, turn.User...), turn.Assistant...) {
			if !utf8.ValidString(string(segment)) {
				t.Fatalf("invalid UTF-8 segment %q", segment)
			}
		}
	}
	if prepared.Coverage != CoveragePartial {
		t.Fatalf("coverage = %s", prepared.Coverage)
	}
}

func TestInvalidLimitsAreInternalError(t *testing.T) {
	policy := HistoryPolicy{Limits: Limits{MaxPriorTurns: 0, MaxTurnBytes: 256, MaxInputBytes: 1024}}
	load := func() ([]llmprotocol.Message, bool) { return conversation(unrelatedHistory(1), user("next")), true }
	result := EvaluateAll(context.Background(), load, []EvalConfig{defaultConfig(policy)}).Rules[0].Result
	if result.Reason != ReasonInternalError {
		t.Fatalf("got %s", result.Reason)
	}
}

func FuzzPrepareStaysBounded(f *testing.F) {
	f.Add("hello world", "İstanbul `x` \"y\" 'new topic'", uint16(300))
	f.Add(strings.Repeat("ą", 400), "unrelated question: anything?", uint16(256))
	f.Fuzz(func(t *testing.T, prior, live string, turnBytes16 uint16) {
		turnBudget := MinTurnBytes + int(turnBytes16)%2048
		policy := HistoryPolicy{Limits: Limits{
			MaxPriorTurns: 4, MaxTurnBytes: turnBudget,
			MaxInputBytes: 5 * turnBudget,
		}, IncludeAssistant: true}
		messages := conversation(exchange(prior, prior), user(live))
		prepared := prepare(context.Background(), messages, true, policy)
		if prepared.InputBytes > policy.Limits.MaxInputBytes {
			t.Fatalf("input bytes %d", prepared.InputBytes)
		}
		for _, turn := range append([]evidenceTurn{prepared.Live}, prepared.Prior...) {
			for _, segment := range append(append([]textSegment{}, turn.User...), turn.Assistant...) {
				if utf8.ValidString(prior+live) && !utf8.ValidString(string(segment)) {
					t.Fatalf("segment split a rune: %q", segment)
				}
			}
		}
		result := classify(defaultConfig(policy), prepared, extract(context.Background(), prepared))
		assertInvariants(t, result)
	})
}

// The live turn fills the whole aggregate budget, so the prior turn must be
// rejected by the upper-bound check, which reads no text, before buildTurn.
func TestAggregateBudgetIsCheckedBeforeBuildingATurn(t *testing.T) {
	policy := HistoryPolicy{
		Limits:           Limits{MaxPriorTurns: 4, MaxTurnBytes: MinInputBytes, MaxInputBytes: MinInputBytes},
		IncludeAssistant: true,
	}
	full := strings.Repeat("parser buffer loop ", 200)
	messages := conversation(exchange(full, full), user(full))
	prepared := prepare(context.Background(), messages, true, policy)
	state := &prepareState{ctx: context.Background(), policy: policy}
	if prepared.InputBytes+state.retainedUpperBound(messages[:2]) <= policy.Limits.MaxInputBytes {
		t.Fatal("precondition: the prior turn must not fit the remaining budget")
	}
	if len(prepared.Prior) != 0 || prepared.Coverage != CoveragePartial {
		t.Fatalf("prior=%d coverage=%s, want the prior turn dropped and partial coverage",
			len(prepared.Prior), prepared.Coverage)
	}
	if prepared.InputBytes > policy.Limits.MaxInputBytes {
		t.Fatalf("input bytes %d exceed the aggregate budget %d", prepared.InputBytes, policy.Limits.MaxInputBytes)
	}
}

// A turn the upper bound admits must fit once built; otherwise the aggregate
// budget could be exceeded.
func TestRetainedUpperBoundNeverUnderestimates(t *testing.T) {
	for _, includeAssistant := range []bool{true, false} {
		policy := HistoryPolicy{
			Limits:           Limits{MaxPriorTurns: 4, MaxTurnBytes: 512, MaxInputBytes: 4096},
			IncludeAssistant: includeAssistant,
		}
		state := &prepareState{ctx: context.Background(), policy: policy}
		calls := make([]llmprotocol.Content, 20)
		for i := range calls {
			calls[i] = toolCall(fmt.Sprintf("c%d", i), strings.Repeat("n", 70), "")
		}
		for _, turn := range [][]llmprotocol.Message{
			exchange("short question", "short answer"),
			exchange(strings.Repeat("ą", 400), strings.Repeat("b", 400)),
			{user("run"), assistantCalls(calls...), assistant("done")},
		} {
			if built, bound := turnBytes(state.buildTurn(turn, false)), state.retainedUpperBound(turn); built > bound {
				t.Fatalf("include_assistant=%v: built %d bytes but bound was %d", includeAssistant, built, bound)
			}
		}
	}
}
