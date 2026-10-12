package topiccontinuity

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type corpusTurn struct {
	Role string `yaml:"role"`
	Text string `yaml:"text"`
	ID   string `yaml:"id"`
	Name string `yaml:"name"`
}

type corpusConversation struct {
	ID                string       `yaml:"id"`
	Label             string       `yaml:"label"`
	Subset            string       `yaml:"subset"`
	LongAssistantCode bool         `yaml:"long_assistant_code"`
	LongUserPrior     bool         `yaml:"long_user_prior"`
	Turns             []corpusTurn `yaml:"turns"`
}

func loadCorpus(t testing.TB) []corpusConversation {
	t.Helper()
	raw, err := os.ReadFile("testdata/corpus.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Conversations []corpusConversation `yaml:"conversations"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	return doc.Conversations
}

func longCode() string {
	var builder strings.Builder
	for i := 0; builder.Len() < 12*1024; i++ {
		fmt.Fprintf(&builder, "func handleUser%d(w http.ResponseWriter, r *http.Request) { serve(w, r) }\n", i)
	}
	return builder.String()
}

func (c corpusConversation) messages() []llmprotocol.Message {
	var out []llmprotocol.Message
	for _, turn := range c.Turns {
		value := turn.Text
		if value == "LONG_CODE" && c.LongAssistantCode {
			value = longCode()
		}
		if value == "LONG_USER" && c.LongUserPrior {
			// Larger than the default per-turn budget, so coverage is partial.
			value = strings.Repeat("The export job failed again overnight with the same timeout. ", 400)
		}
		switch turn.Role {
		case "user":
			out = append(out, user(value))
		case "assistant":
			out = append(out, assistant(value))
		case "call":
			out = append(out, assistantCalls(toolCall(turn.ID, turn.Name, `{"k":"v"}`)))
		case "result":
			out = append(out, toolResult(turn.ID, value))
		default:
			panic("unknown corpus role " + turn.Role)
		}
	}
	return out
}

type corpusOutcome struct {
	conversation corpusConversation
	result       Result
}

func runCorpus(t testing.TB) []corpusOutcome {
	conversations := loadCorpus(t)
	outcomes := make([]corpusOutcome, 0, len(conversations))
	for _, conversation := range conversations {
		outcomes = append(outcomes, corpusOutcome{conversation, evaluate(conversation.messages(), defaultPolicy)})
	}
	return outcomes
}

// TestCorpusGates enforces the regression gates:
//   - no explicit-marker change on any conversation labeled continuation;
//   - no change of any kind on the adversarial subset, whose marker phrases sit
//     in quotes, code, negations, or late positions.
//
// Every result must also satisfy the schema invariants.
func TestCorpusGates(t *testing.T) {
	for _, outcome := range runCorpus(t) {
		c, r := outcome.conversation, outcome.result
		assertInvariants(t, r)
		if c.Label == "continuation" && r.Reason == ReasonExplicitChange {
			t.Errorf("%s: explicit change on a labeled continuation", c.ID)
		}
		if c.Subset == "adversarial" && r.Class == ClassChange {
			t.Errorf("%s: adversarial case produced %s", c.ID, r.Reason)
		}
	}
}

// TestCorpusReport logs heuristic distributions for calibration;
// TestCorpusGates owns pass/fail regressions.
func TestCorpusReport(t *testing.T) {
	outcomes := runCorpus(t)
	coverage := map[Coverage]int{}
	reasons := map[Reason]int{}
	labelClass := map[string]int{}
	var disjointOnContinuation []string
	for _, outcome := range outcomes {
		c, r := outcome.conversation, outcome.result
		coverage[r.Coverage]++
		reasons[r.Reason]++
		labelClass[c.Label+" -> "+string(r.Class)]++
		if c.Label == "continuation" && r.Reason == ReasonDisjoint {
			disjointOnContinuation = append(disjointOnContinuation, c.ID)
		}
	}
	t.Logf("conversations: %d", len(outcomes))
	t.Logf("disjoint change on labeled continuations: %d %v", len(disjointOnContinuation), disjointOnContinuation)
	t.Logf("coverage: %v", coverage)
	t.Logf("label -> class: %s", sortedCounts(labelClass))
	t.Logf("reasons: %s", sortedCounts(reasons))
}

func sortedCounts[K ~string](counts map[K]int) string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, string(key))
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, counts[K(key)]))
	}
	return strings.Join(parts, ", ")
}
