package classification

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

func privacyPolicyClassifier(t *testing.T) *Classifier {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "recipes", "privacy", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	c := &Classifier{Config: cfg}
	c.keywordClassifier, err = NewKeywordClassifier(cfg.KeywordRules)
	if err != nil {
		t.Fatal(err)
	}
	c.structureClassifier, err = NewStructureClassifier(cfg.StructureRules)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func privacyPolicySignals(c *Classifier, text string) *SignalResults {
	in := &SignalResults{Metrics: &SignalMetricsCollection{}, SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}}
	var mu sync.Mutex
	c.evaluateKeywordSignal(in, &mu, text)
	c.evaluateStructureSignal(in, &mu, text)
	return in
}

// The learned observations below are controlled component inputs, not model
// predictions. The actual shipped heuristic, projection and decision chain runs.
func TestPrivacyRecipeLocalityOverridesEffort(t *testing.T) {
	c := privacyPolicyClassifier(t)
	for _, tc := range []struct {
		name, text string
		local      bool
	}{
		{"direct processing", "Process this locally.", true},
		{"backend instruction", "Use only the local model for this request.", true},
		{"transfer prohibition", "Do not upload this content to an external service.", true},
		{"operational requirement", "This request must remain on-premises.", true},
		{"bounded processing", "Summarize this without sending any content outside.", true},
		{"Chinese processing", "请只在本地处理这段内容。", true},
		{"Chinese placement before action", "请在本地处理这段内容。", true},
		{"Chinese prohibition", "不要把这些内容发送到外部服务。", true},
		{"Chinese destination before action", "不要向外部发送这些内容。", true},
		{"quoted phrase before real constraint", `Translate "keep this local", but process this request locally.`, true},
		{"conjoined real constraint", `Translate "hello" and keep this task local.`, true},
		{"real constraint before quotation", `Keep this task local. Translate "hello".`, true},
		{"Chinese mixed request", "解释“仅在本地”这个短语，但只在本地处理当前请求。", true},
		{"translation as data", "Translate: Keep this task local.", false},
		{"quoted data", `Translate "keep this local".`, false},
		{"phrase explanation", `Explain the phrase "local processing only".`, false},
		{"Chinese explanation", "解释“只在本地处理”的含义。", false},
		{"quoted review object", `Review the spelling of "process this locally".`, false},
		{"programming scope", "Use a local variable in the example.", false},
		{"ordinary processing", "Process a short example.", false},
		{"sentence boundary inside quote", `Translate "A sentence. Keep this task local.".`, false},
		{"conjunction inside quote", `Translate "Try once and keep this task local".`, false},
		{"single quoted review object", `Review the spelling of 'process this locally'.`, false},
		{"Chinese quoted sentence", "解释“第一句。只在本地处理。”的含义。", false},
		{"single quoted prefix before real constraint", `Translate 'hello', but keep this task local.`, true},
		{"ordinary contraction before real constraint", "I don't need a summary. Process this locally.", true},
		{"explicit contraction prohibition", "Don't send this to an external service.", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := privacyPolicySignals(c, tc.text)
			if got := slices.Contains(in.MatchedStructureRules, "local_handling_required"); got != tc.local {
				t.Fatalf("locality=%v want %v", got, tc.local)
			}
			// Independent effort is strong in both controls. Locality alone must
			// still veto external selection; quoted language must not forge it.
			in.SignalValues["embedding:frontier_reasoning_request"] = 0.6
			in.MatchedKeywordRules = append(in.MatchedKeywordRules, "reasoning_request_markers")
			in.SignalConfidences["keyword:reasoning_request_markers"] = 1
			want := "cloud_frontier_reasoning"
			if tc.local {
				want = "local_privacy_policy"
			}
			assertPrivacyDecision(t, c, in, want)
		})
	}
}

func assertPrivacyDecision(t *testing.T, c *Classifier, in *SignalResults, want string) {
	t.Helper()
	result, err := c.EvaluateDecisionWithEngine(c.applyProjections(in))
	if err != nil || result == nil || result.Decision == nil || result.Decision.Name != want {
		t.Fatalf("want %s; result=%+v error=%v", want, result, err)
	}
}

func TestPrivacyRecipeContinuousEffortAndErrors(t *testing.T) {
	c := privacyPolicyClassifier(t)
	for _, tc := range []struct {
		name    string
		raw     float64
		request bool
		want    string
	}{
		{"raw alone cannot escalate", 1, false, "local_standard"},
		{"independent effort corroborates raw", 0.6, true, "cloud_frontier_reasoning"},
		{"request alone is insufficient", 0, true, "local_standard"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := privacyPolicySignals(c, "A neutral item.")
			in.SignalValues["embedding:frontier_reasoning_request"] = tc.raw
			if tc.request {
				in.MatchedKeywordRules = []string{"reasoning_request_markers"}
				in.SignalConfidences["keyword:reasoning_request_markers"] = 1
			}
			assertPrivacyDecision(t, c, in, tc.want)
			if slices.Contains(in.MatchedEmbeddingRules, "frontier_reasoning_request") {
				t.Fatal("raw projection forged a predicate match")
			}
		})
	}
	in := privacyPolicySignals(c, "A neutral item.")
	in.SignalValues["embedding:frontier_reasoning_request"] = 0.75
	in.MatchedKeywordRules = []string{"architecture_markers"}
	in.SignalConfidences["keyword:architecture_markers"] = 0.5
	assertPrivacyDecision(t, c, in, "cloud_frontier_reasoning")
	if math.Abs(in.ProjectionScores["reasoning_pressure"]-0.5) > 1e-12 {
		t.Fatal("the existing inclusive boundary changed")
	}
	for _, key := range []string{"embedding:frontier_reasoning_request", "pii:pii_strict", "jailbreak:jailbreak_strict"} {
		t.Run(key, func(t *testing.T) {
			in := privacyPolicySignals(c, "A neutral item.")
			in.SignalValues["embedding:frontier_reasoning_request"] = 1
			in.MatchedKeywordRules = []string{"reasoning_request_markers"}
			in.SignalErrors = map[string]string{key: "unavailable"}
			result, err := c.EvaluateDecisionWithEngine(c.applyProjections(in))
			if result != nil || !errors.Is(err, decision.ErrDecisionUnresolved) {
				t.Fatalf("unknown evidence must reject before provider dispatch: result=%+v error=%v", result, err)
			}
		})
	}
}

func TestPrivacyRecipeExistingSensitivityAndPriority(t *testing.T) {
	c := privacyPolicyClassifier(t)
	in := privacyPolicySignals(c, "Process this locally.")
	assertPrivacyDecision(t, c, in, "local_privacy_policy")
	in = privacyPolicySignals(c, "A neutral item.")
	in.MatchedPIIRules = []string{"pii_strict"}
	assertPrivacyDecision(t, c, in, "local_privacy_policy")
	in = privacyPolicySignals(c, "Process this locally.")
	in.MatchedJailbreakRules = []string{"jailbreak_strict"}
	assertPrivacyDecision(t, c, in, "local_security_containment")
}

func TestPrivacyRecipeKnownRestrictionsSurviveUnrelatedUnknown(t *testing.T) {
	c := privacyPolicyClassifier(t)
	for _, key := range []string{"embedding:frontier_reasoning_request", "pii:pii_strict"} {
		t.Run(key, func(t *testing.T) {
			in := privacyPolicySignals(c, "Process this locally.")
			in.SignalErrors = map[string]string{key: "unavailable"}
			assertPrivacyDecision(t, c, in, "local_privacy_policy")
			if in.SignalErrors[key] != "unavailable" {
				t.Fatal("independent local handling hid the failed signal")
			}
			in.MatchedJailbreakRules = []string{"jailbreak_strict"}
			assertPrivacyDecision(t, c, in, "local_security_containment")
		})
	}
}
