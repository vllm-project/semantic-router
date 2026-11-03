package classification

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	testAWSKeyPattern = `AKIA[0-9A-Z]{16}`
	// AKIAIOSFODNN7EXAMPLE is the example access key id from the AWS
	// documentation, not a live credential.
	testAWSKeyText = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
)

// keywordSignalsFor evaluates every signal for one request through the public
// entry point, with the given tool results, and returns the results.
func keywordSignalsFor(t *testing.T, rules []config.KeywordRule, input SignalEvaluationInput) *SignalResults {
	t.Helper()
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = rules
	classifier := &Classifier{Config: cfg, keywordClassifier: kc}
	input.ForceEvaluateAll = true
	if input.CurrentUserText == "" {
		input.CurrentUserText = input.Text
	}
	results, err := classifier.EvaluateAllSignalsWithHeaders(input)
	if err != nil {
		t.Fatalf("EvaluateAllSignalsWithHeaders: %v", err)
	}
	return results
}

func credentialRule(operator string, keywords ...string) config.KeywordRule {
	return config.KeywordRule{
		Name:          "credentials",
		Operator:      operator,
		Method:        "regex",
		CaseSensitive: true,
		Keywords:      keywords,
		Source:        config.KeywordSourceToolResult,
	}
}

// A source: tool_result rule reads the tool results and nothing else: the same
// key in the prompt or in an earlier user message does not match.
func TestKeywordToolResultSourceMatchesToolResultsOnly(t *testing.T) {
	rules := []config.KeywordRule{credentialRule("OR", testAWSKeyPattern)}

	got := keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "read the file", ToolResultTexts: []string{testAWSKeyText},
	})
	if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("tool result: MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}
	if want := []string{testAWSKeyPattern}; !reflect.DeepEqual(got.MatchedKeywords, want) {
		t.Fatalf("MatchedKeywords = %v, want the configured pattern %v, never the matched text", got.MatchedKeywords, want)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{Text: testAWSKeyText})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("prompt: MatchedKeywordRules = %v, want none", got.MatchedKeywordRules)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "continue", PriorUserMessages: []string{testAWSKeyText},
	})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("history: MatchedKeywordRules = %v, want none", got.MatchedKeywordRules)
	}
}

// Any one tool result may carry the match.
func TestKeywordToolResultSourceMatchesAnyToolResult(t *testing.T) {
	got := keywordSignalsFor(t, []config.KeywordRule{credentialRule("OR", testAWSKeyPattern)}, SignalEvaluationInput{
		Text:            "what did the tools return",
		ToolResultTexts: []string{"total 2\n-rw-r--r-- 1 user user 12 .env", testAWSKeyText, "ok"},
	})
	if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}
}

// Rules with other scopes keep their own content: a default rule still reads
// the prompt, and the prompt never satisfies the tool-result rule.
func TestKeywordToolResultSourceIsPerRule(t *testing.T) {
	rules := []config.KeywordRule{
		credentialRule("OR", testAWSKeyPattern),
		{Name: "reasoning", Operator: "OR", Keywords: []string{"step by step"}},
	}
	got := keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text:            "explain step by step: " + testAWSKeyText,
		ToolResultTexts: []string{"nothing secret here"},
	})
	if want := []string{"reasoning"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}
}

// Each tool result is matched on its own, so an AND rule cannot be assembled
// from two tool results.
func TestKeywordToolResultSourceANDDoesNotSpanToolResults(t *testing.T) {
	rules := []config.KeywordRule{credentialRule("AND", `-----BEGIN RSA PRIVATE KEY-----`, `-----END RSA PRIVATE KEY-----`)}

	got := keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text:            "show the key",
		ToolResultTexts: []string{"-----BEGIN RSA PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"},
	})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("split across tool results: MatchedKeywordRules = %v, want none", got.MatchedKeywordRules)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text:            "show the key",
		ToolResultTexts: []string{"-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"},
	})
	if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("one tool result: MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}
}

// NOR matches only when every tool result is free of the keywords, and never
// when the request carries no tool results at all.
func TestKeywordToolResultSourceNORNeedsCleanToolResults(t *testing.T) {
	rules := []config.KeywordRule{credentialRule("NOR", testAWSKeyPattern)}

	got := keywordSignalsFor(t, rules, SignalEvaluationInput{Text: "hello"})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("no tool results: MatchedKeywordRules = %v, want none (NOR must not match vacuously)", got.MatchedKeywordRules)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "hello", ToolResultTexts: []string{"clean", "also clean"},
	})
	if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("clean tool results: MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "hello", ToolResultTexts: []string{"clean", testAWSKeyText},
	})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("one dirty tool result: MatchedKeywordRules = %v, want none", got.MatchedKeywordRules)
	}
}

// Scored rules follow the same source selection.
// When part of the tool-result content was not inspectable, a tool-result rule
// that did not match is reported as incomplete rather than clean; a rule that
// matched stands.
func TestKeywordToolResultSourceIncompleteScanIsReported(t *testing.T) {
	rules := []config.KeywordRule{credentialRule("OR", testAWSKeyPattern)}
	key := signalConfidenceKey(config.SignalTypeKeyword, "credentials")

	got := keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "look at the screenshot", ToolResultTexts: []string{"no text secret"}, ToolResultScanIncomplete: true,
	})
	if code := got.SignalErrors[key]; code != keywordEvaluationIncompleteCode {
		t.Fatalf("SignalErrors[%q] = %q, want %q", key, code, keywordEvaluationIncompleteCode)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "look at the screenshot", ToolResultTexts: []string{testAWSKeyText}, ToolResultScanIncomplete: true,
	})
	if _, failed := got.SignalErrors[key]; failed {
		t.Fatalf("matched rule must not be reported incomplete: SignalErrors = %v", got.SignalErrors)
	}
	if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}

	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "plain request", ToolResultScanIncomplete: false,
	})
	if _, failed := got.SignalErrors[key]; failed {
		t.Fatalf("complete scan must not report an error: SignalErrors = %v", got.SignalErrors)
	}
}

// The single-text classification APIs receive user content, which a
// tool-result rule never matches.
func TestKeywordClassifierSingleTextSkipsToolResultRules(t *testing.T) {
	kc, err := NewKeywordClassifier([]config.KeywordRule{credentialRule("OR", testAWSKeyPattern)})
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	if category, _, classifyErr := kc.Classify(testAWSKeyText); classifyErr != nil || category != "" {
		t.Fatalf("Classify() = %q, %v; want no category", category, classifyErr)
	}
	matches, err := kc.MatchAll(testAWSKeyText)
	if err != nil || len(matches) != 0 {
		t.Fatalf("MatchAll() = %v, %v; want no matches", matches, err)
	}
	if want := []string{"credentials"}; !reflect.DeepEqual(kc.ToolResultRuleNames(), want) {
		t.Fatalf("ToolResultRuleNames() = %v, want %v", kc.ToolResultRuleNames(), want)
	}
}
