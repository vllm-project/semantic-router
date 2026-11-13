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

// credentialRule reads the tool results and not the current user turn, as
// the credentials fragment does.
func credentialRule(operator string, keywords ...string) config.KeywordRule {
	no := false
	return config.KeywordRule{
		Name:          "credentials",
		Operator:      operator,
		Method:        "regex",
		CaseSensitive: true,
		Keywords:      keywords,
		Input:         config.KeywordRuleInput{ToolResults: true, Request: &no},
	}
}

// A rule with input.request: false reads the tool results and nothing else:
// the same key in the prompt or in an earlier user message does not match.
func TestKeywordToolResultInputMatchesToolResultsOnly(t *testing.T) {
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

// With input.request left on, the rule reads the current turn and the tool
// results alike.
func TestKeywordToolResultInputBesideTheRequest(t *testing.T) {
	rule := credentialRule("OR", testAWSKeyPattern)
	rule.Input.Request = nil
	rules := []config.KeywordRule{rule}

	for name, input := range map[string]SignalEvaluationInput{
		"in the prompt":      {Text: testAWSKeyText, ToolResultTexts: []string{"clean"}},
		"in a tool result":   {Text: "read the file", ToolResultTexts: []string{testAWSKeyText}},
		"without tool calls": {Text: testAWSKeyText},
	} {
		got := keywordSignalsFor(t, rules, input)
		if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
			t.Fatalf("%s: MatchedKeywordRules = %v, want %v", name, got.MatchedKeywordRules, want)
		}
	}
	got := keywordSignalsFor(t, rules, SignalEvaluationInput{Text: "continue", PriorUserMessages: []string{testAWSKeyText}})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("history without include_history: MatchedKeywordRules = %v, want none", got.MatchedKeywordRules)
	}
}

// Any one tool result may carry the match.
func TestKeywordToolResultInputMatchesAnyToolResult(t *testing.T) {
	got := keywordSignalsFor(t, []config.KeywordRule{credentialRule("OR", testAWSKeyPattern)}, SignalEvaluationInput{
		Text:            "what did the tools return",
		ToolResultTexts: []string{"total 2\n-rw-r--r-- 1 user user 12 .env", testAWSKeyText, "ok"},
	})
	if want := []string{"credentials"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}
}

// Rules keep their own input: a default rule still reads the prompt and
// never the tool results, and the prompt never satisfies the tool-result rule.
func TestKeywordToolResultInputIsPerRule(t *testing.T) {
	rules := []config.KeywordRule{
		credentialRule("OR", testAWSKeyPattern),
		{Name: "reasoning", Operator: "OR", Keywords: []string{"step by step"}},
	}
	got := keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text:            "explain step by step: " + testAWSKeyText,
		ToolResultTexts: []string{"nothing secret here, step by step"},
	})
	if want := []string{"reasoning"}; !reflect.DeepEqual(got.MatchedKeywordRules, want) {
		t.Fatalf("MatchedKeywordRules = %v, want %v", got.MatchedKeywordRules, want)
	}
	got = keywordSignalsFor(t, rules, SignalEvaluationInput{
		Text: "explain", ToolResultTexts: []string{"step by step"},
	})
	if len(got.MatchedKeywordRules) != 0 {
		t.Fatalf("default rule read a tool result: MatchedKeywordRules = %v, want none", got.MatchedKeywordRules)
	}
}

// Each tool result is matched on its own, so an AND rule cannot be assembled
// from two tool results.
func TestKeywordToolResultInputANDDoesNotSpanToolResults(t *testing.T) {
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
func TestKeywordToolResultInputNORNeedsCleanToolResults(t *testing.T) {
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

// A tool result the Router could not read as text is decided by on_unscanned
// when the text read holds no keyword: best_effort decides on the text read,
// match matches the rule even when no tool result holds text, and no_match
// does not match it. A keyword found counts whatever the policy says, and a
// NOR rule's strict policy is no_match, which refuses to read the unread
// tool result as clean.
func TestKeywordToolResultInputOnUnscanned(t *testing.T) {
	withPolicy := func(operator, policy string) []config.KeywordRule {
		rule := credentialRule(operator, testAWSKeyPattern)
		rule.OnUnscanned = policy
		return []config.KeywordRule{rule}
	}
	unread := SignalEvaluationInput{Text: "look at the screenshot", ToolResultTexts: []string{"no text secret"}, ToolResultScanIncomplete: true}
	unreadAndFound := SignalEvaluationInput{Text: "look at the screenshot", ToolResultTexts: []string{testAWSKeyText}, ToolResultScanIncomplete: true}
	onlyUnread := SignalEvaluationInput{Text: "look at the screenshot", ToolResultScanIncomplete: true}
	matched := []string{"credentials"}

	for name, test := range map[string]struct {
		rules []config.KeywordRule
		input SignalEvaluationInput
		want  []string
	}{
		"default decides on the text read":      {withPolicy("OR", ""), unread, nil},
		"default keeps a found key":             {withPolicy("OR", ""), unreadAndFound, matched},
		"best_effort decides on the text read":  {withPolicy("OR", config.KeywordUnscannedBestEffort), unread, nil},
		"match matches":                         {withPolicy("OR", config.KeywordUnscannedMatch), unread, matched},
		"match without any text":                {withPolicy("OR", config.KeywordUnscannedMatch), onlyUnread, matched},
		"match keeps a found key":               {withPolicy("OR", config.KeywordUnscannedMatch), unreadAndFound, matched},
		"no_match does not match":               {withPolicy("OR", config.KeywordUnscannedNoMatch), unread, nil},
		"no_match keeps a found key":            {withPolicy("OR", config.KeywordUnscannedNoMatch), unreadAndFound, matched},
		"complete scan is untouched":            {withPolicy("OR", config.KeywordUnscannedMatch), SignalEvaluationInput{Text: "plain", ToolResultTexts: []string{"clean"}}, nil},
		"NOR default reads the unread as clean": {withPolicy("NOR", ""), unread, matched},
		"NOR no_match is strict":                {withPolicy("NOR", config.KeywordUnscannedNoMatch), unread, nil},
		"NOR no_match without any text":         {withPolicy("NOR", config.KeywordUnscannedNoMatch), onlyUnread, nil},
		"NOR match reads the unread as clean":   {withPolicy("NOR", config.KeywordUnscannedMatch), unread, matched},
		"NOR keeps a found key":                 {withPolicy("NOR", config.KeywordUnscannedMatch), unreadAndFound, nil},
	} {
		got := keywordSignalsFor(t, test.rules, test.input)
		if len(test.want) == 0 && len(got.MatchedKeywordRules) != 0 || len(test.want) != 0 && !reflect.DeepEqual(got.MatchedKeywordRules, test.want) {
			t.Errorf("%s: MatchedKeywordRules = %v, want %v", name, got.MatchedKeywordRules, test.want)
		}
		if len(got.SignalErrors) != 0 {
			t.Errorf("%s: SignalErrors = %v, want none: an unread tool result is decided, not an error", name, got.SignalErrors)
		}
	}
	got := keywordSignalsFor(t, withPolicy("OR", config.KeywordUnscannedMatch), onlyUnread)
	if len(got.MatchedKeywords) != 0 {
		t.Fatalf("a match on an unread tool result names no keyword: MatchedKeywords = %v", got.MatchedKeywords)
	}
}

// input.request: false leaves the current turn out of what include_history
// reads, so such a rule reads the earlier user turns alone, and beside the
// tool results when it reads those too.
func TestKeywordInputRequestFalseKeepsHistoryWithoutTheTurn(t *testing.T) {
	no := false
	historyOnly := []config.KeywordRule{{Name: "earlier", Operator: "OR", Keywords: []string{"local model only"}, IncludeHistory: true, Input: config.KeywordRuleInput{Request: &no}}}
	both := credentialRule("OR", testAWSKeyPattern)
	both.IncludeHistory = true
	historyAndToolResults := []config.KeywordRule{both}

	for name, test := range map[string]struct {
		rules []config.KeywordRule
		input SignalEvaluationInput
		want  []string
	}{
		"earlier turn matches":             {historyOnly, SignalEvaluationInput{Text: "continue", PriorUserMessages: []string{"local model only please"}}, []string{"earlier"}},
		"current turn does not":            {historyOnly, SignalEvaluationInput{Text: "local model only please", PriorUserMessages: []string{"hello"}}, nil},
		"current turn alone does not":      {historyOnly, SignalEvaluationInput{Text: "local model only please"}, nil},
		"key in an earlier turn matches":   {historyAndToolResults, SignalEvaluationInput{Text: "continue", PriorUserMessages: []string{testAWSKeyText}, ToolResultTexts: []string{"clean"}}, []string{"credentials"}},
		"key in a tool result matches":     {historyAndToolResults, SignalEvaluationInput{Text: "continue", PriorUserMessages: []string{"hello"}, ToolResultTexts: []string{testAWSKeyText}}, []string{"credentials"}},
		"key in the current turn does not": {historyAndToolResults, SignalEvaluationInput{Text: testAWSKeyText, PriorUserMessages: []string{"hello"}, ToolResultTexts: []string{"clean"}}, nil},
	} {
		got := keywordSignalsFor(t, test.rules, test.input)
		if len(test.want) == 0 && len(got.MatchedKeywordRules) != 0 || len(test.want) != 0 && !reflect.DeepEqual(got.MatchedKeywordRules, test.want) {
			t.Errorf("%s: MatchedKeywordRules = %v, want %v", name, got.MatchedKeywordRules, test.want)
		}
	}
}

// The single-text classification APIs receive the user's turn, which a rule
// with input.request: false leaves out; a rule that reads both still reads
// the turn.
func TestKeywordClassifierSingleTextFollowsInputRequest(t *testing.T) {
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

	both := credentialRule("OR", testAWSKeyPattern)
	both.Input.Request = nil
	if kc, err = NewKeywordClassifier([]config.KeywordRule{both}); err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	if category, _, classifyErr := kc.Classify(testAWSKeyText); classifyErr != nil || category != "credentials" {
		t.Fatalf("Classify() = %q, %v; want credentials", category, classifyErr)
	}
	if matches, err = kc.MatchAll(testAWSKeyText); err != nil || len(matches) != 1 {
		t.Fatalf("MatchAll() = %v, %v; want one match", matches, err)
	}
}
