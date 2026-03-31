package nlgen

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

// mockLLMClient returns canned responses for testing the NL pipeline.
type mockLLMClient struct {
	responses []string
	callIdx   int
	calls     []ChatCompletionRequest
}

func (m *mockLLMClient) ChatCompletion(_ context.Context, req ChatCompletionRequest) (string, error) {
	m.calls = append(m.calls, req)
	if m.callIdx >= len(m.responses) {
		return "", fmt.Errorf("no more mock responses")
	}
	resp := m.responses[m.callIdx]
	m.callIdx++
	return resp, nil
}

func TestGenerateFromNL_CleanOutput(t *testing.T) {
	client := &mockLLMClient{
		responses: []string{
			`SIGNAL domain math {
  description: "Math queries"
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5:7b"
}

ROUTE default {
  PRIORITY 1
  MODEL "qwen2.5:3b"
}`,
		},
	}

	result, err := GenerateFromNL(context.Background(), client, "Route math queries to qwen2.5:7b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Attempts != 1 {
		t.Errorf("expected 1 attempt, got %d", result.Attempts)
	}
	if result.ParseError != "" {
		t.Errorf("unexpected parse error: %s", result.ParseError)
	}
	if !strings.Contains(result.DSL, "ROUTE") {
		t.Errorf("expected DSL to contain ROUTE, got: %s", result.DSL)
	}
}

func TestGenerateFromNL_FencedOutput(t *testing.T) {
	client := &mockLLMClient{
		responses: []string{
			"Here is your DSL:\n```dsl\nSIGNAL domain math {\n  description: \"Math\"\n}\n\nROUTE math {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"qwen\"\n}\n\nROUTE default {\n  PRIORITY 1\n  MODEL \"qwen\"\n}\n```\nThis routes math queries.",
		},
	}

	result, err := GenerateFromNL(context.Background(), client, "Route math queries")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ParseError != "" {
		t.Errorf("unexpected parse error: %s", result.ParseError)
	}
	if strings.Contains(result.DSL, "```") {
		t.Errorf("DSL should not contain code fences: %s", result.DSL)
	}
}

func TestGenerateFromNL_RepairLoop(t *testing.T) {
	client := &mockLLMClient{
		responses: []string{
			"ROUTE math {\n  PRIORITY 100\n  WHEN domain(\"math\")\n",
			"SIGNAL domain math {\n  description: \"Math\"\n}\n\nROUTE math {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"qwen\"\n}\n\nROUTE default {\n  PRIORITY 1\n  MODEL \"qwen\"\n}",
		},
	}

	result, err := GenerateFromNL(context.Background(), client, "Route math queries")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Attempts != 2 {
		t.Errorf("expected 2 attempts, got %d", result.Attempts)
	}
	if result.ParseError != "" {
		t.Errorf("unexpected parse error after repair: %s", result.ParseError)
	}
	if len(client.calls) < 2 {
		t.Fatal("expected at least 2 LLM calls")
	}
	repairPrompt := client.calls[1].Messages[1].Content
	if !strings.Contains(repairPrompt, "Semantic Router DSL Reference") {
		t.Error("repair prompt should include schema reference")
	}
}

func TestGenerateFromNL_ExhaustedRetries(t *testing.T) {
	client := &mockLLMClient{
		responses: []string{
			"ROUTE broken {",
			"ROUTE still_broken {",
			"ROUTE very_broken {",
		},
	}

	result, err := GenerateFromNL(context.Background(), client, "something", WithMaxRetries(2))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Attempts != 3 {
		t.Errorf("expected 3 attempts, got %d", result.Attempts)
	}
	if result.ParseError == "" {
		t.Error("expected parse error after exhausted retries")
	}
}

func TestGenerateFromNL_UnquotedSignalRef(t *testing.T) {
	client := &mockLLMClient{
		responses: []string{
			`SIGNAL domain math {
  description: "Math"
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain(math)
  MODEL "qwen"
}

ROUTE default {
  PRIORITY 1
  MODEL "qwen"
}`,
		},
	}

	result, err := GenerateFromNL(context.Background(), client, "Route math queries")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ParseError != "" {
		t.Errorf("unquoted signal ref should parse successfully, got: %s", result.ParseError)
	}
}

func TestGenerateFromNL_SystemPromptContainsSchema(t *testing.T) {
	client := &mockLLMClient{
		responses: []string{
			"SIGNAL domain math { description: \"m\" }\nROUTE r { PRIORITY 1\n  MODEL \"q\" }",
		},
	}

	_, err := GenerateFromNL(context.Background(), client, "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(client.calls) == 0 {
		t.Fatal("expected at least 1 LLM call")
	}

	sysMsg := client.calls[0].Messages[0].Content
	if !strings.Contains(sysMsg, "Semantic Router DSL generator") {
		t.Error("system message should identify as DSL generator")
	}

	userMsg := client.calls[0].Messages[1].Content
	if !strings.Contains(userMsg, "Semantic Router DSL Reference") {
		t.Error("user message should contain schema reference")
	}
	if !strings.Contains(userMsg, "Example 1") {
		t.Error("user message should contain few-shot examples")
	}
}

// ---------- Eval Harness ----------

// NLEvalCase defines one test case for the NL-to-DSL eval harness.
type NLEvalCase struct {
	Name           string
	Instruction    string
	MustContain    []string
	MustNotContain []string
	MustParse      bool
}

// NLEvalResult records the outcome of running one eval case.
type NLEvalResult struct {
	Case       NLEvalCase
	DSL        string
	Passed     bool
	ParseError string
	Failures   []string
}

// RunNLEval runs the eval harness against a set of test cases using the given LLM client.
func RunNLEval(ctx context.Context, client LLMClient, cases []NLEvalCase, opts ...NLOption) ([]NLEvalResult, float64) {
	results := make([]NLEvalResult, len(cases))
	passed := 0

	for i, tc := range cases {
		r := NLEvalResult{Case: tc}

		nlResult, err := GenerateFromNL(ctx, client, tc.Instruction, opts...)
		if err != nil {
			r.Failures = append(r.Failures, fmt.Sprintf("generation error: %v", err))
			results[i] = r
			continue
		}

		r.DSL = nlResult.DSL
		r.ParseError = nlResult.ParseError

		if tc.MustParse && nlResult.ParseError != "" {
			r.Failures = append(r.Failures, fmt.Sprintf("parse error: %s", nlResult.ParseError))
		}

		for _, substr := range tc.MustContain {
			if !strings.Contains(nlResult.DSL, substr) {
				r.Failures = append(r.Failures, fmt.Sprintf("missing required substring: %q", substr))
			}
		}

		for _, substr := range tc.MustNotContain {
			if strings.Contains(nlResult.DSL, substr) {
				r.Failures = append(r.Failures, fmt.Sprintf("contains forbidden substring: %q", substr))
			}
		}

		r.Passed = len(r.Failures) == 0
		if r.Passed {
			passed++
		}
		results[i] = r
	}

	rate := 0.0
	if len(cases) > 0 {
		rate = float64(passed) / float64(len(cases))
	}
	return results, rate
}

// StandardEvalCases returns a set of canonical eval cases covering common routing patterns.
func StandardEvalCases() []NLEvalCase {
	return []NLEvalCase{
		{
			Name:        "simple_domain_routing",
			Instruction: "Route math questions to qwen2.5-math:7b and everything else to qwen2.5:3b",
			MustContain: []string{"SIGNAL domain", "ROUTE", "MODEL", "math"},
			MustParse:   true,
		},
		{
			Name:        "jailbreak_safety",
			Instruction: "Block jailbreak attempts with a fast rejection model, route everything else to gpt-4o",
			MustContain: []string{"SIGNAL jailbreak", "jailbreak", "MODEL"},
			MustParse:   true,
		},
		{
			Name:        "keyword_boolean_routing",
			Instruction: "Route queries containing 'urgent' AND 'billing' keywords to gpt-4o, queries with just 'billing' to qwen2.5:7b, and everything else to qwen2.5:3b",
			MustContain: []string{"SIGNAL keyword", "AND", "MODEL"},
			MustParse:   true,
		},
		{
			Name:           "decision_tree_routing",
			Instruction:    "Use a decision tree: if jailbreak detected, block with fast-reject model. Else if math domain, use qwen-math. Else if coding domain, use deepseek-coder. Otherwise use qwen2.5:7b as default.",
			MustContain:    []string{"DECISION_TREE", "IF", "ELSE", "MODEL"},
			MustNotContain: []string{"ROUTE"},
			MustParse:      true,
		},
		{
			Name:        "embedding_with_or",
			Instruction: "Route queries about AI or machine learning (using embedding similarity) to gpt-4o, everything else to qwen2.5:3b",
			MustContain: []string{"SIGNAL embedding", "ROUTE", "MODEL"},
			MustParse:   true,
		},
		{
			Name:        "multi_model_with_algorithm",
			Instruction: "For complex math queries, use both qwen2.5-math:72b and deepseek-r1:70b with confidence-based model selection. For simple queries, use qwen2.5:3b.",
			MustContain: []string{"ALGORITHM", "MODEL", "ROUTE"},
			MustParse:   true,
		},
		{
			Name:        "plugin_configuration",
			Instruction: "Route coding queries to deepseek-coder with a system prompt saying 'You are an expert programmer', and enable RAG with top_k=5 for knowledge base queries. Default to qwen2.5:7b.",
			MustContain: []string{"PLUGIN system_prompt", "ROUTE", "MODEL"},
			MustParse:   true,
		},
		{
			Name:        "pii_and_authz",
			Instruction: "Block queries containing PII (threshold 0.9). For admin users, route to gpt-4o. For everyone else, use qwen2.5:7b.",
			MustContain: []string{"SIGNAL pii", "ROUTE", "MODEL"},
			MustParse:   true,
		},
	}
}

func TestRunNLEval_WithMockClient(t *testing.T) {
	validDSL := `SIGNAL domain math {
  description: "Math queries"
}

ROUTE math_route {
  PRIORITY 100
  WHEN domain("math")
  MODEL "qwen2.5-math:7b"
}

ROUTE default {
  PRIORITY 1
  MODEL "qwen2.5:3b"
}`

	client := &mockLLMClient{
		responses: []string{validDSL},
	}

	cases := []NLEvalCase{
		{
			Name:        "test_case",
			Instruction: "Route math to qwen",
			MustContain: []string{"SIGNAL domain", "ROUTE", "MODEL", "math"},
			MustParse:   true,
		},
	}

	results, rate := RunNLEval(context.Background(), client, cases)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if !results[0].Passed {
		t.Errorf("expected test to pass, failures: %v", results[0].Failures)
	}
	if rate != 1.0 {
		t.Errorf("expected 100%% pass rate, got %v", rate)
	}
}

func TestFewShotExamplesParse(t *testing.T) {
	examples := splitFewShotExamples(FewShotExamples)
	if len(examples) == 0 {
		t.Fatal("no examples found in FewShotExamples")
	}

	for i, example := range examples {
		example = strings.TrimSpace(example)
		if example == "" {
			continue
		}
		_, errs := dsl.Parse(example)
		if len(errs) > 0 {
			t.Errorf("few-shot example %d failed to parse: %v\n---\n%s\n---", i+1, errs, example)
		}
	}
}

func splitFewShotExamples(corpus string) []string {
	var examples []string
	parts := strings.Split(corpus, "# Example ")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if idx := strings.Index(part, "\n"); idx >= 0 {
			code := strings.TrimSpace(part[idx+1:])
			if code != "" {
				examples = append(examples, code)
			}
		}
	}
	return examples
}
