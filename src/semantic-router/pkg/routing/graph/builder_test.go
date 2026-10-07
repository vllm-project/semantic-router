package graph

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func specs(t *testing.T, text string) []StepSpec {
	t.Helper()
	var steps []StepSpec
	if err := json.Unmarshal([]byte(text), &steps); err != nil {
		t.Fatal(err)
	}
	return steps
}

// A panel, a vote and a judge that sees the panel's answers, authored as a
// configuration would author it.
const panelGraph = `[
  {"id": "panel", "type": "parallel", "configuration": {
    "max_concurrency": 2, "on_error": "skip",
    "branches": [
      [{"type": "call", "configuration": {"model": "a"}}],
      [{"type": "call", "configuration": {"model": "b"}}],
      [{"type": "subgraph", "configuration": {"name": "careful"}}]
    ]}},
  {"type": "branch", "configuration": {"cases": [
    {"when": {"signal": {"type": "domain", "name": "math"}},
     "then": [{"type": "aggregate", "configuration": {"strategy": "vote"}}]}
  ], "else": [
    {"type": "transform", "configuration": {"transformer": "prompt", "options": {
      "template": "Q: {{.Original}}{{range .Results}} [{{.Model}}] {{.Content}}{{end}}"}}},
    {"id": "judge", "type": "call", "configuration": {"model": "judge"}}
  ]}},
  {"type": "respond"}
]`

func TestBuilderBuildsAnAuthoredGraph(t *testing.T) {
	builder := NewBuilder(map[string][]StepSpec{
		"careful": specs(t, `[
		  {"type": "transform", "configuration": {"transformer": "system_prompt", "options": {"content": "Think twice."}}},
		  {"type": "call", "configuration": {"model": "c", "fields": {"temperature": 0}}}]`),
	})
	program, err := builder.Program("panel-judge", specs(t, panelGraph), Limits{MaxHops: 8})
	if err != nil {
		t.Fatal(err)
	}
	caller := &fakeCaller{answer: func(_ context.Context, req *HopRequest) (*HopResponse, error) {
		return completion(req.Model, "answer-"+req.Model, 1), nil
	}}
	outcome, err := Run(context.Background(), program, Input{Request: chatRequest(t, "2+2?")}, Options{Caller: caller})
	if err != nil {
		t.Fatal(err)
	}
	if got := answerText(t, outcome); got != "answer-judge" || outcome.Hops != 4 {
		t.Fatalf("answer %q hops %d", got, outcome.Hops)
	}
	judge := caller.requests()[3]
	parsed, _ := ParseRequest(judge.Request.Body)
	messages, _ := parsed.Messages()
	want := "Q: 2+2? [a] answer-a [b] answer-b [c] answer-c"
	if got := MessageText(messages[len(messages)-1]); got != want {
		t.Fatalf("judge prompt %q", got)
	}
	for _, sent := range caller.requests() {
		if sent.Model == "c" {
			body, _ := ParseRequest(sent.Request.Body)
			first, _ := body.Messages()
			if MessageText(first[0]) != "Think twice." || string(body.fields["temperature"]) != "0" {
				t.Fatalf("subgraph call %s", sent.Request.Body)
			}
		}
	}

	math, err := Run(context.Background(), program, Input{Request: chatRequest(t, "2+2?")},
		Options{Caller: caller, Signals: map[string][]string{"domain": {"math"}}})
	if err != nil || answerText(t, math) != "answer-a" || math.Hops != 3 {
		t.Fatalf("math: err %v hops %d", err, math.Hops)
	}
}

func TestBuilderRefusesBadGraphs(t *testing.T) {
	for name, tc := range map[string]struct {
		steps     string
		subgraphs map[string]string
		want      string
	}{
		"unknown type":       {`[{"type": "teleport"}]`, nil, `unknown step type "teleport"`},
		"unknown field":      {`[{"type": "call", "configuration": {"model": "a", "model_name": "b"}}]`, nil, "unknown field"},
		"call without model": {`[{"type": "call"}]`, nil, "model is required"},
		"duplicate id":       {`[{"id": "x", "type": "respond"}, {"id": "x", "type": "respond"}]`, nil, `step id "x" is already used`},
		"respond in parallel": {
			`[{"type": "parallel", "configuration": {"branches": [[{"type": "respond"}]]}}]`, nil,
			"cannot run inside a parallel branch",
		},
		"first_k too large": {
			`[{"type": "parallel", "configuration": {"first_k": 3, "branches": [[{"type": "call", "configuration": {"model": "a"}}]]}}]`,
			nil, "cannot exceed",
		},
		"loop without rounds":   {`[{"type": "loop", "configuration": {"body": [{"type": "respond"}]}}]`, nil, "max_rounds"},
		"two-field condition":   {`[{"type": "branch", "configuration": {"cases": [{"when": {"succeeded": true, "content_matches": "x"}, "then": []}]}}]`, nil, "exactly one"},
		"unknown strategy":      {`[{"type": "aggregate", "configuration": {"strategy": "median"}}]`, nil, `unknown aggregate strategy "median"`},
		"bad template":          {`[{"type": "transform", "configuration": {"transformer": "prompt", "options": {"template": "{{"}}}]`, nil, "template"},
		"subgraph cycle":        {`[{"type": "subgraph", "configuration": {"name": "a"}}]`, map[string]string{"a": `[{"type": "subgraph", "configuration": {"name": "b"}}]`, "b": `[{"type": "subgraph", "configuration": {"name": "a"}}]`}, "includes itself (a -> b -> a)"},
		"unknown subgraph":      {`[{"type": "subgraph", "configuration": {"name": "nope"}}]`, nil, `unknown subgraph "nope"`},
		"empty graph":           {`[]`, nil, "at least one step"},
		"negative hop limit":    {`[{"type": "respond"}]`, nil, ""},
		"unknown option":        {`[{"type": "aggregate", "configuration": {"strategy": "concat", "options": {"sep": "x"}}}]`, nil, "unknown field"},
		"bad on_error":          {`[{"type": "parallel", "configuration": {"on_error": "retry", "branches": [[{"type": "call", "configuration": {"model": "a"}}]]}}]`, nil, "on_error"},
		"bad system mode":       {`[{"type": "transform", "configuration": {"transformer": "system_prompt", "options": {"mode": "merge"}}}]`, nil, "mode"},
		"signal without a name": {`[{"type": "branch", "configuration": {"cases": [{"when": {"signal": {"type": "domain"}}, "then": []}]}}]`, nil, "type and a name"},
	} {
		t.Run(name, func(t *testing.T) {
			subgraphs := map[string][]StepSpec{}
			for sub, text := range tc.subgraphs {
				subgraphs[sub] = specs(t, text)
			}
			limits := Limits{}
			if name == "negative hop limit" {
				limits.MaxHops, tc.want = -1, "max_hops must not be negative"
			}
			_, err := NewBuilder(subgraphs).Program("bad", specs(t, tc.steps), limits)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err %v, want %q", err, tc.want)
			}
		})
	}
}

type echoConfig struct {
	Text string `json:"text"`
}

var registerEcho = sync.OnceValue(func() error {
	return RegisterNode("test_echo", NodeType{
		NewPayload: func() any { return &echoConfig{} },
		Strict:     true,
		Defaults: func(payload any) {
			if cfg := payload.(*echoConfig); cfg.Text == "" {
				cfg.Text = "default"
			}
		},
		Build: func(_ *Builder, _ At, payload any) (Node, error) {
			text := payload.(*echoConfig).Text
			return nodeFunc(func(_ context.Context, _ *Exec, st *State) error {
				body, _ := composeCompletion("echo", []string{text}, Usage{})
				st.Results = []*Result{composedResult("echo", body, Usage{})}
				return nil
			}), nil
		},
	})
})

func TestOtherPackagesRegisterStepTypes(t *testing.T) {
	if err := registerEcho(); err != nil {
		t.Fatal(err)
	}
	if err := RegisterNode("test_echo", NodeType{NewPayload: func() any { return nil }, Build: nil}); err == nil {
		t.Fatal("a second registration of one type must fail")
	}
	program, err := NewBuilder(nil).Program("echo", specs(t, `[{"type": "test_echo"}, {"type": "respond"}]`), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	outcome, err := Run(context.Background(), program, Input{Request: chatRequest(t, "x")}, Options{Caller: &fakeCaller{}})
	if err != nil || answerText(t, outcome) != "default" {
		t.Fatalf("err %v", err)
	}
}

type nodeFunc func(ctx context.Context, x *Exec, st *State) error

func (f nodeFunc) Run(ctx context.Context, x *Exec, st *State) error { return f(ctx, x, st) }

func TestStrategies(t *testing.T) {
	results := func(contents ...string) []*Result {
		out := make([]*Result, len(contents))
		for i, content := range contents {
			resp := completion("m", content, 1)
			out[i] = &Result{Model: "m", Status: resp.Status, Body: resp.Body, Usage: Usage{TotalTokens: 1}}
		}
		return out
	}
	st := &State{Results: results(" yes", "no", "yes ", "no")}
	if err := (Vote{}).Aggregate(context.Background(), nil, st); err != nil || st.Results[0].Content() != " yes" {
		t.Fatalf("vote: %v %q", err, st.Results[0].Content())
	}
	st = &State{Results: append([]*Result{{Model: "m", Status: 500}}, results("first", "second")...)}
	if err := (First{}).Aggregate(context.Background(), nil, st); err != nil || st.Results[0].Content() != "first" {
		t.Fatalf("first: %v", err)
	}
	st = &State{Results: results("a", "b")}
	if err := (Concat{}).Aggregate(context.Background(), nil, st); err != nil ||
		st.Results[0].Content() != "a\n\nb" || st.Results[0].Usage.TotalTokens != 2 {
		t.Fatalf("concat: %v %+v", err, st.Results[0])
	}
	if err := (First{}).Aggregate(context.Background(), nil, &State{}); err == nil {
		t.Fatal("no successes must fail")
	}

	request := chatRequest(t, "q")
	st = &State{Request: request}
	if err := (SystemPrompt{Content: "rules"}).Transform(context.Background(), nil, st); err != nil {
		t.Fatal(err)
	}
	if err := (SystemPrompt{Content: "more", Mode: SystemPrepend}).Transform(context.Background(), nil, st); err != nil {
		t.Fatal(err)
	}
	st.Results = results("draft")
	if err := (AppendResults{}).Transform(context.Background(), nil, st); err != nil {
		t.Fatal(err)
	}
	messages, _ := st.Request.Messages()
	roles := []string{}
	for _, m := range messages {
		roles = append(roles, MessageRole(m)+":"+MessageText(m))
	}
	if strings.Join(roles, "|") != "system:more\n\nrules|user:q|assistant:draft" {
		t.Fatalf("messages %v", roles)
	}
}
