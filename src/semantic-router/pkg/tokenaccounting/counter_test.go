package tokenaccounting

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const testModel = "Qwen/Qwen3-8B"

func text(value string) llmprotocol.Content {
	return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: value}
}

// agentRequest has one of every prompt-bearing component Count walks.
func agentRequest() *llmprotocol.Request {
	set := func(value string) llmprotocol.OptionalString {
		return llmprotocol.OptionalString{Value: value, Present: true}
	}
	return &llmprotocol.Request{
		Model: testModel,
		Instructions: []llmprotocol.InstructionBlock{{
			Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{text("You are terse.")},
		}},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				text("Find the open reservation."),
				{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/a.png"},
				{Kind: llmprotocol.ContentAudio, Data: "abcd"},
			}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentReasoning, Text: "check the ledger first", ReasoningDetail: &llmprotocol.ReasoningDetail{
					Text: set("step one"), Summary: set("sum"), Data: set("xxxxxxxx"), Signature: set("sig"),
				}},
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{Name: "lookup", Arguments: `{"id": 7}`}},
			}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
				Kind:       llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{Content: []llmprotocol.Content{text("found reservation r1")}},
			}}},
		},
		Tools: []llmprotocol.Tool{
			{Name: "lookup", Description: "Find a reservation", InputSchema: json.RawMessage(`{"type": "object"}`)},
			{Kind: llmprotocol.ToolKindCustom, Name: "grep", CustomFormat: &llmprotocol.CustomToolFormat{Definition: "start: WORD+"}},
		},
		OutputFormat: llmprotocol.OutputFormat{
			Kind: llmprotocol.OutputJSONSchema, Name: "answer", Description: "one line", Schema: json.RawMessage(`{"type": "string"}`),
		},
	}
}

func wordRegistry(t *testing.T) *Registry {
	t.Helper()
	registry, err := NewRegistry(Family{Name: "qwen3", Models: []string{testModel}, Tokenizer: wordTokenizer{}})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func TestCountWithTokenizerCountsEveryPromptComponentExactly(t *testing.T) {
	got := NewCounter(wordRegistry(t)).Count(testModel, agentRequest(), nil)

	// Words, in walk order: instruction 3, user text 4, reasoning text 4,
	// detail text 2 and summary 1, tool call name 1 and arguments 2, tool
	// result 3, tool definitions 1+3+2 and 1+2, output format 1+2+2.
	words := 3 + 4 + 4 + 2 + 1 + 1 + 2 + 3 + 1 + 3 + 2 + 1 + 2 + 1 + 2 + 2
	reserve := 4*llmprotocol.InputMessageFramingTokens + // 1 instruction, 3 messages
		llmprotocol.InputImageTokens +
		len("abcd") + // audio payload
		len("xxxxxxxx") + len("sig") + // opaque reasoning state
		llmprotocol.InputToolCallFramingTokens +
		2*llmprotocol.InputToolDefinitionFramingTokens
	want := Count{Tokens: words + reserve, ReserveTokens: reserve, Source: SourceExact, Family: "qwen3"}
	if got != want {
		t.Fatalf("Count = %+v, want %+v", got, want)
	}
}

func TestCountWithTokenizerSeesEachComponent(t *testing.T) {
	const marker = "alpha beta gamma"
	cases := map[string]func(*llmprotocol.Request, string){
		"instruction": func(r *llmprotocol.Request, v string) { r.Instructions[0].Content[0].Text = v },
		"user text":   func(r *llmprotocol.Request, v string) { r.Messages[0].Content[0].Text = v },
		"reasoning":   func(r *llmprotocol.Request, v string) { r.Messages[1].Content[0].Text = v },
		"detail text": func(r *llmprotocol.Request, v string) { r.Messages[1].Content[0].ReasoningDetail.Text.Value = v },
		"detail summary": func(r *llmprotocol.Request, v string) {
			r.Messages[1].Content[0].ReasoningDetail.Summary.Value = v
		},
		"detail tool name": func(r *llmprotocol.Request, v string) {
			r.Messages[1].Content[0].ReasoningDetail.ToolName.Value = v
		},
		"detail call id": func(r *llmprotocol.Request, v string) {
			r.Messages[1].Content[0].ReasoningDetail.ToolCallID.Value = v
		},
		"detail arguments": func(r *llmprotocol.Request, v string) {
			r.Messages[1].Content[0].ReasoningDetail.Arguments.Value = v
		},
		"detail result": func(r *llmprotocol.Request, v string) {
			r.Messages[1].Content[0].ReasoningDetail.Result.Value = v
		},
		"tool call name":      func(r *llmprotocol.Request, v string) { r.Messages[1].Content[1].ToolCall.Name = v },
		"tool call arguments": func(r *llmprotocol.Request, v string) { r.Messages[1].Content[1].ToolCall.Arguments = v },
		"tool result": func(r *llmprotocol.Request, v string) {
			r.Messages[2].Content[0].ToolResult.Content[0].Text = v
		},
		"tool name":         func(r *llmprotocol.Request, v string) { r.Tools[0].Name = v },
		"tool description":  func(r *llmprotocol.Request, v string) { r.Tools[0].Description = v },
		"tool schema":       func(r *llmprotocol.Request, v string) { r.Tools[0].InputSchema = json.RawMessage(v) },
		"custom grammar":    func(r *llmprotocol.Request, v string) { r.Tools[1].CustomFormat.Definition = v },
		"output name":       func(r *llmprotocol.Request, v string) { r.OutputFormat.Name = v },
		"output descriptor": func(r *llmprotocol.Request, v string) { r.OutputFormat.Description = v },
		"output schema":     func(r *llmprotocol.Request, v string) { r.OutputFormat.Schema = json.RawMessage(v) },
	}
	counter := NewCounter(wordRegistry(t))
	for name, set := range cases {
		t.Run(name, func(t *testing.T) {
			without, with := agentRequest(), agentRequest()
			set(without, "")
			set(with, marker)
			delta := counter.Count(testModel, with, nil).Tokens - counter.Count(testModel, without, nil).Tokens
			if delta != 3 {
				t.Fatalf("adding %q to %s changed the count by %d, want 3", marker, name, delta)
			}
		})
	}
}

func TestCountWithoutTokenizerKeepsTheProviderNeutralEstimate(t *testing.T) {
	request := agentRequest()
	estimate := llmprotocol.EstimateInput(request)
	want := Count{Tokens: estimate.Tokens, ReserveTokens: estimate.ReserveTokens, Source: SourceFallback}

	for name, counter := range map[string]*Counter{
		"nil counter":   nil,
		"nil registry":  NewCounter(nil),
		"unknown model": NewCounter(wordRegistry(t)),
	} {
		model := testModel
		if name == "unknown model" {
			model = "vllm-sr/auto"
		}
		if got := counter.Count(model, request, nil); got != want {
			t.Fatalf("%s: Count = %+v, want the estimate %+v", name, got, want)
		}
	}
	if got := NewCounter(nil).Count(testModel, nil, nil); got != (Count{Source: SourceFallback}) {
		t.Fatalf("nil request Count = %+v", got)
	}
}

func TestCountReplacesOnlyTheTextRateWhenCalibrated(t *testing.T) {
	request := agentRequest()
	estimate := llmprotocol.EstimateInput(request)
	textRate := (estimate.TextBytes + llmprotocol.InputTextBytesPerToken - 1) / llmprotocol.InputTextBytesPerToken
	var gotBytes int
	calibrated := func(textBytes int) (int, bool) {
		gotBytes = textBytes
		return 1000, true
	}

	got := NewCounter(nil).Count(testModel, request, calibrated)
	want := Count{Tokens: estimate.Tokens - textRate + 1000, ReserveTokens: estimate.ReserveTokens, Source: SourceCalibrated}
	if got != want {
		t.Fatalf("Count = %+v, want %+v", got, want)
	}
	if gotBytes != estimate.TextBytes {
		t.Fatalf("calibration saw %d text bytes, want %d", gotBytes, estimate.TextBytes)
	}

	fallback := Count{Tokens: estimate.Tokens, ReserveTokens: estimate.ReserveTokens, Source: SourceFallback}
	for name, calibration := range map[string]Calibration{
		"too few samples": func(int) (int, bool) { return 1000, false },
		"no count":        func(int) (int, bool) { return 0, true },
	} {
		if uncalibrated := NewCounter(nil).Count(testModel, request, calibration); uncalibrated != fallback {
			t.Fatalf("%s: Count = %+v, want the unchanged estimate %+v", name, uncalibrated, fallback)
		}
	}

	imageOnly := &llmprotocol.Request{Messages: []llmprotocol.Message{{
		Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentImage, URL: "u"}},
	}}}
	called := false
	got = NewCounter(nil).Count(testModel, imageOnly, func(int) (int, bool) { called = true; return 1, true })
	if called || got.Source != SourceFallback {
		t.Fatalf("calibration ran for a request with no text: called=%v, %+v", called, got)
	}
}

func TestCountPrefersTheTokenizerOverCalibration(t *testing.T) {
	called := false
	got := NewCounter(wordRegistry(t)).Count(testModel, agentRequest(), func(int) (int, bool) {
		called = true
		return 1, true
	})
	if called || got.Source != SourceExact {
		t.Fatalf("calibration consulted for a model with a tokenizer: called=%v, %+v", called, got)
	}
}

func TestCountIsSafeForConcurrentUse(t *testing.T) {
	counter := NewCounter(wordRegistry(t))
	want := counter.Count(testModel, agentRequest(), nil)
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				if got := counter.Count(testModel, agentRequest(), nil); got != want {
					t.Errorf("concurrent Count = %+v, want %+v", got, want)
					return
				}
			}
		}()
	}
	wg.Wait()
}
