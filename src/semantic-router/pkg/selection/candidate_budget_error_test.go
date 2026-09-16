package selection

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCandidateBudgetErrorsKeepNonBudgetFailuresDistinct(t *testing.T) {
	limits := &config.CandidateRequirements{Context: config.CandidateContextKnownLimits}
	base := config.ModelParams{ContextWindowSize: 100, MaxOutputTokens: 80}
	for _, test := range []struct {
		name   string
		params config.ModelParams
		input  int
		output *int64
		code   string
	}{
		{"over context", base, 90, llmprotocol.Int64(11), "context_length_exceeded"},
		{"over output", base, 1, llmprotocol.Int64(81), "max_output_tokens_exceeded"},
		{"missing limits", config.ModelParams{}, 1, llmprotocol.Int64(1), ""},
		{"missing reserve", base, 1, nil, ""},
		{"negative estimate", base, -1, llmprotocol.Int64(1), ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateCandidateRequirements(limits, "model", test.params, CandidateDemand{Known: true, InputTokens: test.input, MaxOutputTokens: test.output})
			var budget *RequestBudgetError
			if !errors.Is(err, ErrNoEligibleCandidates) || errors.As(err, &budget) != (test.code != "") {
				t.Fatalf("wrong failure kind: %v", err)
			}
			if budget != nil && budget.Code != test.code {
				t.Fatalf("code=%s", budget.Code)
			}
		})
	}
	if err := ValidateCandidateRequirements(limits, "model", base, CandidateDemand{Known: true, InputTokens: 90, MaxOutputTokens: llmprotocol.Int64(10)}); err != nil {
		t.Fatalf("exact boundary rejected: %v", err)
	}
}

func TestEffectiveCandidateRequestDetachesEditableText(t *testing.T) {
	request := &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{{Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "system"}}}},
		Messages:     []llmprotocol.Message{{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "original"}}}}}}},
	}
	view, err := EffectiveCandidateRequest(request, nil)
	if err != nil {
		t.Fatal(err)
	}
	view.Instructions[0].Content[0].Text = "changed"
	view.Messages[0].Content[0].ToolResult.Content[0].Text = "changed"
	if request.Instructions[0].Content[0].Text != "system" || request.Messages[0].Content[0].ToolResult.Content[0].Text != "original" {
		t.Fatal("effective text view mutated ingress")
	}
}
