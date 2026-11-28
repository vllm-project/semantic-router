package services

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Verify the actual preview selection boundary with synthetic inference. This
// does not model provider tokenization or System One scan budgets.
func TestPrivateTypedPreviewPreservesContextReserve(t *testing.T) {
	for _, kind := range []string{"text", "image"} {
		t.Run(kind, func(t *testing.T) {
			var svc *ClassificationService
			var fake *typedImageRuntime
			var typed, legacy IntentRequest
			if kind == "image" {
				svc, fake = typedImageService(t)
				uri := "data:image/png;base64,YQ=="
				typed.Input = &ClassificationInput{Type: "image", ImageURL: &uri}
				legacy.Messages = []IntentMessage{{Role: "user", Content: json.RawMessage(`[{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]`)}}
			} else {
				cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
					Signals:   config.Signals{ContextRules: []config.ContextRule{{Name: "short", MinTokens: "0", MaxTokens: "10K"}}},
					Decisions: []config.Decision{{Name: "short", Rules: config.RuleNode{Type: config.SignalTypeContext, Name: "short"}, ModelRefs: []config.ModelRef{{Model: "general"}}}},
				}}
				classifier, err := classification.NewClassifier(cfg)
				if err != nil {
					t.Fatal(err)
				}
				svc = NewClassificationService(classifier, cfg)
				t.Cleanup(func() { _ = svc.Close() })
				text := "explain the screenshot"
				typed.Input = &ClassificationInput{Type: "text", Text: &text}
				legacy.Text = text
			}
			tool := json.RawMessage(`{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}`)
			typed.Tools = []json.RawMessage{tool}
			legacy.Tools = []json.RawMessage{tool}
			typed.MaxCompletionTokens = json.RawMessage(`256`)
			legacy.MaxCompletionTokens = json.RawMessage(`256`)
			selector := &evalModelSelectorStub{}
			svc.SetEvalModelSelector(selector)
			first, err := svc.ClassifyIntentForEval(context.Background(), typed)
			if err != nil || first.SelectionStatus != EvalSelectionSelected {
				t.Fatalf("typed preview: %+v %v", first, err)
			}
			got := selector.input
			second, err := svc.ClassifyIntentForEval(context.Background(), legacy)
			if err != nil || second.SelectionStatus != EvalSelectionSelected {
				t.Fatalf("legacy preview: %+v %v", second, err)
			}
			want := selector.input
			if got.ContextTokenCount != want.ContextTokenCount || got.ContextReserveTokens != want.ContextReserveTokens || got.Demand.InputTokens != want.Demand.InputTokens || got.Demand.InputReserveTokens != want.Demand.InputReserveTokens {
				t.Fatalf("typed and legacy estimates differ: typed=%+v legacy=%+v", got, want)
			}
			if got.Demand.MaxOutputTokens == nil || want.Demand.MaxOutputTokens == nil || *got.Demand.MaxOutputTokens != 256 || *want.Demand.MaxOutputTokens != 256 {
				t.Fatalf("lost output token budget: typed=%+v legacy=%+v", got.Demand, want.Demand)
			}
			if !got.Demand.ModelCapabilities.Supports(llmprotocol.CapabilityTools) {
				t.Fatalf("lost tool capability: %+v", got.Demand)
			}
			if kind == "image" {
				if got.ContextReserveTokens < classification.RequestContextImageTokenBudget || !got.Demand.ModelCapabilities.Supports(llmprotocol.CapabilityImageInput) {
					t.Fatalf("lost image reserve or capability: %+v", got)
				}
				requests := fake.snapshot()
				if len(requests) != 2 || len(requests[0].Images) != 1 || len(requests[1].Images) != 1 || requests[0].Images[0] != requests[1].Images[0] {
					t.Fatalf("images not delivered equivalently: %+v", requests)
				}
			} else if got.Demand.ModelCapabilities.Supports(llmprotocol.CapabilityImageInput) {
				t.Fatalf("invented image capability: %+v", got.Demand)
			}
		})
	}
}
