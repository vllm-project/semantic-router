package services

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

type evalModelSelectorStub struct {
	input EvalModelSelectionInput
}

func (s *evalModelSelectorStub) SelectModelForEval(
	input EvalModelSelectionInput,
) EvalModelSelection {
	s.input = input
	return EvalModelSelection{
		SelectedModel: "model-b",
		Status:        EvalSelectionSelected,
		Method:        "multi_factor",
		Reason:        "highest live score",
	}
}

func TestPopulateEvalModelSelectionReturnsConcreteRuntimeChoice(t *testing.T) {
	selector := &evalModelSelectorStub{}
	service := &ClassificationService{}
	service.SetEvalModelSelector(selector)
	response := &EvalResponse{Recipe: "balanced"}
	matchedDecision := &config.Decision{
		Name:      "balanced-route",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
	}
	service.populateEvalModelSelection(
		response,
		intentSignalInput{
			currentUserText: "Explain the tradeoff.",
			requestFacts: classification.RequestFacts{
				Context:           t.Context(),
				ContextTokenFloor: 4096,
			},
		},
		&decision.DecisionResult{
			Decision:     matchedDecision,
			MatchedRules: []string{"domain:engineering"},
		},
		nil,
	)

	if response.SelectedModel != "model-b" || response.SelectionStatus != EvalSelectionSelected {
		t.Fatalf("selection response = %+v", response)
	}
	if selector.input.Decision != matchedDecision || selector.input.Recipe != "balanced" {
		t.Fatalf("selector scope = %+v", selector.input)
	}
	if selector.input.Query != "Explain the tradeoff." || selector.input.Category != "engineering" {
		t.Fatalf("selector semantic input = %+v", selector.input)
	}
	if selector.input.ContextTokenCount != 4096 {
		t.Fatalf("selector context count = %d", selector.input.ContextTokenCount)
	}
	if selector.input.Context != t.Context() {
		t.Fatal("preview lost the caller's cancellation and deadline context")
	}
}

func TestPopulateEvalModelSelectionDoesNotInventFirstRecommendedModel(t *testing.T) {
	response := &EvalResponse{Recipe: "accuracy"}
	service := &ClassificationService{}
	service.populateEvalModelSelection(
		response,
		intentSignalInput{},
		&decision.DecisionResult{Decision: &config.Decision{
			Name:      "fusion-route",
			ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		}},
		nil,
	)

	if response.SelectedModel != "" || response.SelectionStatus != EvalSelectionUnavailable {
		t.Fatalf("unwired Eval invented a final model: %+v", response)
	}
}

func TestPreviewContextValidatesAndPassesProtectionFacts(t *testing.T) {
	for _, identity := range []string{strings.Repeat("a", 1025), "invalid\nidentity"} {
		_, err := (IntentRequest{Text: "test", PreviewContext: &PreviewContext{SessionID: identity}}).resolveSignalInput()
		if !errors.Is(err, ErrInvalidRequestFacts) {
			t.Fatalf("invalid preview identity error=%v", err)
		}
	}
	seed := int64(17)
	context := &PreviewContext{SessionID: "session", ConversationID: "conversation", SamplingSeed: &seed}
	request := IntentRequest{Messages: []IntentMessage{{Role: "assistant", Content: json.RawMessage(`""`), ToolCalls: []json.RawMessage{json.RawMessage(`{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}`)}}, {Role: "tool", Content: json.RawMessage(`"result"`), ToolCallID: "call_1"}}, PreviewContext: context}
	input, err := request.resolveSignalInput()
	if err != nil {
		t.Fatal(err)
	}
	selector := &evalModelSelectorStub{}
	service := &ClassificationService{}
	service.SetEvalModelSelector(selector)
	service.populateEvalModelSelection(&EvalResponse{}, input, &decision.DecisionResult{Decision: &config.Decision{Name: "tool-route"}}, context)
	if selector.input.PreviewContext != context || !selector.input.ConversationFacts.LastMessageToolResult || selector.input.SemanticRequest == nil {
		t.Fatalf("protection context lost: %+v", selector.input)
	}
}

type cancelingEvalModelSelector struct {
	cancel     context.CancelFunc
	sawContext context.Context
}

func (s *cancelingEvalModelSelector) SelectModelForEval(input EvalModelSelectionInput) EvalModelSelection {
	s.sawContext = input.Context
	s.cancel()
	return EvalModelSelection{SelectedModel: "late-model", Status: EvalSelectionSelected}
}

func TestEvalCancellationAfterSelectionCannotReturnLateSuccess(t *testing.T) {
	cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Signals:   config.Signals{ContextRules: []config.ContextRule{{Name: "short", MinTokens: "0", MaxTokens: "10K"}}},
		Decisions: []config.Decision{{Name: "short", Rules: config.RuleNode{Type: config.SignalTypeContext, Name: "short"}, ModelRefs: []config.ModelRef{{Model: "late-model"}}}},
	}}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	service := NewClassificationService(classifier, cfg)
	t.Cleanup(func() {
		if closeErr := service.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	selector := &cancelingEvalModelSelector{cancel: cancel}
	service.SetEvalModelSelector(selector)
	response, err := service.ClassifyIntentForEval(ctx, IntentRequest{Text: "hello"})
	if !errors.Is(err, context.Canceled) || response != nil {
		t.Fatalf("canceled selector produced a partial success: response=%+v err=%v", response, err)
	}
	if selector.sawContext != ctx {
		t.Fatal("selector lost HTTP request context")
	}
}
