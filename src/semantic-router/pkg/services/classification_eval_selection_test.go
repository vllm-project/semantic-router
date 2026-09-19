package services

import (
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
