package services

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ClassifyIntentForEval performs intent classification specifically for evaluation scenarios.
// This method forces evaluation of all signals and returns comprehensive signal information.
func (s *ClassificationService) ClassifyIntentForEval(req IntentRequest) (*EvalResponse, error) {
	input, err := req.resolveSignalInput()
	if err != nil {
		return nil, err
	}
	candidates, recipeName, err := s.evalDecisionCandidates(req.Model)
	if err != nil {
		return nil, err
	}

	if s.classifier == nil {
		return &EvalResponse{
			OriginalText:   input.evaluationText,
			RequestedModel: strings.TrimSpace(req.Model),
			Recipe:         recipeName,
			Metrics:        &classification.SignalMetricsCollection{},
		}, nil
	}

	wantTrace := req.Options != nil && req.Options.Trace
	signals := s.classifier.EvaluateAllSignalsWithContext(
		input.evaluationText,
		input.contextText,
		input.currentUserText,
		input.priorUserMessages,
		input.nonUserMessages,
		input.hasAssistantReply,
		true,
		"",
		nil,
		classification.ConversationFacts{},
		input.imageURL,
	)

	var decisionResult *decision.DecisionResult
	var traces []decision.DecisionTrace
	if len(candidates) > 0 {
		decisionResult, traces = s.evaluateIntentDecision(signals, candidates, wantTrace)
	}

	resp := s.buildEvalResponse(input.evaluationText, signals, decisionResult)
	resp.RequestedModel = strings.TrimSpace(req.Model)
	resp.Recipe = recipeName
	resp.EvalTrace = traces
	return resp, nil
}

func (s *ClassificationService) evaluateIntentDecision(
	signals *classification.SignalResults,
	candidates []config.Decision,
	wantTrace bool,
) (*decision.DecisionResult, []decision.DecisionTrace) {
	if !wantTrace {
		decisionResult, err := s.classifier.EvaluateDecisionWithEngineForDecisions(signals, candidates)
		if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
			logging.Warnf("Decision evaluation failed: %v", err)
		}
		return decisionResult, nil
	}

	decisionResult, traces, err := s.classifier.EvaluateDecisionWithEngineAndTraceForDecisions(
		signals,
		candidates,
	)
	if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
		logging.Warnf("Decision evaluation failed: %v", err)
	}
	return decisionResult, traces
}

func (s *ClassificationService) evalDecisionCandidates(modelName string) ([]config.Decision, string, error) {
	if s.config == nil {
		return nil, "", nil
	}
	trimmed := strings.TrimSpace(modelName)
	if trimmed == "" || s.config.IsAutoModelName(trimmed) {
		return s.config.Decisions, config.DefaultRecipeName, nil
	}
	if recipe, ok := s.config.RecipeForRequestModel(trimmed); ok {
		return recipe.Decisions, recipe.Name, nil
	}
	return nil, "", fmt.Errorf("%w %q", ErrUnknownRoutingModel, trimmed)
}
