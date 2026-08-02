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
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()

	classifier, candidates, recipeName, err := s.evalRoutingScope(req.Model)
	if err != nil {
		return nil, err
	}

	if classifier == nil {
		return &EvalResponse{
			OriginalText:   input.evaluationText,
			RequestedModel: strings.TrimSpace(req.Model),
			Recipe:         recipeName,
			Metrics:        &classification.SignalMetricsCollection{},
		}, nil
	}

	wantTrace := req.Options != nil && req.Options.Trace
	signals := classifier.EvaluateAllSignalsWithContext(
		input.evaluationText,
		input.contextText,
		input.currentUserText,
		input.priorUserMessages,
		input.nonUserMessages,
		input.hasAssistantReply,
		true,
		"",
		nil,
		input.conversationFacts,
		input.imageURL,
	)

	var decisionResult *decision.DecisionResult
	var traces []decision.DecisionTrace
	if len(candidates) > 0 {
		decisionResult, traces = s.evaluateIntentDecision(classifier, signals, candidates, wantTrace)
	}

	resp := s.buildEvalResponse(input.evaluationText, signals, decisionResult)
	resp.RequestedModel = strings.TrimSpace(req.Model)
	resp.Recipe = recipeName
	resp.EvalTrace = traces
	return resp, nil
}

func (s *ClassificationService) evaluateIntentDecision(
	classifier *classification.Classifier,
	signals *classification.SignalResults,
	candidates []config.Decision,
	wantTrace bool,
) (*decision.DecisionResult, []decision.DecisionTrace) {
	if !wantTrace {
		decisionResult, err := classifier.EvaluateDecisionWithEngineForDecisions(signals, candidates)
		if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
			logging.Warnf("Decision evaluation failed: %v", err)
		}
		return decisionResult, nil
	}

	decisionResult, traces, err := classifier.EvaluateDecisionWithEngineAndTraceForDecisions(
		signals,
		candidates,
	)
	if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
		logging.Warnf("Decision evaluation failed: %v", err)
	}
	return decisionResult, traces
}

func (s *ClassificationService) evalRoutingScope(modelName string) (*classification.Classifier, []config.Decision, config.RecipeName, error) {
	if s.config == nil {
		return s.classifier, nil, "", nil
	}
	trimmed := strings.TrimSpace(modelName)
	if trimmed == "" {
		trimmed = config.DefaultVSRAutoModelName
	}
	recipe, ok := s.config.RecipeForRoutingModel(trimmed)
	if !ok {
		return nil, nil, "", fmt.Errorf("%w %q", ErrUnknownRoutingModel, trimmed)
	}
	classifier := s.classifier
	if s.recipeClassifiers != nil {
		var found bool
		classifier, found = s.recipeClassifiers.ForRecipe(recipe.Name)
		if !found {
			return nil, nil, "", fmt.Errorf("classifier for routing recipe %q is unavailable", recipe.Name)
		}
	}
	return classifier, recipe.Profile.Decisions, recipe.Name, nil
}
