package services

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// ClassifyIntentForEval performs intent classification specifically for evaluation scenarios.
// This method forces evaluation of all signals and returns comprehensive signal information.
func (s *ClassificationService) ClassifyIntentForEval(req IntentRequest) (*EvalResponse, error) {
	input, err := req.resolveSignalInput()
	if err != nil {
		return nil, err
	}
	classifier, candidates, recipeName, err := s.evalRoutingScopeSnapshot(req.Model)
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
	signals := classifier.EvaluateAllSignalsWithRequestFacts(
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
		input.requestFacts,
	)

	var decisionResult *decision.DecisionResult
	var traces []decision.DecisionTrace
	var decisionErr error
	if len(candidates) > 0 {
		decisionResult, traces, decisionErr = evaluateIntentDecision(
			classifier,
			signals,
			candidates,
			wantTrace,
		)
	}

	resp := s.buildEvalResponse(
		input.evaluationText,
		signals,
		decisionResult,
		classifier,
	)
	resp.RequestedModel = strings.TrimSpace(req.Model)
	resp.Recipe = recipeName
	resp.EvalTrace = traces
	if decisionErr != nil {
		resp.DecisionError = decisionErr.Error()
		return resp, decisionErr
	}
	s.populateEvalModelSelection(resp, input, decisionResult)
	return resp, nil
}

func (s *ClassificationService) populateEvalModelSelection(
	response *EvalResponse,
	input intentSignalInput,
	decisionResult *decision.DecisionResult,
) {
	if response == nil || decisionResult == nil || decisionResult.Decision == nil {
		return
	}
	selector := s.evalModelSelectorSnapshot()
	if selector == nil {
		response.SelectionStatus = EvalSelectionUnavailable
		response.SelectionReason = "live model selector is unavailable"
		return
	}
	selection := selector.SelectModelForEval(EvalModelSelectionInput{
		Recipe:            response.Recipe,
		Decision:          decisionResult.Decision,
		Query:             input.currentUserText,
		Category:          evalDecisionCategory(decisionResult.MatchedRules),
		ContextTokenCount: input.requestFacts.ContextTokenFloor,
	})
	response.SelectedModel = selection.SelectedModel
	response.SelectionStatus = selection.Status
	response.SelectionMethod = selection.Method
	response.SelectionReason = selection.Reason
}

func evalDecisionCategory(matchedRules []string) string {
	for _, rule := range matchedRules {
		if strings.HasPrefix(rule, "domain:") {
			return strings.TrimPrefix(rule, "domain:")
		}
	}
	return ""
}

func evaluateIntentDecision(
	classifier *classification.Classifier,
	signals *classification.SignalResults,
	candidates []config.Decision,
	wantTrace bool,
) (*decision.DecisionResult, []decision.DecisionTrace, error) {
	if !wantTrace {
		decisionResult, err := classifier.EvaluateDecisionWithEngineForDecisions(signals, candidates)
		if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
			return nil, nil, err
		}
		return decisionResult, nil, nil
	}

	decisionResult, traces, err := classifier.EvaluateDecisionWithEngineAndTraceForDecisions(
		signals,
		candidates,
	)
	if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
		return nil, traces, err
	}
	return decisionResult, traces, nil
}

func (s *ClassificationService) evalRoutingScopeSnapshot(
	modelName string,
) (
	*classification.Classifier,
	[]config.Decision,
	config.RecipeName,
	error,
) {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.evalRoutingScope(modelName)
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
