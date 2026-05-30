package services

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// buildIntentResponseFromSignals builds an IntentResponse from signals and decision result.
func (s *ClassificationService) buildIntentResponseFromSignals(
	signals *classification.SignalResults,
	decisionResult *decision.DecisionResult,
	category string,
	confidence float64,
	processingTime int64,
	req IntentRequest,
) *IntentResponse {
	response := &IntentResponse{
		Classification: Classification{
			Category:         category,
			Confidence:       confidence,
			ProcessingTimeMs: processingTime,
		},
	}

	populateIntentProbabilities(response, category, confidence, req.Options)
	response.RecommendedModel = s.resolveRecommendedModel(decisionResult, category, confidence)
	response.RoutingDecision = s.resolveRoutingDecision(decisionResult, confidence, req.Options)
	if signals != nil {
		response.MatchedSignals = buildMatchedSignals(signals)
	}
	if decisionPayload := buildDecisionResultPayload(decisionResult); decisionPayload != nil {
		response.DecisionResult = decisionPayload
	}

	return response
}

// buildEvalResponse builds an EvalResponse from signal results and decision result.
func (s *ClassificationService) buildEvalResponse(
	text string,
	signals *classification.SignalResults,
	decisionResult *decision.DecisionResult,
) *EvalResponse {
	response := &EvalResponse{
		OriginalText:      text,
		Metrics:           signals.Metrics,
		SignalConfidences: signals.SignalConfidences,
		SignalValues:      signals.SignalValues,
	}

	matchedSignals := buildMatchedSignals(signals)
	unmatchedSignals := s.getUnmatchedSignals(signals)

	if decisionResult != nil && decisionResult.Decision != nil {
		usedSignals := s.extractUsedSignalsFromDecision(decisionResult.Decision)

		response.DecisionResult = &EvalDecisionResult{
			DecisionName:     decisionResult.Decision.Name,
			UsedSignals:      usedSignals,
			MatchedSignals:   matchedSignals,
			UnmatchedSignals: unmatchedSignals,
		}

		if len(decisionResult.Decision.ModelRefs) > 0 {
			models := make([]string, 0, len(decisionResult.Decision.ModelRefs))
			for _, modelRef := range decisionResult.Decision.ModelRefs {
				models = append(models, modelRef.Model)
			}
			response.RecommendedModels = models
			response.RoutingDecision = decisionResult.Decision.Name
		}
	} else {
		response.DecisionResult = &EvalDecisionResult{
			DecisionName:     "",
			UsedSignals:      &MatchedSignals{},
			MatchedSignals:   matchedSignals,
			UnmatchedSignals: unmatchedSignals,
		}
	}

	return response
}

func populateIntentProbabilities(
	response *IntentResponse,
	category string,
	confidence float64,
	options *IntentOptions,
) {
	if options == nil || !options.ReturnProbabilities {
		return
	}
	response.Probabilities = map[string]float64{category: confidence}
}

func (s *ClassificationService) resolveRecommendedModel(
	decisionResult *decision.DecisionResult,
	category string,
	confidence float64,
) string {
	if decisionResult != nil && decisionResult.Decision != nil && len(decisionResult.Decision.ModelRefs) > 0 {
		modelRef := decisionResult.Decision.ModelRefs[0]
		if modelRef.LoRAName != "" {
			return modelRef.LoRAName
		}
		return modelRef.Model
	}
	return s.getRecommendedModel(category, confidence)
}

func (s *ClassificationService) resolveRoutingDecision(
	decisionResult *decision.DecisionResult,
	confidence float64,
	options *IntentOptions,
) string {
	if decisionResult != nil && decisionResult.Decision != nil {
		return decisionResult.Decision.Name
	}
	return s.getRoutingDecision(confidence, options)
}

func buildDecisionResultPayload(decisionResult *decision.DecisionResult) *DecisionResult {
	if decisionResult == nil || decisionResult.Decision == nil {
		return nil
	}
	return &DecisionResult{
		DecisionName: decisionResult.Decision.Name,
		Confidence:   decisionResult.Confidence,
		MatchedRules: decisionResult.MatchedRules,
	}
}
