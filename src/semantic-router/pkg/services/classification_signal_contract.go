package services

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
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

	if s.classifier == nil {
		return &EvalResponse{
			OriginalText: input.evaluationText,
			Metrics:      &classification.SignalMetricsCollection{},
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
	)

	var decisionResult *decision.DecisionResult
	var traces []decision.DecisionTrace
	if s.config != nil && len(s.config.Decisions) > 0 {
		decisionResult, traces = s.evaluateIntentDecision(signals, wantTrace)
	}

	resp := s.buildEvalResponse(input.evaluationText, signals, decisionResult)
	resp.EvalTrace = traces
	return resp, nil
}

func (s *ClassificationService) evaluateIntentDecision(
	signals *classification.SignalResults,
	wantTrace bool,
) (*decision.DecisionResult, []decision.DecisionTrace) {
	if !wantTrace {
		decisionResult, err := s.classifier.EvaluateDecisionWithEngine(signals)
		if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
			logging.Warnf("Decision evaluation failed: %v", err)
		}
		return decisionResult, nil
	}

	decisionResult, traces, err := s.classifier.EvaluateDecisionWithEngineAndTrace(signals)
	if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
		logging.Warnf("Decision evaluation failed: %v", err)
	}
	return decisionResult, traces
}
