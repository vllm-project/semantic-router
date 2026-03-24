package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

type metaRoutingPassExecution struct {
	passCtx           *RequestContext
	signalInput       signalEvaluationInput
	signals           *classification.SignalResults
	result            *decision.DecisionResult
	decisionName      string
	confidence        float64
	reasoningDecision entropy.ReasoningDecision
	selectedModel     string
	fallbackModel     string
	passTrace         PassTrace
	assessment        *MetaAssessment
}

type metaRoutingBasePassRunner struct {
	router *OpenAIRouter
}

type metaRoutingExecutor struct {
	router *OpenAIRouter
}

type metaRoutingOrchestrator struct {
	router          *OpenAIRouter
	policy          config.MetaRoutingConfig
	originalModel   string
	userContent     string
	nonUserMessages []string
	signalInput     signalEvaluationInput
	baseCtx         *RequestContext
	provider        metaRoutingPolicyProvider
	baseRunner      metaRoutingBasePassRunner
	executor        metaRoutingExecutor
}

func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, userContent string, nonUserMessages []string, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string, error) {
	if len(nonUserMessages) == 0 && userContent == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	if len(r.Config.Decisions) == 0 {
		if r.Config.IsAutoModelName(originalModel) {
			logging.Warnf("No decisions configured, using default model")
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signalInput := r.prepareSignalEvaluationInput(userContent, nonUserMessages)
	if signalInput.evaluationText == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	orchestrator := metaRoutingOrchestrator{
		router:          r,
		policy:          r.Config.MetaRouting,
		originalModel:   originalModel,
		userContent:     userContent,
		nonUserMessages: nonUserMessages,
		signalInput:     signalInput,
		baseCtx:         ctx,
		provider:        r.metaRoutingPolicyProvider(),
		baseRunner:      metaRoutingBasePassRunner{router: r},
		executor:        metaRoutingExecutor{router: r},
	}

	finalPass, err := orchestrator.execute()
	if err != nil {
		return "", 0, entropy.ReasoningDecision{}, "", err
	}
	if finalPass == nil {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	commitMetaRoutingPassContext(ctx, finalPass.passCtx)
	if finalPass.signals != nil {
		r.processUserFeedbackForElo(finalPass.signals.MatchedUserFeedbackRules, originalModel, ctx)
	}

	if finalPass.result == nil {
		return "", 0.0, entropy.ReasoningDecision{}, finalPass.selectedModel, nil
	}
	return finalPass.decisionName, finalPass.confidence, finalPass.reasoningDecision, finalPass.selectedModel, nil
}

func (o metaRoutingOrchestrator) execute() (*metaRoutingPassExecution, error) {
	basePass, err := o.baseRunner.run(o, nil, nil, metaRoutingPassKindBase, 0)
	if err != nil || basePass == nil {
		return basePass, err
	}
	if !o.policy.Enabled() {
		return basePass, nil
	}

	trace := ensureMetaRoutingTrace(o.baseCtx, o.policy, o.provider.Descriptor())
	appendMetaRoutingPass(trace, basePass.passTrace)
	logMetaRoutingObservation(basePass.passTrace)

	plan := o.provider.Plan(o.policy, o.signalInput, basePass.signals, basePass.assessment)
	if !o.policy.Enabled() || o.policy.Mode == config.MetaRoutingModeObserve || plan == nil || o.policy.MaxPasses <= 1 {
		finalizeMetaRoutingTrace(trace, basePass.passTrace, basePass.assessment, plan, plannedSignalFamilies(plan), false)
		return basePass, nil
	}

	refinedPass, err := o.executor.run(o, basePass, plan, 1)
	if err != nil {
		return nil, err
	}
	if refinedPass == nil {
		finalizeMetaRoutingTrace(trace, basePass.passTrace, basePass.assessment, plan, plannedSignalFamilies(plan), false)
		return basePass, nil
	}

	appendMetaRoutingPass(trace, refinedPass.passTrace)
	logMetaRoutingObservation(refinedPass.passTrace)

	chosenPass := chooseMetaRoutingFinalPass(o.policy.Mode, basePass, refinedPass)
	overturned := metaRoutingDecisionOverturned(basePass, chosenPass)
	finalizeMetaRoutingTrace(trace, chosenPass.passTrace, chosenPass.assessment, plan, plannedSignalFamilies(plan), overturned)
	return chosenPass, nil
}

func (r metaRoutingBasePassRunner) run(
	orchestrator metaRoutingOrchestrator,
	baseSignals *classification.SignalResults,
	plan *RefinementPlan,
	kind string,
	index int,
) (*metaRoutingPassExecution, error) {
	passCtx := cloneRequestContextForMetaRouting(orchestrator.baseCtx)
	input := orchestrator.signalInput

	families := metaRoutingPlannedFamilies(plan)
	if plan != nil && metaRoutingPlanDisablesCompression(plan) {
		input.compressedText = input.evaluationText
		input.skipCompressionSignals = nil
	}

	passStart := time.Now()
	signals, err := r.router.evaluateSignalsForRoutingPass(
		input,
		orchestrator.nonUserMessages,
		passCtx,
		families,
		baseSignals,
	)
	if err != nil {
		return nil, err
	}

	result, fallbackModel := r.router.runDecisionEngine(orchestrator.originalModel, passCtx, signals)
	decisionName := ""
	confidence := 0.0
	reasoningDecision := entropy.ReasoningDecision{}
	selectedModel := fallbackModel
	if result != nil {
		decisionName, confidence, reasoningDecision, selectedModel = r.router.finalizeDecisionEvaluation(
			result,
			orchestrator.originalModel,
			orchestrator.userContent,
			passCtx,
		)
	}

	passTrace := buildMetaRoutingPassTrace(
		index,
		kind,
		input,
		signals,
		result,
		passCtx,
		selectedModel,
		time.Since(passStart),
	)
	assessment := orchestrator.provider.Assess(orchestrator.policy, input, signals, passTrace)
	if assessment != nil {
		passTrace.Assessment = assessment
		passTrace.TraceQuality.Fragile = assessment.NeedsRefine
	}

	return &metaRoutingPassExecution{
		passCtx:           passCtx,
		signalInput:       input,
		signals:           signals,
		result:            result,
		decisionName:      decisionName,
		confidence:        confidence,
		reasoningDecision: reasoningDecision,
		selectedModel:     selectedModel,
		fallbackModel:     fallbackModel,
		passTrace:         passTrace,
		assessment:        assessment,
	}, nil
}

func (e metaRoutingExecutor) run(
	orchestrator metaRoutingOrchestrator,
	basePass *metaRoutingPassExecution,
	plan *RefinementPlan,
	index int,
) (*metaRoutingPassExecution, error) {
	if plan == nil || len(plan.Actions) == 0 {
		return nil, nil
	}
	return metaRoutingBasePassRunner(e).run(
		orchestrator,
		basePass.signals,
		plan,
		metaRoutingPassKindRefinement,
		index,
	)
}

func (r *OpenAIRouter) evaluateSignalsForRoutingPass(
	signalInput signalEvaluationInput,
	nonUserMessages []string,
	ctx *RequestContext,
	signalFamilies []string,
	baseSignals *classification.SignalResults,
) (*classification.SignalResults, error) {
	signalStart := time.Now()
	signalCtx, signalSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanSignalEvaluation)

	var (
		signals  *classification.SignalResults
		authzErr error
	)
	if len(signalFamilies) == 0 || baseSignals == nil {
		signals, authzErr = r.Classifier.EvaluateAllSignalsWithHeaders(
			signalInput.compressedText,
			signalInput.allMessagesText,
			nonUserMessages,
			ctx.Headers,
			false,
			ctx.RequestImageURL,
			signalInput.evaluationText,
			signalInput.skipCompressionSignals,
		)
	} else {
		refreshed, err := r.Classifier.EvaluateSignalFamiliesWithHeaders(classification.SignalFamilyEvaluationRequest{
			Text:                 signalInput.compressedText,
			ContextText:          signalInput.allMessagesText,
			NonUserMessages:      nonUserMessages,
			Headers:              ctx.Headers,
			ImageURL:             ctx.RequestImageURL,
			UncompressedText:     signalInput.evaluationText,
			SkipCompressionRules: signalInput.skipCompressionSignals,
			SignalFamilies:       signalFamilies,
		})
		if err != nil {
			authzErr = err
		} else {
			signals = r.Classifier.RefreshSignalFamilies(baseSignals, refreshed, signalFamilies)
		}
	}
	if authzErr != nil {
		signalSpan.End()
		logging.Errorf("[Signal Evaluation] Authz evaluation failed: %v", authzErr)
		return nil, authzErr
	}

	signalLatency := time.Since(signalStart).Milliseconds()
	r.applySignalResultsToContext(ctx, signals)
	logSignalEvaluationResults(signals)
	tracing.EndSignalSpan(signalSpan, collectMatchedSignalRules(signals), 1.0, signalLatency)
	ctx.TraceContext = signalCtx
	return signals, nil
}

func metaRoutingPlanDisablesCompression(plan *RefinementPlan) bool {
	if plan == nil {
		return false
	}
	for _, action := range plan.Actions {
		if action.Type == config.MetaRoutingActionDisableCompression {
			return true
		}
	}
	return false
}

func metaRoutingPlannedFamilies(plan *RefinementPlan) []string {
	if plan == nil {
		return nil
	}
	return plannedSignalFamilies(plan)
}

func chooseMetaRoutingFinalPass(
	mode string,
	basePass *metaRoutingPassExecution,
	refinedPass *metaRoutingPassExecution,
) *metaRoutingPassExecution {
	if refinedPass == nil || mode == config.MetaRoutingModeShadow {
		return basePass
	}
	if refinedPass.result == nil && basePass != nil && basePass.result != nil {
		return basePass
	}
	return refinedPass
}

func metaRoutingDecisionOverturned(basePass *metaRoutingPassExecution, chosenPass *metaRoutingPassExecution) bool {
	if basePass == nil || chosenPass == nil {
		return false
	}
	return basePass.decisionName != chosenPass.decisionName || basePass.selectedModel != chosenPass.selectedModel
}

func cloneRequestContextForMetaRouting(src *RequestContext) *RequestContext {
	if src == nil {
		return &RequestContext{}
	}

	clone := *src
	if src.Headers != nil {
		clone.Headers = make(map[string]string, len(src.Headers))
		for key, value := range src.Headers {
			clone.Headers[key] = value
		}
	}
	if src.StreamingChunks != nil {
		clone.StreamingChunks = append([]string(nil), src.StreamingChunks...)
	}
	if src.StreamingMetadata != nil {
		clone.StreamingMetadata = make(map[string]interface{}, len(src.StreamingMetadata))
		for key, value := range src.StreamingMetadata {
			clone.StreamingMetadata[key] = value
		}
	}
	clone.PIIEntities = append([]string(nil), src.PIIEntities...)
	clone.MetaRoutingTrace = nil
	clone.MetaRoutingFeedbackID = ""
	clone.MetaRoutingFeedbackWritten = false
	return &clone
}

func commitMetaRoutingPassContext(dst *RequestContext, passCtx *RequestContext) {
	if dst == nil || passCtx == nil {
		return
	}

	trace := dst.MetaRoutingTrace
	feedbackID := dst.MetaRoutingFeedbackID
	feedbackWritten := dst.MetaRoutingFeedbackWritten
	routerReplayID := dst.RouterReplayID
	routerReplayRecorder := dst.RouterReplayRecorder
	*dst = *passCtx
	dst.MetaRoutingTrace = trace
	dst.MetaRoutingFeedbackID = feedbackID
	dst.MetaRoutingFeedbackWritten = feedbackWritten
	dst.RouterReplayID = routerReplayID
	dst.RouterReplayRecorder = routerReplayRecorder
}
