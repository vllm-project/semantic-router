package extproc

import (
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// responseTrace encodes x-vsr-trace from what the request recorded, for a
// caller global.router.headers.debug allows, and is "" for every other
// caller. The full detail stays in the replay record.
func responseTrace(ctx *RequestContext) string {
	if ctx == nil || !debugHeadersRequested(ctx) {
		return ""
	}
	var trace headers.Trace
	addBenchmarkTrace(&trace, ctx)
	addRoutingTrace(&trace, ctx)
	if signals := matchedSignalTrace(ctx); len(signals) > 0 {
		trace.Set(headers.TraceSignals, headers.List(signals...))
	}
	if coverage := scanCoverageTrace(ctx.VSRScanCoverage); len(coverage) > 0 {
		trace.Set(headers.TraceScanCoverage, headers.List(coverage...))
	}
	if ctx.VSRContextTokenCount > 0 {
		trace.Set(headers.TraceContextTokens, headers.Integer(int64(ctx.VSRContextTokenCount)))
	}
	addDecisionDetailTrace(&trace, ctx)
	addLearningTrace(&trace, ctx)
	ctx.LooperTrace.addTo(&trace)
	ctx.ToolsSelection.addTo(&trace)
	addResponseGuardTrace(&trace, ctx)
	return trace.Encode()
}

// scanCoverageTrace lists, in rule order, the guard rules that read their
// input in windows and how much of it they read.
func scanCoverageTrace(coverage map[string]string) []headers.Item {
	items := make([]headers.Item, 0, len(coverage))
	for _, rule := range slices.Sorted(maps.Keys(coverage)) {
		var read, windows int64
		if _, err := fmt.Sscanf(coverage[rule], "partial %d/%d", &read, &windows); err == nil {
			items = append(items, headers.String(rule).With("coverage", headers.Token("partial")).
				With("read", headers.Integer(read)).With("windows", headers.Integer(windows)))
			continue
		}
		items = append(items, headers.String(rule).With("coverage", headers.Token(coverage[rule])))
	}
	return items
}

// addResponseGuardTrace adds what the selected decision's response_guard did
// with the answer, and whether the Router held a streamed answer for it.
func addResponseGuardTrace(trace *headers.Trace, ctx *RequestContext) {
	outcome := ""
	if hold := ctx.ResponseGuardHold; hold != nil && hold.outcome != "" {
		outcome = hold.outcome
	} else if ctx.ResponseGuard != nil {
		outcome = ctx.ResponseGuard.outcome()
	}
	if outcome == "" {
		return
	}
	item := headers.Token(outcome)
	if ctx.ResponseGuardHold != nil {
		item = item.With("buffered", headers.Boolean(true))
	}
	trace.Set(headers.TraceResponseGuard, item)
}

func addBenchmarkTrace(trace *headers.Trace, ctx *RequestContext) {
	if ctx.BenchmarkConfigHash == "" {
		return
	}
	trace.Set(headers.TraceConfigHash, headers.String(ctx.BenchmarkConfigHash))
	if ctx.BenchmarkModelUsage != "" {
		trace.Set(headers.TraceModelUsage, headers.String(ctx.BenchmarkModelUsage))
	} else if ctx.BenchmarkInferenceCalls > 0 {
		trace.Set(headers.TraceInferenceCalls, headers.Integer(int64(ctx.BenchmarkInferenceCalls)))
	}
}

// addRoutingTrace adds how the model was chosen: the decision's confidence,
// the algorithm, the time it took, fallback, the unknown policies applied,
// both wire protocols and the automatic-output budget.
func addRoutingTrace(trace *headers.Trace, ctx *RequestContext) {
	if ctx.VSRSelectedDecisionName != "" && ctx.VSRSelectedDecisionConfidenceScored && ctx.VSRSelectedDecisionConfidence >= 0 {
		trace.Set(headers.TraceConfidence, headers.Decimal(ctx.VSRSelectedDecisionConfidence))
	}
	if ctx.VSRSelectionMethod != "" {
		trace.Set(headers.TraceAlgorithm, headers.Token(ctx.VSRSelectionMethod))
	}
	if ctx.RoutingLatency > 0 {
		trace.Set(headers.TraceRoutingLatencyMs, headers.Decimal(float64(ctx.RoutingLatency)/float64(time.Millisecond)))
	}
	if record := ctx.FallbackRecord; record != nil && record.FinalStatus == "succeeded" && len(record.Attempts) > 1 {
		trace.Set(headers.TraceFallbackAttempts, headers.Integer(int64(len(record.Attempts))))
	}
	if policies := ctx.VSRDecisionDiagnostics.AppliedUnknownPolicies; len(policies) > 0 {
		decisions := make([]headers.Item, 0, len(policies))
		for _, name := range slices.Sorted(maps.Keys(policies)) {
			decisions = append(decisions, headers.String(name).With("policy", headers.Token(policies[name])))
		}
		trace.Set(headers.TraceOnUnknown, headers.List(decisions...))
	}
	trace.Set(headers.TraceClientProtocol, headers.Token(normalizeProtocol(string(ctx.SourceFormat))))
	trace.Set(headers.TraceUpstreamProtocol, headers.Token(normalizeProtocol(string(ctx.TargetFormat))))
	if ctx.SkipProcessing || ctx.LooperRequest || ctx.VSRCacheHit || ctx.SemanticRequest == nil {
		return
	}
	if sampling := ctx.SemanticRequest.Sampling; sampling.AutomaticOutput && sampling.AutomaticInputTokens != nil &&
		sampling.MaxOutputTokens != nil && *sampling.AutomaticInputTokens > 0 && *sampling.MaxOutputTokens > 0 {
		trace.Set(headers.TraceEffectiveInputTokens, headers.Integer(*sampling.AutomaticInputTokens))
		trace.Set(headers.TraceEffectiveMaxOutputTokens, headers.Integer(*sampling.MaxOutputTokens))
	}
}

// matchedSignalTrace lists the matched rules as type:name, each family in
// the order it is evaluated, with the response-stage matches once the
// selected decision's response_guard has read the answer.
func matchedSignalTrace(ctx *RequestContext) []headers.Item {
	var response map[string][]string
	if observed := ctx.ResponseGuard.observed(); observed != nil {
		response = observed.Reported
	}
	families := []struct {
		signalType string
		matched    []string
	}{
		{config.SignalTypeKeyword, ctx.VSRMatchedKeywords},
		{config.SignalTypeDomain, ctx.VSRMatchedDomains},
		{config.SignalTypeFactCheck, ctx.VSRMatchedFactCheck},
		{config.SignalTypeUserFeedback, ctx.VSRMatchedUserFeedback},
		{config.SignalTypeTurnRelation, ctx.VSRMatchedTurnRelation},
		{config.SignalTypeLanguage, ctx.VSRMatchedLanguage},
		{config.SignalTypeContext, ctx.VSRMatchedContext},
		{config.SignalTypeComplexity, ctx.VSRMatchedComplexity},
		{config.SignalTypeModality, ctx.VSRMatchedModality},
		{config.SignalTypeAuthz, ctx.VSRMatchedAuthz},
		{config.SignalTypeJailbreak, append(slices.Clone(ctx.VSRMatchedJailbreak), response[config.SignalTypeJailbreak]...)},
		{config.SignalTypeSafety, ctx.VSRMatchedSafety},
		{config.SignalTypeHallucination, response[config.SignalTypeHallucination]},
		{config.SignalTypePII, ctx.VSRMatchedPII},
		{config.SignalTypeConversation, ctx.VSRMatchedConversation},
		{config.SignalTypeMetadata, ctx.VSRMatchedMetadata},
		{config.SignalTypeInputModality, ctx.VSRMatchedInputModality},
		{config.SignalTypeSystemOne, append(slices.Clone(ctx.VSRMatchedDecisionModel), response[config.SignalTypeSystemOne]...)},
		{config.SignalTypeProjection, ctx.VSRMatchedProjection},
	}
	var items []headers.Item
	for _, family := range families {
		for _, name := range family.matched {
			items = append(items, headers.String(family.signalType+":"+name))
		}
	}
	return items
}

// addDecisionDetailTrace adds what shaped the request on the way to its
// model: the category and reasoning chosen, the output kinds of the
// decision's output contract, the injected system prompt, the cache lookup,
// the retention directive, the prompt cache receipt, the session phase and
// the decision ranking.
func addDecisionDetailTrace(trace *headers.Trace, ctx *RequestContext) {
	if ctx.VSRSelectedCategory != "" {
		trace.Set(headers.TraceCategory, headers.String(ctx.VSRSelectedCategory))
	}
	if ctx.VSRReasoningMode != "" {
		trace.Set(headers.TraceReasoning, headers.Token(ctx.VSRReasoningMode))
	}
	if decision := ctx.VSRSelectedDecision; decision != nil && decision.Output != nil {
		kinds := decision.Output.Kinds()
		items := make([]headers.Item, 0, len(kinds))
		for _, kind := range kinds {
			items = append(items, headers.Token(kind))
		}
		trace.Set(headers.TraceModality, headers.List(items...))
	}
	if ctx.VSRInjectedSystemPrompt {
		trace.Set(headers.TraceSystemPrompt, headers.Boolean(true))
	}
	if ctx.VSRCacheSimilarity > 0 {
		trace.Set(headers.TraceCacheSimilarity, headers.Decimal(float64(ctx.VSRCacheSimilarity)))
	}
	if retention := retentionTrace(ctx.Retention); retention != nil {
		trace.Set(headers.TraceRetention, *retention)
	}
	if ctx.PromptCacheAction != "" {
		item := headers.Token(ctx.PromptCacheAction)
		if ctx.PromptCacheReason != "" {
			item = item.With("reason", headers.Token(ctx.PromptCacheReason))
		}
		if ctx.PromptCacheInserted > 0 {
			item = item.With("inserted", headers.Integer(int64(ctx.PromptCacheInserted)))
		}
		if ctx.PromptCachePreserved > 0 {
			item = item.With("preserved", headers.Integer(int64(ctx.PromptCachePreserved)))
		}
		trace.Set(headers.TracePromptCache, item)
	}
	if phase := sessionPolicyPhase(ctx); phase != "" {
		trace.Set(headers.TraceSessionPhase, headers.Token(phase))
	}
	if ranking := ctx.VSRDecisionDiagnostics.Ranking; ranking != nil && ranking.RunnerUp != "" && ranking.Winner == ctx.VSRSelectedDecisionName {
		trace.Set(headers.TraceRanking, headers.String(ranking.RunnerUp).With("reason", headers.String(ranking.Reason)))
	}
}

// retentionTrace is the fields the matched decision's retention directive
// sets; an unset field is left out rather than reported as a default.
func retentionTrace(retention *config.DecisionRetention) *headers.Item {
	if retention == nil {
		return nil
	}
	item := headers.Boolean(true)
	set := false
	if retention.Drop != nil {
		item, set = item.With("drop", headers.Boolean(*retention.Drop)), true
	}
	if retention.TTLTurns != nil && *retention.TTLTurns >= 0 {
		item, set = item.With("ttl_turns", headers.Integer(int64(*retention.TTLTurns))), true
	}
	if retention.KeepCurrentModel != nil {
		item, set = item.With("keep_current_model", headers.Boolean(*retention.KeepCurrentModel)), true
	}
	if retention.PreferPrefixRetention != nil {
		item, set = item.With("prefer_prefix", headers.Boolean(*retention.PreferPrefixRetention)), true
	}
	if !set {
		return nil
	}
	return &item
}

// addLearningTrace adds each Router Learning method with its action, scope
// and reason.
func addLearningTrace(trace *headers.Trace, ctx *RequestContext) {
	policies := learningPoliciesForTrace(ctx)
	if len(policies) == 0 {
		return
	}
	methods := make([]headers.Item, 0, len(policies))
	for _, policy := range policies {
		item := headers.Token(string(policy.Method))
		for _, field := range []struct {
			key   string
			field routerLearningPolicyField
		}{
			{"action", learningPolicyFieldAction},
			{"scope", learningPolicyFieldScope},
			{"reason", learningPolicyFieldReason},
		} {
			if value := policy.StringField(field.field); value != "" {
				item = item.With(field.key, headers.Token(value))
			}
		}
		methods = append(methods, item)
	}
	trace.Set(headers.TraceLearning, headers.List(methods...))
}

func sessionPolicyPhase(ctx *RequestContext) string {
	if policy, ok := protectionLearningPolicyForContext(ctx); ok {
		return policy.SessionPhase()
	}
	return ""
}

func learningPoliciesForTrace(ctx *RequestContext) []routerLearningPolicy {
	if !ctx.VSRLearningPolicies.Empty() {
		policies := make([]routerLearningPolicy, 0, 2)
		if policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation); ok {
			policies = append(policies, policy)
		}
		if policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection); ok {
			policies = append(policies, policy)
		}
		return policies
	}
	if ctx.VSRLearningPolicy == nil || ctx.VSRLearningPolicy.Empty() {
		return nil
	}
	policy := *ctx.VSRLearningPolicy
	if policy.Method == "" {
		policy.Method = routerLearningMethodProtection
	}
	return []routerLearningPolicy{policy}
}

// looperTrace is what a Looper execution reports: the models it called, its
// algorithm and final model, how many calls it made, how long they took and
// the tokens they used.
type looperTrace struct {
	algorithm  string
	model      string
	modelsUsed []string
	iterations int
	latencyMs  int64
	usage      looper.TokenUsage
}

func newLooperTrace(resp *looper.Response) *looperTrace {
	if resp == nil {
		return nil
	}
	return &looperTrace{
		algorithm: resp.AlgorithmType, model: resp.Model, modelsUsed: slices.Clone(resp.ModelsUsed),
		iterations: resp.Iterations, latencyMs: resp.LatencyMs, usage: resp.Usage,
	}
}

func (l *looperTrace) addTo(trace *headers.Trace) {
	if l == nil {
		return
	}
	models := make([]headers.Item, len(l.modelsUsed))
	for i, model := range l.modelsUsed {
		models[i] = headers.String(model)
	}
	item := headers.List(models...)
	if l.algorithm != "" {
		item = item.With("algorithm", headers.Token(l.algorithm))
	}
	if l.model != "" {
		item = item.With("model", headers.String(l.model))
	}
	item = item.With("iterations", headers.Integer(int64(l.iterations)))
	// A zero here means "not measured", such as an algorithm that does not
	// aggregate usage, rather than a free execution, so it is left out.
	for _, count := range []struct {
		key   string
		value int64
	}{
		{"latency_ms", l.latencyMs},
		{"prompt_tokens", l.usage.PromptTokens},
		{"completion_tokens", l.usage.CompletionTokens},
		{"total_tokens", l.usage.TotalTokens},
	} {
		if count.value > 0 {
			item = item.With(count.key, headers.Integer(count.value))
		}
	}
	trace.Set(headers.TraceLooper, item)
}

func (s *toolsSelection) addTo(trace *headers.Trace) {
	if s == nil || s.outcome == "" {
		return
	}
	item := headers.Token(s.outcome).
		With("candidates", headers.Integer(int64(s.candidates))).
		With("kept", headers.Integer(int64(len(s.kept))))
	if s.latency > 0 {
		item = item.With("latency_ms", headers.Decimal(float64(s.latency)/float64(time.Millisecond)))
	}
	trace.Set(headers.TraceTools, item)
}
