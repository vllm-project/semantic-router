package extproc

import (
	"slices"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) recordMetaRoutingBasePass(
	ctx *RequestContext,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	result *decision.DecisionResult,
	selectedModel string,
	latency time.Duration,
) {
	if ctx == nil || r == nil || r.Config == nil || !r.Config.MetaRouting.Enabled() {
		return
	}

	trace := ensureMetaRoutingTrace(ctx, r.Config.MetaRouting, r.metaRoutingPolicyProvider().Descriptor())
	pass := buildMetaRoutingPassTrace(0, metaRoutingPassKindBase, signalInput, signals, result, ctx, selectedModel, latency)
	assessment := r.metaRoutingPolicyProvider().Assess(r.Config.MetaRouting, signalInput, signals, pass)
	if assessment != nil {
		pass.Assessment = assessment
		pass.TraceQuality.Fragile = assessment.NeedsRefine
	}
	plan := r.metaRoutingPolicyProvider().Plan(r.Config.MetaRouting, signalInput, signals, assessment)

	appendMetaRoutingPass(trace, pass)
	finalizeMetaRoutingTrace(
		trace,
		pass,
		assessment,
		plan,
		plannedSignalFamilies(plan),
		false,
	)
	logMetaRoutingObservation(pass)
}

func ensureMetaRoutingTrace(
	ctx *RequestContext,
	metaCfg config.MetaRoutingConfig,
	provider *MetaRoutingPolicyDescriptor,
) *RoutingTrace {
	if ctx.MetaRoutingTrace != nil {
		return ctx.MetaRoutingTrace
	}
	ctx.MetaRoutingTrace = &RoutingTrace{
		Mode:           metaCfg.Mode,
		MaxPasses:      metaCfg.MaxPasses,
		PolicyProvider: cloneMetaRoutingPolicyDescriptor(provider),
	}
	return ctx.MetaRoutingTrace
}

func cloneMetaRoutingPolicyDescriptor(input *MetaRoutingPolicyDescriptor) *MetaRoutingPolicyDescriptor {
	if input == nil {
		return nil
	}
	cloned := *input
	return &cloned
}

func buildMetaRoutingPassTrace(
	index int,
	kind string,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	result *decision.DecisionResult,
	ctx *RequestContext,
	selectedModel string,
	latency time.Duration,
) PassTrace {
	pass := PassTrace{
		Index:               index,
		Kind:                kind,
		LatencyMs:           latency.Milliseconds(),
		InputCompressed:     signalInput.compressedText != "" && signalInput.compressedText != signalInput.evaluationText,
		SelectedModel:       selectedModel,
		CategoryName:        ctx.VSRSelectedCategory,
		SelectionMethod:     ctx.VSRSelectionMethod,
		MatchedSignalCounts: metaRoutingMatchedSignalCounts(signals),
		PartitionConflicts:  metaRoutingPartitionConflictNames(signals),
	}

	if result != nil {
		pass.DecisionName = result.Decision.Name
		pass.DecisionConfidence = result.Confidence
		pass.DecisionMargin = result.DecisionMargin
		pass.DecisionCandidateCount = result.CandidateCount
		pass.DecisionWinnerBasis = result.DecisionWinnerBasis
		if result.RunnerUp != nil {
			pass.RunnerUpDecisionName = result.RunnerUp.Name
			pass.RunnerUpConfidence = result.RunnerUp.Confidence
		}
	}

	pass.TraceQuality = TraceQuality{
		SignalDominance:               metaRoutingSignalDominance(pass.MatchedSignalCounts),
		AvgSignalConfidence:           metaRoutingAverageSignalConfidence(signals),
		DecisionMargin:                pass.DecisionMargin,
		ProjectionBoundaryMinDistance: metaRoutingProjectionBoundaryMinDistance(signals),
	}
	return pass
}

func appendMetaRoutingPass(trace *RoutingTrace, pass PassTrace) {
	if trace == nil {
		return
	}
	trace.Passes = append(trace.Passes, pass)
	trace.PassCount = len(trace.Passes)
}

func finalizeMetaRoutingTrace(
	trace *RoutingTrace,
	finalPass PassTrace,
	finalAssessment *MetaAssessment,
	plan *RefinementPlan,
	refinedFamilies []string,
	overturned bool,
) {
	if trace == nil {
		return
	}

	trace.FinalDecisionName = finalPass.DecisionName
	trace.FinalDecisionConfidence = finalPass.DecisionConfidence
	trace.FinalModel = finalPass.SelectedModel
	trace.FinalAssessment = finalAssessment
	trace.TriggerNames = assessmentTriggers(planAssessmentOrFallback(plan, finalAssessment))
	trace.FinalPlan = plan
	trace.RefinedSignalFamilies = uniqueSortedStrings(refinedFamilies)
	trace.OverturnedDecision = overturned

	if len(trace.Passes) > 1 {
		basePass := trace.Passes[0]
		lastPass := trace.Passes[len(trace.Passes)-1]
		trace.LatencyDeltaMs = lastPass.LatencyMs - basePass.LatencyMs
		trace.DecisionMarginDelta = lastPass.DecisionMargin - basePass.DecisionMargin
		trace.ProjectionBoundaryDelta = metaRoutingBoundaryDelta(basePass.TraceQuality.ProjectionBoundaryMinDistance, lastPass.TraceQuality.ProjectionBoundaryMinDistance)
	}
}

func planAssessmentOrFallback(plan *RefinementPlan, fallback *MetaAssessment) *MetaAssessment {
	if plan == nil {
		return fallback
	}
	return &MetaAssessment{Triggers: append([]string(nil), plan.TriggerNames...)}
}

func metaRoutingBoundaryDelta(base *float64, other *float64) *float64 {
	if base == nil || other == nil {
		return nil
	}
	delta := *other - *base
	return &delta
}

func assessMetaRoutingPass(
	metaCfg config.MetaRoutingConfig,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	pass PassTrace,
) *MetaAssessment {
	assessment := &MetaAssessment{
		TraceQuality: pass.TraceQuality,
	}

	policy := metaCfg.TriggerPolicy
	if policy == nil {
		return assessment
	}

	applyDecisionMarginTrigger(assessment, policy, pass)
	applyProjectionBoundaryTrigger(assessment, policy, pass)
	applyPartitionConflictTrigger(assessment, policy, pass)
	applyRequiredFamilyTriggers(assessment, policy.RequiredFamilies, signals)
	applyFamilyDisagreementTriggers(assessment, policy.FamilyDisagreements, signals)
	finalizeMetaRoutingAssessment(assessment, pass.InputCompressed)
	return assessment
}

func applyDecisionMarginTrigger(
	assessment *MetaAssessment,
	policy *config.MetaTriggerPolicy,
	pass PassTrace,
) {
	if policy.DecisionMarginBelow == nil || pass.DecisionCandidateCount <= 1 {
		return
	}
	if pass.TraceQuality.DecisionMargin >= *policy.DecisionMarginBelow {
		return
	}
	assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerLowDecisionMargin)
	assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseDecisionOverlap)
}

func applyProjectionBoundaryTrigger(
	assessment *MetaAssessment,
	policy *config.MetaTriggerPolicy,
	pass PassTrace,
) {
	if policy.ProjectionBoundaryWithin == nil || pass.TraceQuality.ProjectionBoundaryMinDistance == nil {
		return
	}
	if *pass.TraceQuality.ProjectionBoundaryMinDistance >= *policy.ProjectionBoundaryWithin {
		return
	}
	assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerProjectionBoundary)
	assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseProjectionBoundary)
}

func applyPartitionConflictTrigger(
	assessment *MetaAssessment,
	policy *config.MetaTriggerPolicy,
	pass PassTrace,
) {
	if policy.PartitionConflict == nil || !*policy.PartitionConflict || len(pass.PartitionConflicts) == 0 {
		return
	}
	assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerPartitionConflict)
	assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCausePartitionConflict)
}

func applyRequiredFamilyTriggers(
	assessment *MetaAssessment,
	requiredFamilies []config.MetaRequiredSignalFamily,
	signals *classification.SignalResults,
) {
	for _, family := range requiredFamilies {
		applyRequiredFamilyThresholds(assessment, family, signals)
	}
}

func applyRequiredFamilyThresholds(
	assessment *MetaAssessment,
	family config.MetaRequiredSignalFamily,
	signals *classification.SignalResults,
) {
	matchedCount := metaRoutingFamilyMatchCount(signals, family.Type)
	if family.MinMatches != nil && matchedCount < *family.MinMatches {
		assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerRequiredFamilyMissing)
		assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseMissingRequiredFamily)
	}
	if family.MinConfidence == nil {
		return
	}

	bestConfidence, ok := metaRoutingBestFamilyConfidence(signals, family.Type)
	if !ok {
		assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerRequiredFamilyMissing)
		assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseMissingRequiredFamily)
		return
	}
	if bestConfidence < *family.MinConfidence {
		assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerRequiredFamilyLowConf)
		assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseLowConfidenceFamily)
	}
}

func applyFamilyDisagreementTriggers(
	assessment *MetaAssessment,
	disagreements []config.MetaSignalFamilyDisagreement,
	signals *classification.SignalResults,
) {
	for _, disagreement := range disagreements {
		if !metaRoutingFamiliesDisagree(signals, disagreement.Cheap, disagreement.Expensive) {
			continue
		}
		assessment.Triggers = appendUniqueString(assessment.Triggers, metaRoutingTriggerSignalFamilyDisagreement)
		assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseFamilyDisagreement)
	}
}

func finalizeMetaRoutingAssessment(assessment *MetaAssessment, inputCompressed bool) {
	assessment.NeedsRefine = len(assessment.Triggers) > 0
	if assessment.NeedsRefine && inputCompressed {
		assessment.RootCauses = appendUniqueString(assessment.RootCauses, metaRoutingCauseCompressionLossRisk)
	}
	assessment.TraceQuality.Fragile = assessment.NeedsRefine
}

func buildMetaRoutingPlan(
	metaCfg config.MetaRoutingConfig,
	signalInput signalEvaluationInput,
	signals *classification.SignalResults,
	assessment *MetaAssessment,
) *RefinementPlan {
	if assessment == nil || !assessment.NeedsRefine || len(metaCfg.AllowedActions) == 0 {
		return nil
	}

	plan := &RefinementPlan{
		MaxPasses:    metaCfg.MaxPasses,
		TriggerNames: append([]string(nil), assessment.Triggers...),
		RootCauses:   append([]string(nil), assessment.RootCauses...),
	}

	relevantFamilies := metaRoutingRelevantFamilies(metaCfg, signals, assessment)
	for _, action := range metaCfg.AllowedActions {
		switch strings.TrimSpace(strings.ToLower(action.Type)) {
		case config.MetaRoutingActionDisableCompression:
			if signalInput.compressedText == "" || signalInput.compressedText == signalInput.evaluationText {
				continue
			}
			if !slices.Contains(assessment.RootCauses, metaRoutingCauseCompressionLossRisk) {
				continue
			}
			plan.Actions = append(plan.Actions, RefinementActionPlan{Type: action.Type})
		case config.MetaRoutingActionRerunSignalFamilies:
			families := intersectStringSets(action.SignalFamilies, relevantFamilies)
			if len(families) == 0 {
				continue
			}
			plan.Actions = append(plan.Actions, RefinementActionPlan{
				Type:           action.Type,
				SignalFamilies: families,
			})
		}
	}

	if len(plan.Actions) == 0 {
		return nil
	}
	return plan
}

func assessmentTriggers(assessment *MetaAssessment) []string {
	if assessment == nil || len(assessment.Triggers) == 0 {
		return nil
	}
	return append([]string(nil), assessment.Triggers...)
}

func plannedSignalFamilies(plan *RefinementPlan) []string {
	if plan == nil {
		return nil
	}
	families := make([]string, 0)
	for _, action := range plan.Actions {
		families = append(families, action.SignalFamilies...)
	}
	return uniqueSortedStrings(families)
}

func logMetaRoutingObservation(pass PassTrace) {
	if pass.Assessment == nil {
		return
	}
	logging.Debugf(
		"[MetaRouting] pass=%d kind=%s decision=%s model=%s fragile=%t triggers=%v root_causes=%v margin=%.4f boundary=%v",
		pass.Index,
		pass.Kind,
		pass.DecisionName,
		pass.SelectedModel,
		pass.TraceQuality.Fragile,
		pass.Assessment.Triggers,
		pass.Assessment.RootCauses,
		pass.TraceQuality.DecisionMargin,
		pass.TraceQuality.ProjectionBoundaryMinDistance,
	)
}

func metaRoutingMatchedSignalCounts(signals *classification.SignalResults) map[string]int {
	if signals == nil {
		return nil
	}

	counts := map[string]int{
		config.SignalTypeKeyword:      len(signals.MatchedKeywordRules),
		config.SignalTypeEmbedding:    len(signals.MatchedEmbeddingRules),
		config.SignalTypeDomain:       len(signals.MatchedDomainRules),
		config.SignalTypeFactCheck:    len(signals.MatchedFactCheckRules),
		config.SignalTypeUserFeedback: len(signals.MatchedUserFeedbackRules),
		config.SignalTypePreference:   len(signals.MatchedPreferenceRules),
		config.SignalTypeLanguage:     len(signals.MatchedLanguageRules),
		config.SignalTypeContext:      len(signals.MatchedContextRules),
		config.SignalTypeStructure:    len(signals.MatchedStructureRules),
		config.SignalTypeComplexity:   len(signals.MatchedComplexityRules),
		config.SignalTypeModality:     len(signals.MatchedModalityRules),
		config.SignalTypeAuthz:        len(signals.MatchedAuthzRules),
		config.SignalTypeJailbreak:    len(signals.MatchedJailbreakRules),
		config.SignalTypePII:          len(signals.MatchedPIIRules),
		config.SignalTypeProjection:   len(signals.MatchedProjectionRules),
	}

	for family, count := range counts {
		if count == 0 {
			delete(counts, family)
		}
	}
	return counts
}

func metaRoutingSignalDominance(counts map[string]int) float64 {
	if len(counts) == 0 {
		return 0
	}

	total := 0
	best := 0
	for _, count := range counts {
		total += count
		if count > best {
			best = count
		}
	}
	if total == 0 {
		return 0
	}
	return float64(best) / float64(total)
}

func metaRoutingAverageSignalConfidence(signals *classification.SignalResults) float64 {
	if signals == nil {
		return 0
	}

	total := 0.0
	count := 0
	for family, rules := range metaRoutingMatchedRuleSets(signals) {
		for _, rule := range rules {
			total += metaRoutingSignalConfidence(signals.SignalConfidences, family, rule)
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

func metaRoutingProjectionBoundaryMinDistance(signals *classification.SignalResults) *float64 {
	if signals == nil || len(signals.ProjectionBoundaryDistances) == 0 {
		return nil
	}

	var best *float64
	for _, distance := range signals.ProjectionBoundaryDistances {
		current := distance
		if best == nil || current < *best {
			best = &current
		}
	}
	return best
}

func metaRoutingPartitionConflictNames(signals *classification.SignalResults) []string {
	if signals == nil || len(signals.ProjectionPartitionConflicts) == 0 {
		return nil
	}

	names := make([]string, 0, len(signals.ProjectionPartitionConflicts))
	for _, conflict := range signals.ProjectionPartitionConflicts {
		names = append(names, conflict.Name)
	}
	return uniqueSortedStrings(names)
}

func metaRoutingFamilyMatchCount(signals *classification.SignalResults, family string) int {
	return len(metaRoutingMatchedRulesForFamily(signals, family))
}

func metaRoutingBestFamilyConfidence(signals *classification.SignalResults, family string) (float64, bool) {
	rules := metaRoutingMatchedRulesForFamily(signals, family)
	if len(rules) == 0 {
		return 0, false
	}

	best := 0.0
	for _, rule := range rules {
		confidence := metaRoutingSignalConfidence(signals.SignalConfidences, family, rule)
		if confidence > best {
			best = confidence
		}
	}
	return best, true
}

func metaRoutingFamiliesDisagree(signals *classification.SignalResults, left string, right string) bool {
	leftRules := uniqueSortedStrings(metaRoutingMatchedRulesForFamily(signals, left))
	rightRules := uniqueSortedStrings(metaRoutingMatchedRulesForFamily(signals, right))
	if len(leftRules) == 0 && len(rightRules) == 0 {
		return false
	}
	return !slices.Equal(leftRules, rightRules)
}

func metaRoutingRelevantFamilies(
	metaCfg config.MetaRoutingConfig,
	signals *classification.SignalResults,
	assessment *MetaAssessment,
) []string {
	if assessment == nil || !assessment.NeedsRefine || metaCfg.TriggerPolicy == nil {
		return nil
	}

	families := make([]string, 0)
	for _, family := range metaCfg.TriggerPolicy.RequiredFamilies {
		matchedCount := metaRoutingFamilyMatchCount(signals, family.Type)
		if family.MinMatches != nil && matchedCount < *family.MinMatches {
			families = append(families, family.Type)
			continue
		}
		if family.MinConfidence == nil {
			continue
		}
		bestConfidence, ok := metaRoutingBestFamilyConfidence(signals, family.Type)
		if !ok || bestConfidence < *family.MinConfidence {
			families = append(families, family.Type)
		}
	}

	for _, disagreement := range metaCfg.TriggerPolicy.FamilyDisagreements {
		if metaRoutingFamiliesDisagree(signals, disagreement.Cheap, disagreement.Expensive) {
			families = append(families, disagreement.Cheap, disagreement.Expensive)
		}
	}
	return uniqueSortedStrings(families)
}

func metaRoutingMatchedRuleSets(signals *classification.SignalResults) map[string][]string {
	if signals == nil {
		return nil
	}
	return map[string][]string{
		config.SignalTypeKeyword:      signals.MatchedKeywordRules,
		config.SignalTypeEmbedding:    signals.MatchedEmbeddingRules,
		config.SignalTypeDomain:       signals.MatchedDomainRules,
		config.SignalTypeFactCheck:    signals.MatchedFactCheckRules,
		config.SignalTypeUserFeedback: signals.MatchedUserFeedbackRules,
		config.SignalTypePreference:   signals.MatchedPreferenceRules,
		config.SignalTypeLanguage:     signals.MatchedLanguageRules,
		config.SignalTypeContext:      signals.MatchedContextRules,
		config.SignalTypeStructure:    signals.MatchedStructureRules,
		config.SignalTypeComplexity:   signals.MatchedComplexityRules,
		config.SignalTypeModality:     signals.MatchedModalityRules,
		config.SignalTypeAuthz:        signals.MatchedAuthzRules,
		config.SignalTypeJailbreak:    signals.MatchedJailbreakRules,
		config.SignalTypePII:          signals.MatchedPIIRules,
		config.SignalTypeProjection:   signals.MatchedProjectionRules,
	}
}

func metaRoutingMatchedRulesForFamily(signals *classification.SignalResults, family string) []string {
	ruleSets := metaRoutingMatchedRuleSets(signals)
	if len(ruleSets) == 0 {
		return nil
	}
	return append([]string(nil), ruleSets[strings.TrimSpace(strings.ToLower(family))]...)
}

func metaRoutingSignalConfidence(confidences map[string]float64, family string, rule string) float64 {
	if confidences == nil {
		return 1.0
	}
	key := strings.TrimSpace(strings.ToLower(family)) + ":" + rule
	if confidence, ok := confidences[key]; ok && confidence > 0 {
		return confidence
	}
	return 1.0
}

func intersectStringSets(left []string, right []string) []string {
	if len(left) == 0 || len(right) == 0 {
		return nil
	}

	rightSet := make(map[string]struct{}, len(right))
	for _, value := range right {
		rightSet[strings.TrimSpace(strings.ToLower(value))] = struct{}{}
	}

	intersection := make([]string, 0)
	for _, value := range left {
		normalized := strings.TrimSpace(strings.ToLower(value))
		if _, ok := rightSet[normalized]; ok {
			intersection = append(intersection, normalized)
		}
	}
	return uniqueSortedStrings(intersection)
}

func uniqueSortedStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		normalized := strings.TrimSpace(strings.ToLower(value))
		if normalized == "" {
			continue
		}
		if _, ok := seen[normalized]; ok {
			continue
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	slices.Sort(result)
	return result
}

func appendUniqueString(values []string, candidate string) []string {
	normalized := strings.TrimSpace(strings.ToLower(candidate))
	if normalized == "" {
		return values
	}
	if slices.Contains(values, normalized) {
		return values
	}
	return append(values, normalized)
}
