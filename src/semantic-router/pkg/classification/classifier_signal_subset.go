package classification

import (
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// SignalFamilyEvaluationRequest narrows signal evaluation to the selected
// families while preserving the standard request-time inputs.
type SignalFamilyEvaluationRequest struct {
	Text                 string
	ContextText          string
	NonUserMessages      []string
	Headers              map[string]string
	ImageURL             string
	UncompressedText     string
	SkipCompressionRules map[string]bool
	SignalFamilies       []string
}

func (c *Classifier) evaluateSignalsWithSelection(
	text string,
	contextText string,
	nonUserMessages []string,
	usedSignals map[string]bool,
	uncompressedText string,
	skipCompressionSignals map[string]bool,
	imageURL string,
) *SignalResults {
	defer c.enterSignalEvaluationLoadGate()()

	textForSignal := textForSignalFunc(text, uncompressedText, skipCompressionSignals)
	ready := c.signalReadiness()

	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	dispatchers := c.buildSignalDispatchers(results, &mu, textForSignal, contextText, nonUserMessages, imageURL)
	runSignalDispatchers(dispatchers, usedSignals, ready, &wg)
	wg.Wait()

	results = c.applySignalGroups(results)
	results = c.applySignalComposers(results)
	results = c.applyProjections(results)
	return results
}

// EvaluateSignalFamiliesWithHeaders evaluates only the requested signal families.
// The returned result is suitable for merging back into an existing SignalResults.
func (c *Classifier) EvaluateSignalFamiliesWithHeaders(req SignalFamilyEvaluationRequest) (*SignalResults, error) {
	usedSignals := c.usedSignalsForFamilies(req.SignalFamilies)
	if len(usedSignals) == 0 {
		return &SignalResults{
			Metrics:           &SignalMetricsCollection{},
			SignalConfidences: make(map[string]float64),
			SignalValues:      make(map[string]float64),
		}, nil
	}

	results := c.evaluateSignalsWithSelection(
		req.Text,
		req.ContextText,
		req.NonUserMessages,
		usedSignals,
		req.UncompressedText,
		req.SkipCompressionRules,
		req.ImageURL,
	)

	if !isSignalTypeUsed(usedSignals, config.SignalTypeAuthz) || c.authzClassifier == nil {
		return results, nil
	}

	userID := req.Headers[c.authzUserIDHeader]
	userGroups := ParseUserGroups(req.Headers[c.authzUserGroupsHeader])
	authzResult, err := c.authzClassifier.Classify(userID, userGroups)
	if err != nil {
		return nil, err
	}
	for _, ruleName := range authzResult.MatchedRules {
		results.MatchedAuthzRules = append(results.MatchedAuthzRules, ruleName)
		results.SignalConfidences["authz:"+ruleName] = 1.0
	}
	results.Metrics.Authz.Confidence = 1.0
	return results, nil
}

func (c *Classifier) usedSignalsForFamilies(signalFamilies []string) map[string]bool {
	if len(signalFamilies) == 0 {
		return nil
	}

	familySet := make(map[string]struct{}, len(signalFamilies))
	for _, family := range signalFamilies {
		family = strings.ToLower(strings.TrimSpace(family))
		if family != "" {
			familySet[family] = struct{}{}
		}
	}
	if len(familySet) == 0 {
		return nil
	}

	allSignals := c.getAllSignalTypes()
	usedSignals := make(map[string]bool)
	for key := range allSignals {
		parts := strings.SplitN(key, ":", 2)
		if len(parts) != 2 {
			continue
		}
		if _, ok := familySet[parts[0]]; ok {
			usedSignals[key] = true
		}
	}
	return usedSignals
}

// RefreshSignalFamilies merges refreshed family results into the base result and
// recomputes dependent groups, composers, and projections.
func (c *Classifier) RefreshSignalFamilies(
	base *SignalResults,
	refreshed *SignalResults,
	signalFamilies []string,
) *SignalResults {
	if base == nil {
		return refreshed
	}
	if refreshed == nil || len(signalFamilies) == 0 {
		return cloneSignalResults(base)
	}

	merged := cloneSignalResults(base)
	familySet := normalizeSignalFamilySet(signalFamilies)
	for family := range familySet {
		applySignalFamilyResults(merged, refreshed, family)
	}

	merged.MatchedProjectionRules = nil
	merged.ProjectionScores = nil
	merged.ProjectionBoundaryDistances = nil
	merged.ProjectionPartitionConflicts = nil
	merged = c.applySignalGroups(merged)
	merged = c.applySignalComposers(merged)
	merged = c.applyProjections(merged)
	return merged
}

func normalizeSignalFamilySet(signalFamilies []string) map[string]struct{} {
	familySet := make(map[string]struct{}, len(signalFamilies))
	for _, family := range signalFamilies {
		family = strings.ToLower(strings.TrimSpace(family))
		if family != "" {
			familySet[family] = struct{}{}
		}
	}
	return familySet
}

func applySignalFamilyResults(dst *SignalResults, src *SignalResults, family string) {
	if dst == nil || src == nil {
		return
	}

	switch family {
	case config.SignalTypeKeyword:
		dst.MatchedKeywordRules = cloneStrings(src.MatchedKeywordRules)
		dst.MatchedKeywords = cloneStrings(src.MatchedKeywords)
	case config.SignalTypeEmbedding:
		dst.MatchedEmbeddingRules = cloneStrings(src.MatchedEmbeddingRules)
	case config.SignalTypeDomain:
		dst.MatchedDomainRules = cloneStrings(src.MatchedDomainRules)
	case config.SignalTypeFactCheck:
		dst.MatchedFactCheckRules = cloneStrings(src.MatchedFactCheckRules)
	case config.SignalTypeUserFeedback:
		dst.MatchedUserFeedbackRules = cloneStrings(src.MatchedUserFeedbackRules)
	case config.SignalTypePreference:
		dst.MatchedPreferenceRules = cloneStrings(src.MatchedPreferenceRules)
	case config.SignalTypeLanguage:
		dst.MatchedLanguageRules = cloneStrings(src.MatchedLanguageRules)
	case config.SignalTypeContext:
		dst.MatchedContextRules = cloneStrings(src.MatchedContextRules)
		dst.TokenCount = src.TokenCount
	case config.SignalTypeStructure:
		dst.MatchedStructureRules = cloneStrings(src.MatchedStructureRules)
	case config.SignalTypeComplexity:
		dst.MatchedComplexityRules = cloneStrings(src.MatchedComplexityRules)
	case config.SignalTypeModality:
		dst.MatchedModalityRules = cloneStrings(src.MatchedModalityRules)
	case config.SignalTypeAuthz:
		dst.MatchedAuthzRules = cloneStrings(src.MatchedAuthzRules)
	case config.SignalTypeJailbreak:
		dst.MatchedJailbreakRules = cloneStrings(src.MatchedJailbreakRules)
		dst.JailbreakDetected = src.JailbreakDetected
		dst.JailbreakType = src.JailbreakType
		dst.JailbreakConfidence = src.JailbreakConfidence
	case config.SignalTypePII:
		dst.MatchedPIIRules = cloneStrings(src.MatchedPIIRules)
		dst.PIIDetected = src.PIIDetected
		dst.PIIEntities = cloneStrings(src.PIIEntities)
	}

	copySignalMetric(dst, src, family)
	copySignalConfidenceFamily(dst, src, family)
	copySignalValueFamily(dst, src, family)
}

func copySignalMetric(dst *SignalResults, src *SignalResults, family string) {
	if dst == nil || src == nil || dst.Metrics == nil || src.Metrics == nil {
		return
	}

	switch family {
	case config.SignalTypeKeyword:
		dst.Metrics.Keyword = src.Metrics.Keyword
	case config.SignalTypeEmbedding:
		dst.Metrics.Embedding = src.Metrics.Embedding
	case config.SignalTypeDomain:
		dst.Metrics.Domain = src.Metrics.Domain
	case config.SignalTypeFactCheck:
		dst.Metrics.FactCheck = src.Metrics.FactCheck
	case config.SignalTypeUserFeedback:
		dst.Metrics.UserFeedback = src.Metrics.UserFeedback
	case config.SignalTypePreference:
		dst.Metrics.Preference = src.Metrics.Preference
	case config.SignalTypeLanguage:
		dst.Metrics.Language = src.Metrics.Language
	case config.SignalTypeContext:
		dst.Metrics.Context = src.Metrics.Context
	case config.SignalTypeStructure:
		dst.Metrics.Structure = src.Metrics.Structure
	case config.SignalTypeComplexity:
		dst.Metrics.Complexity = src.Metrics.Complexity
	case config.SignalTypeModality:
		dst.Metrics.Modality = src.Metrics.Modality
	case config.SignalTypeAuthz:
		dst.Metrics.Authz = src.Metrics.Authz
	case config.SignalTypeJailbreak:
		dst.Metrics.Jailbreak = src.Metrics.Jailbreak
	case config.SignalTypePII:
		dst.Metrics.PII = src.Metrics.PII
	}
}

func copySignalConfidenceFamily(dst *SignalResults, src *SignalResults, family string) {
	if dst == nil || src == nil {
		return
	}
	if dst.SignalConfidences == nil {
		dst.SignalConfidences = make(map[string]float64)
	}

	prefix := strings.ToLower(strings.TrimSpace(family)) + ":"
	for key := range dst.SignalConfidences {
		if strings.HasPrefix(strings.ToLower(key), prefix) {
			delete(dst.SignalConfidences, key)
		}
	}
	for key, value := range src.SignalConfidences {
		if strings.HasPrefix(strings.ToLower(key), prefix) {
			dst.SignalConfidences[key] = value
		}
	}
}

func copySignalValueFamily(dst *SignalResults, src *SignalResults, family string) {
	if dst == nil || src == nil {
		return
	}
	if dst.SignalValues == nil {
		dst.SignalValues = make(map[string]float64)
	}

	prefix := strings.ToLower(strings.TrimSpace(family)) + ":"
	for key := range dst.SignalValues {
		if strings.HasPrefix(strings.ToLower(key), prefix) {
			delete(dst.SignalValues, key)
		}
	}
	for key, value := range src.SignalValues {
		if strings.HasPrefix(strings.ToLower(key), prefix) {
			dst.SignalValues[key] = value
		}
	}
}

func cloneSignalResults(src *SignalResults) *SignalResults {
	if src == nil {
		return nil
	}

	clone := *src
	clone.MatchedKeywordRules = cloneStrings(src.MatchedKeywordRules)
	clone.MatchedKeywords = cloneStrings(src.MatchedKeywords)
	clone.MatchedEmbeddingRules = cloneStrings(src.MatchedEmbeddingRules)
	clone.MatchedDomainRules = cloneStrings(src.MatchedDomainRules)
	clone.MatchedFactCheckRules = cloneStrings(src.MatchedFactCheckRules)
	clone.MatchedUserFeedbackRules = cloneStrings(src.MatchedUserFeedbackRules)
	clone.MatchedPreferenceRules = cloneStrings(src.MatchedPreferenceRules)
	clone.MatchedLanguageRules = cloneStrings(src.MatchedLanguageRules)
	clone.MatchedContextRules = cloneStrings(src.MatchedContextRules)
	clone.MatchedStructureRules = cloneStrings(src.MatchedStructureRules)
	clone.MatchedComplexityRules = cloneStrings(src.MatchedComplexityRules)
	clone.MatchedModalityRules = cloneStrings(src.MatchedModalityRules)
	clone.MatchedAuthzRules = cloneStrings(src.MatchedAuthzRules)
	clone.MatchedJailbreakRules = cloneStrings(src.MatchedJailbreakRules)
	clone.MatchedPIIRules = cloneStrings(src.MatchedPIIRules)
	clone.MatchedProjectionRules = cloneStrings(src.MatchedProjectionRules)
	clone.PIIEntities = cloneStrings(src.PIIEntities)
	clone.SignalConfidences = maps.Clone(src.SignalConfidences)
	clone.SignalValues = maps.Clone(src.SignalValues)
	clone.ProjectionScores = maps.Clone(src.ProjectionScores)
	clone.ProjectionBoundaryDistances = maps.Clone(src.ProjectionBoundaryDistances)

	if src.Metrics != nil {
		metricsCopy := *src.Metrics
		clone.Metrics = &metricsCopy
	} else {
		clone.Metrics = &SignalMetricsCollection{}
	}

	if len(src.ProjectionPartitionConflicts) > 0 {
		clone.ProjectionPartitionConflicts = make([]ProjectionPartitionConflict, len(src.ProjectionPartitionConflicts))
		for i, conflict := range src.ProjectionPartitionConflicts {
			clone.ProjectionPartitionConflicts[i] = ProjectionPartitionConflict{
				Name:       conflict.Name,
				SignalType: conflict.SignalType,
				Contenders: cloneStrings(conflict.Contenders),
			}
		}
	}

	return &clone
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	return slices.Clone(values)
}
