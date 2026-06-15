package classification

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type projectionMatchAccessor func(*SignalResults) []string

var projectionMatchAccessors = map[string]projectionMatchAccessor{
	config.SignalTypeKeyword:      func(results *SignalResults) []string { return results.MatchedKeywordRules },
	config.SignalTypeEmbedding:    func(results *SignalResults) []string { return results.MatchedEmbeddingRules },
	config.SignalTypeDomain:       func(results *SignalResults) []string { return results.MatchedDomainRules },
	config.SignalTypeFactCheck:    func(results *SignalResults) []string { return results.MatchedFactCheckRules },
	config.SignalTypeUserFeedback: func(results *SignalResults) []string { return results.MatchedUserFeedbackRules },
	config.SignalTypeReask:        func(results *SignalResults) []string { return results.MatchedReaskRules },
	config.SignalTypePreference:   func(results *SignalResults) []string { return results.MatchedPreferenceRules },
	config.SignalTypeLanguage:     func(results *SignalResults) []string { return results.MatchedLanguageRules },
	config.SignalTypeContext:      func(results *SignalResults) []string { return results.MatchedContextRules },
	config.SignalTypeStructure:    func(results *SignalResults) []string { return results.MatchedStructureRules },
	config.SignalTypeComplexity:   func(results *SignalResults) []string { return results.MatchedComplexityRules },
	config.SignalTypeModality:     func(results *SignalResults) []string { return results.MatchedModalityRules },
	config.SignalTypeAuthz:        func(results *SignalResults) []string { return results.MatchedAuthzRules },
	config.SignalTypeJailbreak:    func(results *SignalResults) []string { return results.MatchedJailbreakRules },
	config.SignalTypePII:          func(results *SignalResults) []string { return results.MatchedPIIRules },
	config.SignalTypeKB:           func(results *SignalResults) []string { return results.MatchedKBRules },
	config.SignalTypeConversation: func(results *SignalResults) []string { return results.MatchedConversationRules },
	config.SignalTypeEvent:        func(results *SignalResults) []string { return results.MatchedEventRules },
	config.SignalTypeProjection:   func(results *SignalResults) []string { return results.MatchedProjectionRules },
}

func projectionScoreValue(score config.ProjectionScore, results *SignalResults) float64 {
	total := 0.0
	for _, input := range score.Inputs {
		total += input.Weight * projectionInputValue(input, results)
	}
	return total
}

func projectionInputValue(input config.ProjectionScoreInput, results *SignalResults) float64 {
	normalizedType := strings.ToLower(strings.TrimSpace(input.Type))
	if normalizedType == config.ProjectionInputKBMetric {
		if results.KBMetricValues == nil {
			return 0
		}
		return results.KBMetricValues[kbMetricKey(input.KB, input.Metric)]
	}
	if normalizedType == config.SignalTypeProjection {
		return projectionInputProjectionValue(input, results)
	}
	switch strings.ToLower(strings.TrimSpace(input.ValueSource)) {
	case "raw":
		if results.SignalValues == nil {
			return 0
		}
		return results.SignalValues[strings.ToLower(input.Type)+":"+input.Name]
	case "confidence":
		return projectionInputConfidenceValue(input, results)
	default:
		return projectionInputBinaryValue(input, results)
	}
}

func projectionInputConfidenceValue(input config.ProjectionScoreInput, results *SignalResults) float64 {
	if !projectionInputMatched(input.Type, input.Name, results) {
		return 0
	}
	if results.SignalConfidences == nil {
		return 1.0
	}
	if score, ok := results.SignalConfidences[strings.ToLower(input.Type)+":"+input.Name]; ok && score > 0 {
		return score
	}
	return 1.0
}

func projectionInputBinaryValue(input config.ProjectionScoreInput, results *SignalResults) float64 {
	matchValue := input.Match
	if matchValue == 0 {
		matchValue = 1.0
	}
	if projectionInputMatched(input.Type, input.Name, results) {
		return matchValue
	}
	return input.Miss
}

func projectionInputProjectionValue(input config.ProjectionScoreInput, results *SignalResults) float64 {
	switch strings.ToLower(strings.TrimSpace(input.ValueSource)) {
	case "confidence":
		if results.SignalConfidences == nil {
			return 0
		}
		key := signalConfidenceKey(config.SignalTypeProjection, input.Name)
		return results.SignalConfidences[key]
	default:
		if results.ProjectionScores == nil {
			return 0
		}
		return results.ProjectionScores[input.Name]
	}
}

func kbMetricKey(kbName, metric string) string {
	return strings.ToLower(config.ProjectionInputKBMetric + ":" + kbName + ":" + metric)
}

func projectionInputMatched(signalType string, name string, results *SignalResults) bool {
	matches := projectionMatchSet(results, signalType)
	return slices.Contains(matches, name)
}

func projectionMatchSet(results *SignalResults, signalType string) []string {
	accessor, ok := projectionMatchAccessors[strings.ToLower(strings.TrimSpace(signalType))]
	if !ok {
		return nil
	}
	return accessor(results)
}
