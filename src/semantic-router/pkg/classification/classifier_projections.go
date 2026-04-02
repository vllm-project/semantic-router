package classification

import (
	"math"
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
}

func (c *Classifier) applyProjections(results *SignalResults) *SignalResults {
	if results == nil {
		return nil
	}
	if len(c.Config.Projections.Scores) == 0 || len(c.Config.Projections.Mappings) == 0 {
		return results
	}

	if results.ProjectionScores == nil {
		results.ProjectionScores = make(map[string]float64)
	}
	if results.SignalConfidences == nil {
		results.SignalConfidences = make(map[string]float64)
	}

	for _, score := range c.Config.Projections.Scores {
		results.ProjectionScores[score.Name] = projectionScoreValue(score, results)
	}

	for _, mapping := range c.Config.Projections.Mappings {
		scoreValue, ok := results.ProjectionScores[mapping.Source]
		if !ok {
			continue
		}
		output, matched := matchProjectionOutput(mapping, scoreValue)
		if !matched {
			continue
		}
		results.MatchedProjectionRules = append(results.MatchedProjectionRules, output.Name)
		results.SignalConfidences[signalConfidenceKey(config.SignalTypeProjection, output.Name)] = projectionOutputConfidence(mapping, output, scoreValue)
	}

	return results
}

func projectionScoreValue(score config.ProjectionScore, results *SignalResults) float64 {
	total := 0.0
	for _, input := range score.Inputs {
		total += input.Weight * projectionInputValue(input, results)
	}
	return total
}

func projectionInputValue(input config.ProjectionScoreInput, results *SignalResults) float64 {
	if strings.EqualFold(strings.TrimSpace(input.Type), config.ProjectionInputKBMetric) {
		if results.KBMetricValues == nil {
			return 0
		}
		return results.KBMetricValues[kbMetricKey(input.KB, input.Metric)]
	}
	switch strings.ToLower(strings.TrimSpace(input.ValueSource)) {
	case "confidence":
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
	default:
		matchValue := input.Match
		missValue := input.Miss
		if matchValue == 0 {
			matchValue = 1.0
		}
		if projectionInputMatched(input.Type, input.Name, results) {
			return matchValue
		}
		return missValue
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

func matchProjectionOutput(
	mapping config.ProjectionMapping,
	scoreValue float64,
) (config.ProjectionMappingOutput, bool) {
	for _, output := range mapping.Outputs {
		if projectionOutputMatches(output, scoreValue) {
			return output, true
		}
	}
	return config.ProjectionMappingOutput{}, false
}

func projectionOutputMatches(output config.ProjectionMappingOutput, scoreValue float64) bool {
	if output.GT != nil && !(scoreValue > *output.GT) {
		return false
	}
	if output.GTE != nil && !(scoreValue >= *output.GTE) {
		return false
	}
	if output.LT != nil && !(scoreValue < *output.LT) {
		return false
	}
	if output.LTE != nil && !(scoreValue <= *output.LTE) {
		return false
	}
	return true
}

func projectionOutputConfidence(
	mapping config.ProjectionMapping,
	output config.ProjectionMappingOutput,
	scoreValue float64,
) float64 {
	slope := 12.0
	if mapping.Calibration != nil && mapping.Calibration.Slope > 0 {
		slope = mapping.Calibration.Slope
	}

	distance := projectionBoundaryDistance(output, scoreValue)
	return 1.0 / (1.0 + math.Exp(-slope*distance))
}

func projectionBoundaryDistance(output config.ProjectionMappingOutput, scoreValue float64) float64 {
	distances := make([]float64, 0, 4)
	if output.GT != nil {
		distances = append(distances, math.Abs(scoreValue-*output.GT))
	}
	if output.GTE != nil {
		distances = append(distances, math.Abs(scoreValue-*output.GTE))
	}
	if output.LT != nil {
		distances = append(distances, math.Abs(*output.LT-scoreValue))
	}
	if output.LTE != nil {
		distances = append(distances, math.Abs(*output.LTE-scoreValue))
	}
	if len(distances) == 0 {
		return 1.0
	}

	best := distances[0]
	for _, distance := range distances[1:] {
		if distance < best {
			best = distance
		}
	}
	return best
}
