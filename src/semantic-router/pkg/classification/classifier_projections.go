package classification

import (
	"math"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

type projectionMatchAccessor func(*SignalResults) []string

var projectionMatchAccessors = map[string]projectionMatchAccessor{
	config.SignalTypeKeyword:       func(results *SignalResults) []string { return results.MatchedKeywordRules },
	config.SignalTypeEmbedding:     func(results *SignalResults) []string { return results.MatchedEmbeddingRules },
	config.SignalTypeDomain:        func(results *SignalResults) []string { return results.MatchedDomainRules },
	config.SignalTypeFactCheck:     func(results *SignalResults) []string { return results.MatchedFactCheckRules },
	config.SignalTypeUserFeedback:  func(results *SignalResults) []string { return results.MatchedUserFeedbackRules },
	config.SignalTypeReask:         func(results *SignalResults) []string { return results.MatchedReaskRules },
	config.SignalTypePreference:    func(results *SignalResults) []string { return results.MatchedPreferenceRules },
	config.SignalTypeLanguage:      func(results *SignalResults) []string { return results.MatchedLanguageRules },
	config.SignalTypeContext:       func(results *SignalResults) []string { return results.MatchedContextRules },
	config.SignalTypeStructure:     func(results *SignalResults) []string { return results.MatchedStructureRules },
	config.SignalTypeComplexity:    func(results *SignalResults) []string { return results.MatchedComplexityRules },
	config.SignalTypeModality:      func(results *SignalResults) []string { return results.MatchedModalityRules },
	config.SignalTypeAuthz:         func(results *SignalResults) []string { return results.MatchedAuthzRules },
	config.SignalTypeJailbreak:     func(results *SignalResults) []string { return results.MatchedJailbreakRules },
	config.SignalTypePII:           func(results *SignalResults) []string { return results.MatchedPIIRules },
	config.SignalTypeKB:            func(results *SignalResults) []string { return results.MatchedKBRules },
	config.SignalTypeConversation:  func(results *SignalResults) []string { return results.MatchedConversationRules },
	config.SignalTypeSessionMetric: func(results *SignalResults) []string { return results.MatchedSessionMetricRules },
	config.SignalTypeProjection:    func(results *SignalResults) []string { return results.MatchedProjectionRules },
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

	orderedScores := topologicalScoreOrder(c.Config.Projections.Scores, c.Config.Projections.Mappings)

	mappingBySource := make(map[string][]config.ProjectionMapping, len(c.Config.Projections.Mappings))
	for _, mapping := range c.Config.Projections.Mappings {
		mappingBySource[mapping.Source] = append(mappingBySource[mapping.Source], mapping)
	}

	for _, score := range orderedScores {
		results.ProjectionScores[score.Name] = projectionScoreValue(score, results)

		for _, mapping := range mappingBySource[score.Name] {
			scoreValue := results.ProjectionScores[mapping.Source]
			output, matched := matchProjectionOutput(mapping, scoreValue)
			if !matched {
				continue
			}
			results.MatchedProjectionRules = append(results.MatchedProjectionRules, output.Name)
			results.SignalConfidences[signalConfidenceKey(config.SignalTypeProjection, output.Name)] = projectionOutputConfidence(mapping, output, scoreValue)
		}
	}

	results.ProjectionTrace = mergeProjectionTrace(results, c.Config.Projections)
	return results
}

func hasProjectionDependency(scores []config.ProjectionScore) bool {
	for _, s := range scores {
		for _, inp := range s.Inputs {
			if strings.EqualFold(strings.TrimSpace(inp.Type), config.SignalTypeProjection) {
				return true
			}
		}
	}
	return false
}

func buildScoreAdjacency(scores []config.ProjectionScore, outputToSource map[string]string) map[string][]string {
	adj := make(map[string][]string, len(scores))
	for _, s := range scores {
		for _, inp := range s.Inputs {
			if !strings.EqualFold(strings.TrimSpace(inp.Type), config.SignalTypeProjection) {
				continue
			}
			vs := strings.ToLower(strings.TrimSpace(inp.ValueSource))
			if vs == "confidence" {
				if src, ok := outputToSource[inp.Name]; ok {
					adj[s.Name] = append(adj[s.Name], src)
				}
			} else {
				adj[s.Name] = append(adj[s.Name], inp.Name)
			}
		}
	}
	return adj
}

func topologicalScoreOrder(scores []config.ProjectionScore, mappings []config.ProjectionMapping) []config.ProjectionScore {
	if !hasProjectionDependency(scores) {
		return scores
	}

	byName := make(map[string]config.ProjectionScore, len(scores))
	for _, s := range scores {
		byName[s.Name] = s
	}

	outputToSource := make(map[string]string)
	for _, m := range mappings {
		for _, out := range m.Outputs {
			if out.Name != "" {
				outputToSource[out.Name] = m.Source
			}
		}
	}

	adj := buildScoreAdjacency(scores, outputToSource)
	state := make(map[string]int, len(scores))
	ordered := make([]config.ProjectionScore, 0, len(scores))

	var visit func(name string)
	visit = func(name string) {
		if state[name] != 0 {
			return
		}
		state[name] = 1
		for _, dep := range adj[name] {
			if _, ok := byName[dep]; ok {
				visit(dep)
			}
		}
		if s, ok := byName[name]; ok {
			ordered = append(ordered, s)
		}
	}

	for _, s := range scores {
		visit(s.Name)
	}

	return ordered
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

func mergeProjectionTrace(results *SignalResults, p config.Projections) *projectiontrace.Trace {
	var existingPartitions []projectiontrace.PartitionResolution
	if results.ProjectionTrace != nil && len(results.ProjectionTrace.Partitions) > 0 {
		existingPartitions = append([]projectiontrace.PartitionResolution(nil), results.ProjectionTrace.Partitions...)
	}
	tr := &projectiontrace.Trace{SchemaVersion: projectiontrace.SchemaVersion}
	tr.Partitions = existingPartitions
	for _, score := range p.Scores {
		sb := projectiontrace.ScoreBreakdown{Name: score.Name}
		var sum float64
		for _, input := range score.Inputs {
			v := projectionInputValue(input, results)
			contrib := input.Weight * v
			sum += contrib
			sb.Inputs = append(sb.Inputs, projectiontrace.ScoreInputPart{
				Type:         input.Type,
				Name:         input.Name,
				KB:           input.KB,
				Metric:       input.Metric,
				Weight:       input.Weight,
				Value:        v,
				Contribution: contrib,
			})
		}
		sb.Total = sum
		tr.Scores = append(tr.Scores, sb)
	}
	for _, mapping := range p.Mappings {
		scoreValue, ok := results.ProjectionScores[mapping.Source]
		if !ok {
			continue
		}
		md := projectiontrace.MappingDecision{
			MappingName: mapping.Name,
			SourceScore: mapping.Source,
			ScoreValue:  scoreValue,
		}
		for _, output := range mapping.Outputs {
			matched := projectionOutputMatches(output, scoreValue)
			d := projectionBoundaryDistance(output, scoreValue)
			md.Outputs = append(md.Outputs, projectiontrace.OutputEvalStep{
				Name:             output.Name,
				Matched:          matched,
				BoundaryDistance: d,
			})
			if matched && md.SelectedOutput == "" {
				out := output
				md.SelectedOutput = out.Name
				md.Confidence = projectionOutputConfidence(mapping, out, scoreValue)
				md.BoundaryDistance = d
			}
		}
		tr.Mappings = append(tr.Mappings, md)
	}
	return tr
}
