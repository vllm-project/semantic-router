package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func dynamicRetrievalConfigMap(cfg *config.DynamicRetrievalConfig) map[string]interface{} {
	if cfg == nil {
		return nil
	}
	values := map[string]interface{}{}
	if cfg.Enabled {
		values["enabled"] = true
	}
	if cfg.Strategy != "" {
		values["strategy"] = cfg.Strategy
	}
	if cfg.HistoryWindow != 0 {
		values["history_window"] = cfg.HistoryWindow
	}
	if cfg.Weights != nil {
		values["weights"] = dynamicRetrievalWeightsMap(cfg.Weights)
	}
	if cfg.MinHistoryConfidence != 0 {
		values["min_history_confidence"] = cfg.MinHistoryConfidence
	}
	if cfg.FallbackOnLowConfidence {
		values["fallback_on_low_confidence"] = true
	}
	return values
}

func dynamicRetrievalWeightsMap(weights *config.DynamicRetrievalWeights) map[string]interface{} {
	if weights == nil {
		return nil
	}
	values := map[string]interface{}{}
	if weights.Semantic != 0 {
		values["semantic"] = weights.Semantic
	}
	if weights.History != 0 {
		values["history"] = weights.History
	}
	if weights.DecisionPrior != 0 {
		values["decision_prior"] = weights.DecisionPrior
	}
	if weights.RepetitionPenalty != 0 {
		values["repetition_penalty"] = weights.RepetitionPenalty
	}
	return values
}

func dynamicRetrievalObjectValue(cfg *config.DynamicRetrievalConfig) ObjectValue {
	fields := map[string]Value{}
	if cfg == nil {
		return ObjectValue{Fields: fields}
	}
	if cfg.Enabled {
		fields["enabled"] = BoolValue{V: true}
	}
	if cfg.Strategy != "" {
		fields["strategy"] = StringValue{V: cfg.Strategy}
	}
	if cfg.HistoryWindow != 0 {
		fields["history_window"] = IntValue{V: cfg.HistoryWindow}
	}
	if cfg.Weights != nil {
		fields["weights"] = dynamicRetrievalWeightsObjectValue(cfg.Weights)
	}
	if cfg.MinHistoryConfidence != 0 {
		fields["min_history_confidence"] = FloatValue{V: cfg.MinHistoryConfidence}
	}
	if cfg.FallbackOnLowConfidence {
		fields["fallback_on_low_confidence"] = BoolValue{V: true}
	}
	return ObjectValue{Fields: fields}
}

func dynamicRetrievalWeightsObjectValue(weights *config.DynamicRetrievalWeights) ObjectValue {
	fields := map[string]Value{}
	if weights == nil {
		return ObjectValue{Fields: fields}
	}
	if weights.Semantic != 0 {
		fields["semantic"] = FloatValue{V: weights.Semantic}
	}
	if weights.History != 0 {
		fields["history"] = FloatValue{V: weights.History}
	}
	if weights.DecisionPrior != 0 {
		fields["decision_prior"] = FloatValue{V: weights.DecisionPrior}
	}
	if weights.RepetitionPenalty != 0 {
		fields["repetition_penalty"] = FloatValue{V: weights.RepetitionPenalty}
	}
	return ObjectValue{Fields: fields}
}

func structureFeatureValue(feature config.StructureFeature) ObjectValue {
	fields := map[string]Value{
		"type":   StringValue{V: feature.Type},
		"source": structureSourceValue(feature.Source),
	}
	return ObjectValue{Fields: fields}
}

func structureSourceValue(source config.StructureSource) ObjectValue {
	fields := map[string]Value{
		"type": StringValue{V: source.Type},
	}
	if source.Pattern != "" {
		fields["pattern"] = StringValue{V: source.Pattern}
	}
	if len(source.Keywords) > 0 {
		fields["keywords"] = stringsToArray(source.Keywords)
	}
	if source.CaseSensitive {
		fields["case_sensitive"] = BoolValue{V: true}
	}
	if len(source.Sequences) > 0 {
		items := make([]Value, 0, len(source.Sequences))
		for _, sequence := range source.Sequences {
			items = append(items, stringsToArray(sequence))
		}
		fields["sequences"] = ArrayValue{Items: items}
	}
	return ObjectValue{Fields: fields}
}

func structurePredicateValue(predicate *config.NumericPredicate) ObjectValue {
	fields := make(map[string]Value)
	if predicate.GT != nil {
		fields["gt"] = FloatValue{V: *predicate.GT}
	}
	if predicate.GTE != nil {
		fields["gte"] = FloatValue{V: *predicate.GTE}
	}
	if predicate.LT != nil {
		fields["lt"] = FloatValue{V: *predicate.LT}
	}
	if predicate.LTE != nil {
		fields["lte"] = FloatValue{V: *predicate.LTE}
	}
	return ObjectValue{Fields: fields}
}

func structureFeatureToMap(feature config.StructureFeature) map[string]interface{} {
	values := map[string]interface{}{
		"type":   feature.Type,
		"source": structureSourceToMap(feature.Source),
	}
	return values
}

func structureSourceToMap(source config.StructureSource) map[string]interface{} {
	values := map[string]interface{}{
		"type": source.Type,
	}
	if source.Pattern != "" {
		values["pattern"] = source.Pattern
	}
	if len(source.Keywords) > 0 {
		values["keywords"] = source.Keywords
	}
	if source.CaseSensitive {
		values["case_sensitive"] = true
	}
	if len(source.Sequences) > 0 {
		values["sequences"] = source.Sequences
	}
	return values
}

func conversationFeatureValue(feature config.ConversationFeature) ObjectValue {
	fields := map[string]Value{
		"type":   StringValue{V: feature.Type},
		"source": conversationSourceValue(feature.Source),
	}
	return ObjectValue{Fields: fields}
}

func conversationSourceValue(source config.ConversationSource) ObjectValue {
	fields := map[string]Value{
		"type": StringValue{V: source.Type},
	}
	if source.Role != "" {
		fields["role"] = StringValue{V: source.Role}
	}
	return ObjectValue{Fields: fields}
}

func conversationFeatureToMap(feature config.ConversationFeature) map[string]interface{} {
	return map[string]interface{}{
		"type":   feature.Type,
		"source": conversationSourceToMap(feature.Source),
	}
}

func conversationSourceToMap(source config.ConversationSource) map[string]interface{} {
	values := map[string]interface{}{
		"type": source.Type,
	}
	if source.Role != "" {
		values["role"] = source.Role
	}
	return values
}

func structurePredicateToMap(predicate *config.NumericPredicate) map[string]interface{} {
	values := make(map[string]interface{})
	if predicate == nil {
		return values
	}
	if predicate.GT != nil {
		values["gt"] = *predicate.GT
	}
	if predicate.GTE != nil {
		values["gte"] = *predicate.GTE
	}
	if predicate.LT != nil {
		values["lt"] = *predicate.LT
	}
	if predicate.LTE != nil {
		values["lte"] = *predicate.LTE
	}
	return values
}

func formatProjectionScoreInputs(inputs []config.ProjectionScoreInput) string {
	parts := make([]string, 0, len(inputs))
	for _, input := range inputs {
		fields := []string{
			fmt.Sprintf("type: %q", input.Type),
			fmt.Sprintf("weight: %g", input.Weight),
		}
		if input.Name != "" {
			fields = append(fields, fmt.Sprintf("name: %q", input.Name))
		}
		if input.KB != "" {
			fields = append(fields, fmt.Sprintf("kb: %q", input.KB))
		}
		if input.Metric != "" {
			fields = append(fields, fmt.Sprintf("metric: %q", input.Metric))
		}
		if input.ValueSource != "" {
			fields = append(fields, fmt.Sprintf("value_source: %q", input.ValueSource))
		}
		if input.Match != 0 {
			fields = append(fields, fmt.Sprintf("match: %g", input.Match))
		}
		if input.Miss != 0 {
			fields = append(fields, fmt.Sprintf("miss: %g", input.Miss))
		}
		parts = append(parts, "{ "+strings.Join(fields, ", ")+" }")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatProjectionMappingCalibration(calibration *config.ProjectionMappingCalibration) string {
	if calibration == nil {
		return "{}"
	}
	fields := make([]string, 0, 2)
	if calibration.Method != "" {
		fields = append(fields, fmt.Sprintf("method: %q", calibration.Method))
	}
	if calibration.Slope != 0 {
		fields = append(fields, fmt.Sprintf("slope: %g", calibration.Slope))
	}
	return "{ " + strings.Join(fields, ", ") + " }"
}

func formatProjectionMappingOutputs(outputs []config.ProjectionMappingOutput) string {
	parts := make([]string, 0, len(outputs))
	for _, output := range outputs {
		fields := []string{fmt.Sprintf("name: %q", output.Name)}
		if output.GT != nil {
			fields = append(fields, fmt.Sprintf("gt: %g", *output.GT))
		}
		if output.GTE != nil {
			fields = append(fields, fmt.Sprintf("gte: %g", *output.GTE))
		}
		if output.LT != nil {
			fields = append(fields, fmt.Sprintf("lt: %g", *output.LT))
		}
		if output.LTE != nil {
			fields = append(fields, fmt.Sprintf("lte: %g", *output.LTE))
		}
		parts = append(parts, "{ "+strings.Join(fields, ", ")+" }")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
