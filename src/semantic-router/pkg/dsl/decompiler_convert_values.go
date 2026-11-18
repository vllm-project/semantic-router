package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func numericPredicateObject(predicate *config.NumericPredicate) ObjectValue {
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

func numericPredicateToMap(predicate *config.NumericPredicate) map[string]interface{} {
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
		if input.ValueSource != "" {
			fields = append(fields, fmt.Sprintf("value_source: %q", input.ValueSource))
		}
		if input.Match != nil {
			fields = append(fields, fmt.Sprintf("match: %g", *input.Match))
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
