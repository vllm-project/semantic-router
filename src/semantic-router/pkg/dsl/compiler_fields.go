package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func getStringField(fields map[string]Value, key string) (string, bool) {
	if v, ok := fields[key]; ok {
		if sv, ok := v.(StringValue); ok {
			return sv.V, true
		}
	}
	return "", false
}

func getIntField(fields map[string]Value, key string) (int, bool) {
	if v, ok := fields[key]; ok {
		if iv, ok := v.(IntValue); ok {
			return iv.V, true
		}
		// Also accept float as int
		if fv, ok := v.(FloatValue); ok {
			return int(fv.V), true
		}
	}
	return 0, false
}

func getFloat32Field(fields map[string]Value, key string) (float32, bool) {
	if v, ok := fields[key]; ok {
		if fv, ok := v.(FloatValue); ok {
			return float32(fv.V), true
		}
		if iv, ok := v.(IntValue); ok {
			return float32(iv.V), true
		}
	}
	return 0, false
}

func getFloat64Field(fields map[string]Value, key string) (float64, bool) {
	if v, ok := fields[key]; ok {
		if fv, ok := v.(FloatValue); ok {
			return fv.V, true
		}
		if iv, ok := v.(IntValue); ok {
			return float64(iv.V), true
		}
	}
	return 0, false
}

func getBoolField(fields map[string]Value, key string) (bool, bool) {
	if v, ok := fields[key]; ok {
		if bv, ok := v.(BoolValue); ok {
			return bv.V, true
		}
	}
	return false, false
}

func getStringArrayField(fields map[string]Value, key string) ([]string, bool) {
	if v, ok := fields[key]; ok {
		if av, ok := v.(ArrayValue); ok {
			var result []string
			for _, item := range av.Items {
				if sv, ok := item.(StringValue); ok {
					result = append(result, sv.V)
				}
			}
			return result, true
		}
	}
	return nil, false
}

func getModelScoresField(fields map[string]Value, key string) ([]config.ModelScore, bool) {
	raw, ok := fields[key]
	if !ok {
		return nil, false
	}
	items, ok := raw.(ArrayValue)
	if !ok {
		return nil, false
	}
	result := make([]config.ModelScore, 0, len(items.Items))
	for _, item := range items.Items {
		obj, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		score := config.ModelScore{}
		if model, ok := getStringField(obj.Fields, "model"); ok {
			score.Model = model
		}
		if value, ok := getFloat64Field(obj.Fields, "score"); ok {
			score.Score = value
		}
		if useReasoning, ok := getBoolField(obj.Fields, "use_reasoning"); ok {
			value := useReasoning
			score.UseReasoning = &value
		}
		result = append(result, score)
	}
	return result, true
}

func compileDomainSuggestionSuffix(value string) string {
	suggestion := config.SuggestSupportedRoutingDomainName(value)
	if suggestion == "" || suggestion == value {
		return ""
	}
	return fmt.Sprintf("; did you mean %q?", suggestion)
}

func (c *Compiler) validateSoftmaxDomainProjectionPartition(
	partition *ProjectionPartitionDecl,
) {
	if partition.Semantics != "softmax_exclusive" {
		return
	}
	for _, member := range partition.Members {
		signal := c.findSignalDeclByName(member)
		if signal == nil || signal.SignalType != "domain" {
			continue
		}
		mmluCategories := getMMLUCategories(signal)
		if len(mmluCategories) == 0 {
			if config.IsSupportedRoutingDomainName(signal.Name) {
				continue
			}
			c.addError(
				partition.Pos,
				"PROJECTION partition %q member %q must use a supported routing domain name (%s) or declare mmlu_categories explicitly%s",
				partition.Name,
				member,
				strings.Join(config.SupportedRoutingDomainNames(), ", "),
				compileDomainSuggestionSuffix(member),
			)
			continue
		}
		for _, value := range mmluCategories {
			if config.IsSupportedRoutingDomainName(value) {
				continue
			}
			c.addError(
				partition.Pos,
				"PROJECTION partition %q member %q has unsupported mmlu_categories value %q (supported: %s)%s",
				partition.Name,
				member,
				value,
				strings.Join(config.SupportedRoutingDomainNames(), ", "),
				compileDomainSuggestionSuffix(value),
			)
		}
	}
}

func (c *Compiler) findSignalDeclByName(name string) *SignalDecl {
	for _, signal := range c.prog.Signals {
		if signal.Name == name {
			return signal
		}
	}
	return nil
}

func getIntArrayField(fields map[string]Value, key string) ([]int, bool) {
	if v, ok := fields[key]; ok {
		if av, ok := v.(ArrayValue); ok {
			var result []int
			for _, item := range av.Items {
				if iv, ok := item.(IntValue); ok {
					result = append(result, iv.V)
				}
			}
			return result, true
		}
	}
	return nil, false
}

func fieldsToMap(fields map[string]Value) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range fields {
		result[k] = valueToInterface(v)
	}
	return result
}

func valueToInterface(v Value) interface{} {
	switch val := v.(type) {
	case StringValue:
		return val.V
	case IntValue:
		return val.V
	case FloatValue:
		return val.V
	case BoolValue:
		return val.V
	case ArrayValue:
		var arr []interface{}
		for _, item := range val.Items {
			arr = append(arr, valueToInterface(item))
		}
		return arr
	case ObjectValue:
		return fieldsToMap(val.Fields)
	default:
		return nil
	}
}
