package extproc

import (
	"encoding/json"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// allowedTopLevelFields mirrors common OpenAI Chat Completions top-level keys. When strip_unknown
// is enabled, fields not listed here are removed; extend this map when the router adds support for
// new documented request fields.
var allowedTopLevelFields = map[string]bool{
	"model":             true,
	"messages":          true,
	"frequency_penalty": true,
	"logit_bias":        true,
	"logprobs":          true,
	"max_tokens":        true,
	"n":                 true,
	"presence_penalty":  true,
	"response_format":   true,
	"seed":              true,
	"stop":              true,
	"stream":            true,
	"temperature":       true,
	"top_logprobs":      true,
	"top_p":             true,
	"tools":             true,
	"tool_choice":       true,
	"user":              true,
	"reasoning_effort":  true,
}

func (r *OpenAIRouter) buildRequestParamsMutations(
	decision *config.Decision,
	requestBody []byte,
) ([]byte, error) {
	if decision == nil {
		return requestBody, nil
	}

	paramsConfig := decision.GetRequestParamsConfig()
	if paramsConfig == nil {
		return requestBody, nil
	}

	logging.Debugf("Applying request params validation for decision %s", decision.Name)

	var body map[string]interface{}
	if err := json.Unmarshal(requestBody, &body); err != nil {
		logging.Warnf("Failed to parse request body: %v", err)
		return requestBody, nil
	}

	modified := applyBlockedParams(body, paramsConfig.BlockedParams, decision.Name)
	modified = capIntField(body, "max_tokens", paramsConfig.MaxTokensLimit, decision.Name, metrics.RecordMaxTokensCapped) || modified
	modified = capIntField(body, "n", paramsConfig.MaxN, decision.Name, metrics.RecordMaxNCapped) || modified
	modified = stripUnknownFields(body, paramsConfig.StripUnknown, decision.Name) || modified

	if !modified {
		return requestBody, nil
	}

	modifiedBody, err := json.Marshal(body)
	if err != nil {
		logging.Warnf("Failed to serialize modified request body: %v", err)
		return requestBody, err
	}

	logging.Debugf("Request params validated and modified for decision '%s'", decision.Name)
	return modifiedBody, nil
}

// applyBlockedParams removes blocked parameters from the request body.
func applyBlockedParams(body map[string]interface{}, blockedParams []string, decisionName string) bool {
	modified := false
	for _, param := range blockedParams {
		if _, exists := body[param]; exists {
			delete(body, param)
			modified = true
			logging.Debugf("Blocked parameter '%s' removed for decision '%s'", param, decisionName)
			metrics.RecordBlockedParam(decisionName, param)
		}
	}
	return modified
}

// capIntField caps a numeric field in the body to the given limit.
func capIntField(
	body map[string]interface{},
	fieldName string,
	limit *int,
	decisionName string,
	recordMetric func(string),
) bool {
	if limit == nil {
		return false
	}
	val, exists := body[fieldName]
	if !exists {
		return false
	}
	original, ok := toInt(val)
	if !ok {
		if fv, isFloat := val.(float64); isFloat && fv != math.Trunc(fv) {
			delete(body, fieldName)
			logging.Warnf("Removed non-integer %s (fractional JSON number) for decision '%s'", fieldName, decisionName)
			return true
		}
		return false
	}
	if original <= *limit {
		return false
	}
	body[fieldName] = *limit
	logging.Debugf("Capped %s from %d to %d for decision '%s'", fieldName, original, *limit, decisionName)
	recordMetric(decisionName)
	return true
}

// toInt converts a JSON numeric value to an int. float64 values must be whole numbers
// (no fractional part); otherwise returns false so callers cannot bypass caps via truncation.
func toInt(val interface{}) (int, bool) {
	switch v := val.(type) {
	case float64:
		if v != math.Trunc(v) {
			return 0, false
		}
		return int(v), true
	case int:
		return v, true
	default:
		return 0, false
	}
}

// stripUnknownFields removes fields not in the OpenAI spec allowlist.
func stripUnknownFields(body map[string]interface{}, strip bool, decisionName string) bool {
	if !strip {
		return false
	}
	modified := false
	for key := range body {
		if !allowedTopLevelFields[key] {
			delete(body, key)
			modified = true
			logging.Debugf("Removed unknown field '%s' for decision '%s'", key, decisionName)
			metrics.RecordUnknownFieldStripped(decisionName, key)
		}
	}
	return modified
}
