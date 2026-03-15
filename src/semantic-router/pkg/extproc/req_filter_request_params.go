package extproc

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

var allowedTopLevelFields = map[string]bool{
	"model":         true,
	"messages":       true,
	"frequency_penalty": true,
	"logit_bias":    true,
	"max_tokens":    true,
	"n":             true,
	"presence_penalty": true,
	"response_format": true,
	"seed":          true,
	"stop":          true,
	"stream":         true,
	"temperature":    true,
	"top_p":          true,
	"tools":          true,
	"tool_choice":    true,
	"user":           true,
	"reasoning_effort": true,
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

	modified := false

	for _, blockedParam := range paramsConfig.BlockedParams {
		if _, exists := body[blockedParam]; exists {
			delete(body, blockedParam)
			modified = true
			logging.Debugf("Blocked parameter '%s' removed for decision '%s'", blockedParam, decision.Name)
			metrics.RecordBlockedParam(decision.Name, blockedParam)
		}
	}

	if paramsConfig.MaxTokensLimit != nil {
		if maxTokens, exists := body["max_tokens"]; exists {
			if mt, ok := maxTokens.(float64); ok {
				if mt > float64(*paramsConfig.MaxTokensLimit) {
					body["max_tokens"] = float64(*paramsConfig.MaxTokensLimit)
					modified = true
					logging.Debugf("Capped max_tokens from %.0f to %d for decision '%s'", mt, *paramsConfig.MaxTokensLimit, decision.Name)
					metrics.RecordMaxTokensCapped(decision.Name, int(mt), *paramsConfig.MaxTokensLimit)
				}
			} else if mt, ok := maxTokens.(int); ok {
				if mt > *paramsConfig.MaxTokensLimit {
					body["max_tokens"] = *paramsConfig.MaxTokensLimit
					modified = true
					logging.Debugf("Capped max_tokens from %d to %d for decision '%s'", mt, *paramsConfig.MaxTokensLimit, decision.Name)
					metrics.RecordMaxTokensCapped(decision.Name, mt, *paramsConfig.MaxTokensLimit)
				}
			}
		}
	}

	if paramsConfig.MaxN != nil {
		if n, exists := body["n"]; exists {
			if nVal, ok := n.(float64); ok {
				if nVal > float64(*paramsConfig.MaxN) {
					body["n"] = float64(*paramsConfig.MaxN)
					modified = true
					logging.Debugf("Capped n from %.0f to %d for decision '%s'", nVal, *paramsConfig.MaxN, decision.Name)
					metrics.RecordMaxNCapped(decision.Name, int(nVal), *paramsConfig.MaxN)
				}
			} else if nVal, ok := n.(int); ok {
				if nVal > *paramsConfig.MaxN {
					body["n"] = *paramsConfig.MaxN
					modified = true
					logging.Debugf("Capped n from %d to %d for decision '%s'", nVal, *paramsConfig.MaxN, decision.Name)
					metrics.RecordMaxNCapped(decision.Name, nVal, *paramsConfig.MaxN)
				}
			}
		}
	}

	if paramsConfig.StripUnknown {
		for key := range body {
			if !allowedTopLevelFields[key] {
				delete(body, key)
				modified = true
				logging.Debugf("Removed unknown field '%s' for decision '%s'", key, decision.Name)
				metrics.RecordUnknownFieldStripped(decision.Name, key)
			}
		}
	}

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
