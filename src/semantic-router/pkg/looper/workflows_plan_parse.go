package looper

import (
	"encoding/json"
	"fmt"
	"strings"
)

type workflowPlanRawResponse struct {
	Choices []workflowPlanRawChoice `json:"choices"`
}

type workflowPlanRawChoice struct {
	Message map[string]json.RawMessage `json:"message"`
}

func parseWorkflowPlanFromResponse(resp *ModelResponse) (*workflowPlan, error) {
	candidates := workflowPlanCandidates(resp)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("empty planner response")
	}
	var failures []string
	for _, candidate := range candidates {
		plan, err := parseWorkflowPlan(candidate)
		if err == nil {
			return plan, nil
		}
		failures = append(failures, err.Error())
	}
	return nil, fmt.Errorf("%s", strings.Join(failures, "; "))
}

func workflowPlanCandidates(resp *ModelResponse) []string {
	if resp == nil {
		return nil
	}
	var candidates []string
	candidates = appendWorkflowPlanCandidate(candidates, resp.Content)
	candidates = appendWorkflowPlanCandidate(candidates, resp.ReasoningContent)

	var raw workflowPlanRawResponse
	if err := json.Unmarshal(resp.Raw, &raw); err == nil {
		for _, choice := range raw.Choices {
			for _, key := range []string{"content", "reasoning_content", "reasoning"} {
				var value string
				if rawValue, ok := choice.Message[key]; ok && json.Unmarshal(rawValue, &value) == nil {
					candidates = appendWorkflowPlanCandidate(candidates, value)
				}
			}
		}
	}
	return candidates
}

func appendWorkflowPlanCandidate(candidates []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return candidates
	}
	for _, existing := range candidates {
		if existing == value {
			return candidates
		}
	}
	return append(candidates, value)
}

func parseWorkflowPlan(content string) (*workflowPlan, error) {
	candidates := jsonObjectParseCandidates(content)
	var failures []string
	for _, candidate := range candidates {
		var plan workflowPlan
		if err := json.Unmarshal([]byte(candidate), &plan); err == nil {
			return &plan, nil
		} else {
			failures = append(failures, err.Error())
		}
	}
	if len(failures) == 0 {
		return nil, fmt.Errorf("empty planner response")
	}
	return nil, fmt.Errorf("%s", strings.Join(failures, "; "))
}
