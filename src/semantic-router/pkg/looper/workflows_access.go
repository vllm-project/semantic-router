package looper

import (
	"fmt"
	"strings"

	"github.com/openai/openai-go"
)

func buildWorkflowStepPrompt(originalReq *openai.ChatCompletionNewParams, step workflowPlanStep, previous []workflowStepResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Router Flow step %q (%s).\n\nInstruction:\n%s\n", step.ID, step.Role, step.Prompt)
	visiblePrevious := workflowVisibleStepResults(step, previous)
	if len(visiblePrevious) > 0 {
		fmt.Fprintf(&b, "\nAccessible prior step outputs:\n%s\n", formatWorkflowStepResults(visiblePrevious))
	}
	original := extractOriginalContent(originalReq)
	if original != "" {
		fmt.Fprintf(&b, "\nOriginal user request:\n%s\n", original)
	}
	return b.String()
}

func workflowVisibleStepResults(step workflowPlanStep, previous []workflowStepResult) []workflowStepResult {
	if len(previous) == 0 {
		return nil
	}
	if step.AccessList == nil {
		return previous
	}
	allowed := map[string]bool{}
	for _, id := range step.AccessList {
		allowed[id] = true
	}
	visible := make([]workflowStepResult, 0, len(previous))
	for _, result := range previous {
		if allowed[result.step.ID] {
			visible = append(visible, result)
			continue
		}
		if filtered, ok := workflowStepResultForAllowedAgents(result, allowed); ok {
			visible = append(visible, filtered)
		}
	}
	return visible
}

func workflowStepResultForAllowedAgents(result workflowStepResult, allowed map[string]bool) (workflowStepResult, bool) {
	filtered := workflowStepResult{
		step:   result.step,
		failed: append([]FusionFailedModel(nil), result.failed...),
	}
	for _, resp := range result.responses {
		if resp == nil {
			continue
		}
		agentID := workflowResponseAgentID(result.step, resp)
		if allowed[agentID] {
			filtered.responses = append(filtered.responses, resp)
			if turns := result.toolTrajectories[agentID]; len(turns) > 0 {
				if filtered.toolTrajectories == nil {
					filtered.toolTrajectories = map[string][]workflowAgentToolTurn{}
				}
				filtered.toolTrajectories[agentID] = cloneWorkflowToolTrajectory(turns)
			}
		}
	}
	return filtered, len(filtered.responses) > 0
}

func workflowResponseAgentID(step workflowPlanStep, resp *ModelResponse) string {
	if resp == nil {
		return workflowAgentID(workflowToolPhaseStep, step, "", 0)
	}
	return workflowAgentID(workflowToolPhaseStep, step, resp.Model, workflowResponseModelIndex(step, resp.Model))
}

func workflowResponseModelIndex(step workflowPlanStep, modelName string) int {
	for idx, model := range step.Models {
		if model == modelName {
			return idx
		}
	}
	return 0
}
