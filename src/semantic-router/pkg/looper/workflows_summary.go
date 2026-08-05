package looper

import "strings"

func summarizeWorkflowExecution(
	cfg workflowsExecutionConfig,
	plannerResp *ModelResponse,
	stepResults []workflowStepResult,
	finalResp *ModelResponse,
) workflowExecutionSummary {
	summary := workflowExecutionSummary{
		usage:      SumUsage(plannerResp, finalResp),
		modelsUsed: make([]string, 0, 2+len(stepResults)),
		iterations: 1,
	}
	summary.modelsUsed = appendUniqueWorkflowModel(summary.modelsUsed, cfg.PlannerModel)
	if plannerResp != nil {
		summary.iterations++
	}
	for _, result := range stepResults {
		summary.usage = summary.usage.Add(result.responses...)
		summary.iterations += len(result.responses) + len(result.failed)
		summary.failed = append(summary.failed, result.failed...)
		for _, resp := range result.responses {
			summary.modelsUsed = appendUniqueWorkflowModel(summary.modelsUsed, resp.Model)
		}
	}
	summary.modelsUsed = appendUniqueWorkflowModel(summary.modelsUsed, finalResp.Model)
	return summary
}

func appendUniqueWorkflowModel(models []string, model string) []string {
	if strings.TrimSpace(model) == "" {
		return models
	}
	for _, existing := range models {
		if existing == model {
			return models
		}
	}
	return append(models, model)
}
