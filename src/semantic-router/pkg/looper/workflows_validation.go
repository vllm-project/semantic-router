package looper

import (
	"fmt"
	"strings"
)

func validateWorkflowPlan(plan *workflowPlan, workerModels []string, cfg workflowsExecutionConfig) error {
	if plan == nil || len(plan.Steps) == 0 {
		return fmt.Errorf("workflows plan must include at least one step")
	}
	if len(plan.Steps) > cfg.MaxSteps {
		return fmt.Errorf("workflows plan has %d steps, exceeding max_steps=%d", len(plan.Steps), cfg.MaxSteps)
	}
	allowed := allowedWorkflowModels(workerModels)
	previousAccessIDs := map[string]bool{}
	for i := range plan.Steps {
		if err := validateWorkflowPlanStep(&plan.Steps[i], i, allowed, previousAccessIDs, cfg); err != nil {
			return err
		}
		registerWorkflowAccessIDs(previousAccessIDs, plan.Steps[i])
	}
	return validateWorkflowFinal(plan.Final, allowed)
}

func allowedWorkflowModels(workerModels []string) map[string]bool {
	allowed := map[string]bool{}
	for _, model := range workerModels {
		allowed[model] = true
	}
	return allowed
}

func validateWorkflowPlanStep(
	step *workflowPlanStep,
	index int,
	allowed map[string]bool,
	previousAccessIDs map[string]bool,
	cfg workflowsExecutionConfig,
) error {
	if strings.TrimSpace(step.ID) == "" {
		step.ID = fmt.Sprintf("step-%d", index+1)
	}
	if strings.TrimSpace(step.Role) == "" {
		step.Role = "worker"
	}
	step.Models = normalizeModelNames(step.Models)
	if len(step.Models) == 0 {
		return fmt.Errorf("workflows plan step %q must include at least one model", step.ID)
	}
	if len(step.Models) > cfg.MaxParallel {
		return fmt.Errorf("workflows plan step %q has %d models, exceeding max_parallel=%d", step.ID, len(step.Models), cfg.MaxParallel)
	}
	for _, model := range step.Models {
		if !allowed[model] {
			return fmt.Errorf("workflows plan step %q references model %q outside decision modelRefs", step.ID, model)
		}
	}
	if strings.TrimSpace(step.Prompt) == "" {
		step.Prompt = "Work on the original user request and provide a concise, useful result."
	}
	if step.AccessList != nil {
		step.AccessList = normalizeWorkflowAccessList(step.AccessList)
		for _, id := range step.AccessList {
			if !previousAccessIDs[id] {
				return fmt.Errorf("workflows plan step %q access_list references unknown or future step/agent %q", step.ID, id)
			}
		}
	}
	return nil
}

func registerWorkflowAccessIDs(ids map[string]bool, step workflowPlanStep) {
	ids[step.ID] = true
	for modelIndex, model := range step.Models {
		ids[workflowAgentID(workflowToolPhaseStep, step, model, modelIndex)] = true
	}
}

func normalizeWorkflowAccessList(values []string) []string {
	if values == nil {
		return nil
	}
	seen := map[string]bool{}
	normalized := make([]string, 0, len(values))
	for _, value := range values {
		id := strings.TrimSpace(value)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		normalized = append(normalized, id)
	}
	return normalized
}

func validateWorkflowFinal(final *workflowFinalStep, allowed map[string]bool) error {
	if final != nil && strings.TrimSpace(final.Model) != "" && !allowed[final.Model] {
		return fmt.Errorf("workflows final model %q is outside decision modelRefs", final.Model)
	}
	return nil
}
