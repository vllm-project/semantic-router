package looper

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func buildStaticWorkflowPlan(cfg workflowsExecutionConfig) (*workflowPlan, error) {
	if len(cfg.Roles) == 0 {
		return nil, fmt.Errorf("workflows static mode requires roles")
	}
	steps := make([]workflowPlanStep, 0, len(cfg.Roles))
	for idx, role := range cfg.Roles {
		step, err := workflowStepFromStaticRole(role, idx)
		if err != nil {
			return nil, err
		}
		steps = append(steps, step)
	}
	return &workflowPlan{
		Steps: steps,
		Final: &workflowFinalStep{
			Model:  strings.TrimSpace(cfg.Final.Model),
			Prompt: workflowStaticFinalPrompt(cfg.Final.Prompt),
		},
	}, nil
}

func workflowStepFromStaticRole(role config.WorkflowRoleConfig, index int) (workflowPlanStep, error) {
	name := strings.TrimSpace(role.Name)
	if name == "" {
		return workflowPlanStep{}, fmt.Errorf("workflows static role %d requires name", index)
	}
	models := normalizeModelNames(role.Models)
	if len(models) == 0 {
		return workflowPlanStep{}, fmt.Errorf("workflows static role %q requires models", name)
	}
	return workflowPlanStep{
		ID:         workflowStaticRoleID(name, index),
		Role:       name,
		Models:     models,
		Prompt:     workflowStaticRolePrompt(name, role.Prompt),
		AccessList: normalizeWorkflowAccessList(role.AccessList),
	}, nil
}

func workflowStaticRoleID(name string, index int) string {
	id := strings.ToLower(strings.TrimSpace(name))
	id = strings.NewReplacer(" ", "-", "_", "-").Replace(id)
	if id == "" {
		return fmt.Sprintf("role-%d", index+1)
	}
	return id
}

func workflowStaticRolePrompt(name string, configured string) string {
	if strings.TrimSpace(configured) != "" {
		return strings.TrimSpace(configured)
	}
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "thinker", "plan", "planner":
		return "Break down the request, identify constraints, and call out risks or missing assumptions."
	case "worker", "solver":
		return "Solve the request independently with concrete reasoning, code, or evidence as needed."
	case "verifier", "reviewer", "critic":
		return "Check prior outputs against the original request, find mistakes, and provide corrections."
	default:
		return "Work on the original request according to this role and provide a concise, useful result."
	}
}

func workflowStaticFinalPrompt(configured string) string {
	if strings.TrimSpace(configured) != "" {
		return strings.TrimSpace(configured)
	}
	return "Synthesize the role outputs into one final answer. Prefer correct, tested, and specific details. Do not mention internal model names unless the user asks."
}
