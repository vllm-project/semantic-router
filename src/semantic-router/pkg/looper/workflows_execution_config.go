package looper

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func resolveWorkflowsExecutionConfig(req *Request) workflowsExecutionConfig {
	cfg := defaultWorkflowsExecutionConfig()
	if req != nil && req.Algorithm != nil && req.Algorithm.Workflows != nil {
		applyWorkflowsAlgorithmConfig(&cfg, req.Algorithm.Workflows)
	}
	return normalizeWorkflowsExecutionConfig(cfg)
}

func defaultWorkflowsExecutionConfig() workflowsExecutionConfig {
	return workflowsExecutionConfig{
		Mode:                         config.WorkflowModeStatic,
		Template:                     "micro_agent",
		PlannerMaxCompletionTokens:   2048,
		MaxSteps:                     3,
		MaxParallel:                  2,
		IncludeIntermediateResponses: true,
		OnError:                      config.WorkflowOnErrorFail,
	}
}

func applyWorkflowsAlgorithmConfig(cfg *workflowsExecutionConfig, src *config.WorkflowsAlgorithmConfig) {
	if src.Mode != "" {
		cfg.Mode = strings.TrimSpace(src.Mode)
	}
	if src.Template != "" {
		cfg.Template = strings.TrimSpace(src.Template)
	}
	applyWorkflowPlannerConfig(cfg, src.Planner)
	applyWorkflowPlanConfig(cfg, src)
	applyWorkflowLimitsConfig(cfg, src)
	applyWorkflowOutputConfig(cfg, src)
}

func applyWorkflowPlanConfig(cfg *workflowsExecutionConfig, src *config.WorkflowsAlgorithmConfig) {
	if len(src.Roles) > 0 {
		cfg.Roles = append([]config.WorkflowRoleConfig(nil), src.Roles...)
	}
	if !src.Final.IsZero() {
		cfg.Final = src.Final
	}
}

func applyWorkflowLimitsConfig(cfg *workflowsExecutionConfig, src *config.WorkflowsAlgorithmConfig) {
	if src.MaxSteps > 0 {
		cfg.MaxSteps = src.MaxSteps
	}
	if src.MaxParallel > 0 {
		cfg.MaxParallel = src.MaxParallel
	}
	if src.MaxCompletionTokens > 0 {
		cfg.MaxCompletionTokens = src.MaxCompletionTokens
	}
	if src.RoundTimeoutSeconds > 0 {
		cfg.RoundTimeoutSeconds = src.RoundTimeoutSeconds
	}
	if src.MinSuccessfulResponses > 0 {
		cfg.MinSuccessfulResponses = src.MinSuccessfulResponses
	}
}

func applyWorkflowOutputConfig(cfg *workflowsExecutionConfig, src *config.WorkflowsAlgorithmConfig) {
	if src.Temperature != nil {
		cfg.Temperature = src.Temperature
	}
	if src.IncludeIntermediateResponses != nil {
		cfg.IncludeIntermediateResponses = *src.IncludeIntermediateResponses
	}
	if src.OnError != "" {
		cfg.OnError = strings.TrimSpace(src.OnError)
	}
}

func applyWorkflowPlannerConfig(cfg *workflowsExecutionConfig, planner config.WorkflowPlannerConfig) {
	if planner.Model != "" {
		cfg.PlannerModel = strings.TrimSpace(planner.Model)
	}
	if planner.MaxCompletionTokens > 0 {
		cfg.PlannerMaxCompletionTokens = planner.MaxCompletionTokens
	}
}

func normalizeWorkflowsExecutionConfig(cfg workflowsExecutionConfig) workflowsExecutionConfig {
	if cfg.Mode == "" {
		cfg.Mode = config.WorkflowModeStatic
	}
	if cfg.Template == "" {
		cfg.Template = "micro_agent"
	}
	if cfg.OnError == "" {
		cfg.OnError = config.WorkflowOnErrorFail
	}
	if cfg.Mode == config.WorkflowModeDynamic && cfg.PlannerMaxCompletionTokens <= 0 {
		cfg.PlannerMaxCompletionTokens = 2048
	}
	return cfg
}

func (l *WorkflowsLooper) validateWorkflowControlModels(cfg workflowsExecutionConfig) error {
	if cfg.PlannerModel == "" {
		return nil
	}
	for _, flowName := range l.cfg.Flow.EffectiveModelNames() {
		if cfg.PlannerModel == flowName {
			return fmt.Errorf("workflows planner.model cannot be direct Flow model %q", cfg.PlannerModel)
		}
	}
	return nil
}
