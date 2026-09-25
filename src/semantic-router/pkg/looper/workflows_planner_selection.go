package looper

import (
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// resolveDynamicWorkflowPlanner resolves only an omitted coordinator from the
// ordered assigned worker pool. Admission uses the exact planner stage without
// calling a model; an explicit coordinator is checked and never substituted.
func (l *WorkflowsLooper) resolveDynamicWorkflowPlanner(
	req *Request, cfg workflowsExecutionConfig, original string, workers []string,
) (workflowsExecutionConfig, error) {
	if cfg.Mode != config.WorkflowModeDynamic {
		return cfg, nil
	}
	if len(normalizeModelNames(workers)) < max(1, cfg.MinSuccessfulResponses) {
		return cfg, fmt.Errorf("%w: dynamic workflow has fewer assigned workers than its required minimum", selection.ErrNoEligibleCandidates)
	}
	candidates := workers
	explicit := cfg.PlannerModel != ""
	if explicit {
		candidates = []string{cfg.PlannerModel}
	}
	planReq := dynamicWorkflowPlannerRequest(req, cfg, original, workers)
	stageReq := workflowModelRequest(planReq, workflowPlannerStageConfig(cfg), false)
	for _, model := range candidates {
		candidate := cfg
		candidate.PlannerModel = model
		err := l.validateWorkflowControlModels(candidate)
		if err == nil {
			err = validateLooperStageContext(req, stageReq, model)
		}
		if err == nil {
			_, err = prepareModelCallBody(stageReq, ModelTarget{Name: model}, CallOptions{
				DecisionName: req.DecisionName, Iteration: 1, candidateRequest: req,
			})
		}
		if err == nil {
			return candidate, nil
		}
		if explicit {
			return cfg, fmt.Errorf("workflow explicit planner %q is ineligible: %w", model, err)
		}
	}
	return cfg, fmt.Errorf("%w: no assigned worker can serve the dynamic planner stage", selection.ErrNoEligibleCandidates)
}

func resolvedWorkflowPlannerForResume(cfg workflowsExecutionConfig, state *workflowPendingToolState, workers []string) (workflowsExecutionConfig, error) {
	if cfg.Mode != config.WorkflowModeDynamic || cfg.PlannerModel != "" {
		return cfg, nil
	}
	// The stored planner response records the actual admitted coordinator.
	// Resume preserves that identity; it does not plan again or silently choose
	// a different coordinator from the new tool-result text.
	if state == nil || state.PlannerResp == nil || !slices.Contains(workers, state.PlannerResp.Model) {
		return cfg, fmt.Errorf("%w: resumed workflow has no planner in the current assigned worker set", selection.ErrNoEligibleCandidates)
	}
	cfg.PlannerModel = state.PlannerResp.Model
	return cfg, nil
}
