package looper

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (l *WorkflowsLooper) executeWorkflowSteps(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	plan *workflowPlan,
	plannerResp *ModelResponse,
	workerModels []string,
) ([]workflowStepResult, *workflowToolCallInterrupt, error) {
	results := make([]workflowStepResult, 0, len(plan.Steps))
	var previous []workflowStepResult
	for idx, step := range plan.Steps {
		prompt := buildWorkflowStepPrompt(req.executionRequest, step, previous)
		stepReq := appendFusionStageMessage(req.executionRequest, prompt)
		responses, accountingResponses, failed, interrupt, err := l.executeWorkflowStep(ctx, req, cfg, plan, step, stepReq, idx, 0, idx+2)
		if err != nil {
			results = append(results, workflowStepResult{step: step, responses: responses, accountingResponses: accountingResponses, failed: failed})
			return results, nil, err
		}
		if interrupt != nil {
			interrupt.state.Plan = plan
			interrupt.state.PlannerResp = plannerResp
			interrupt.state.WorkerModels = append([]string(nil), workerModels...)
			interrupt.state.StepResults = append([]workflowStepResult(nil), previous...)
			return nil, interrupt, nil
		}
		result := workflowStepResult{step: step, responses: responses, accountingResponses: accountingResponses, failed: failed}
		results = append(results, result)
		previous = append(previous, result)
	}
	return results, nil, nil
}

func (l *WorkflowsLooper) executeWorkflowStep(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	plan *workflowPlan,
	step workflowPlanStep,
	stepReq *openai.ChatCompletionNewParams,
	stepIndex int,
	modelStartIndex int,
	iterationStart int,
) ([]*ModelResponse, []*ModelResponse, []FusionFailedModel, *workflowToolCallInterrupt, error) {
	if requestHasTools(stepReq) {
		responses, failed, interrupt, err := l.executeWorkflowStepSequential(ctx, req, cfg, plan, step, stepReq, stepIndex, modelStartIndex, iterationStart)
		return responses, responses, failed, interrupt, err
	}
	stepCtx, cancel := workflowRoundContext(ctx, cfg)
	defer cancel()
	models := step.Models[modelStartIndex:]
	results := l.startWorkflowStepWorkers(stepCtx, req, cfg, stepReq, models, modelStartIndex, iterationStart)

	collector := newWorkflowStepCollector(step, cfg, len(models), cancel)
	remaining := len(models)
	for remaining > 0 {
		select {
		case result := <-results:
			remaining--
			collector.handleResult(result)
			if collector.quorumReached || collector.terminalErr != nil {
				drainWorkflowStepResults(results, collector, &remaining)
				return collector.responses(), collector.accountingResponses(), collector.failed, nil, collector.terminalErr
			}
		case <-stepCtx.Done():
			responses, err := collector.handleTimeout(stepCtx.Err())
			return responses, collector.accountingResponses(), collector.failed, nil, err
		}
	}

	if collector.terminalErr != nil {
		return collector.responses(), collector.accountingResponses(), collector.failed, nil, collector.terminalErr
	}
	responses, err := collector.finalize()
	return responses, collector.accountingResponses(), collector.failed, nil, err
}

func drainWorkflowStepResults(results <-chan workflowModelResult, collector *workflowStepCollector, remaining *int) {
	for *remaining > 0 {
		select {
		case result := <-results:
			(*remaining)--
			collector.handleResult(result)
		default:
			return
		}
	}
}

func (l *WorkflowsLooper) startWorkflowStepWorkers(
	stepCtx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	stepReq *openai.ChatCompletionNewParams,
	models []string,
	modelStartIndex int,
	iterationStart int,
) <-chan workflowModelResult {
	results := make(chan workflowModelResult, len(models))
	sem := make(chan struct{}, cfg.MaxParallel)
	for idx, modelName := range models {
		modelIndex := modelStartIndex + idx
		go func(idx int, modelIndex int, modelName string) {
			select {
			case sem <- struct{}{}:
			case <-stepCtx.Done():
				results <- workflowModelResult{index: modelIndex, model: modelName, err: stepCtx.Err()}
				return
			}
			defer func() { <-sem }()
			resp, err := l.callWorkflowModel(stepCtx, stepReq, cfg, modelName, false, iterationStart+idx, req)
			results <- workflowModelResult{index: modelIndex, model: modelName, resp: resp, err: err}
		}(idx, modelIndex, modelName)
	}
	return results
}

type workflowStepCollector struct {
	step          workflowPlanStep
	cfg           workflowsExecutionConfig
	minSuccessful int
	cancel        context.CancelFunc
	ordered       []*ModelResponse
	quorumOrdered []*ModelResponse
	failed        []FusionFailedModel
	quorumReached bool
	terminalErr   error
}

func newWorkflowStepCollector(
	step workflowPlanStep,
	cfg workflowsExecutionConfig,
	modelCount int,
	cancel context.CancelFunc,
) *workflowStepCollector {
	return &workflowStepCollector{
		step:          step,
		cfg:           cfg,
		minSuccessful: workflowRoundMinSuccessful(modelCount, cfg.MinSuccessfulResponses),
		cancel:        cancel,
		ordered:       make([]*ModelResponse, len(step.Models)),
		failed:        make([]FusionFailedModel, 0),
	}
}

func (c *workflowStepCollector) handleResult(result workflowModelResult) {
	if result.err != nil {
		c.failed = append(c.failed, FusionFailedModel{Model: result.model, Error: result.err.Error()})
		if c.cfg.OnError == config.WorkflowOnErrorFail && !c.quorumReached && c.terminalErr == nil {
			c.cancel()
			c.terminalErr = fmt.Errorf("workflow step %q failed for model %q: %w", c.step.ID, result.model, result.err)
		}
		return
	}
	c.ordered[result.index] = result.resp
	responses := c.responses()
	if c.quorumReached || c.terminalErr != nil || len(responses) < c.minSuccessful {
		return
	}
	c.quorumReached = true
	c.quorumOrdered = append([]*ModelResponse(nil), c.ordered...)
	c.cancel()
}

func (c *workflowStepCollector) handleTimeout(err error) ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) > 0 && c.cfg.OnError != config.WorkflowOnErrorFail {
		logging.Warnf("[Workflows] Step %q timed out with %d partial responses; continuing because on_error=skip", c.step.ID, len(responses))
		return responses, nil
	}
	return responses, err
}

func (c *workflowStepCollector) finalize() ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) == 0 {
		return nil, fmt.Errorf("workflow step %q failed: all models failed", c.step.ID)
	}
	return responses, nil
}

func (c *workflowStepCollector) responses() []*ModelResponse {
	if c.quorumReached {
		return workflowResponsesFromOrdered(c.quorumOrdered)
	}
	return workflowResponsesFromOrdered(c.ordered)
}

func (c *workflowStepCollector) accountingResponses() []*ModelResponse {
	return workflowResponsesFromOrdered(c.ordered)
}

func (r workflowStepResult) usageResponses() []*ModelResponse {
	if r.accountingResponses != nil {
		return r.accountingResponses
	}
	return r.responses
}

func (l *WorkflowsLooper) executeWorkflowStepSequential(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	plan *workflowPlan,
	step workflowPlanStep,
	stepReq *openai.ChatCompletionNewParams,
	stepIndex int,
	modelStartIndex int,
	iterationStart int,
) ([]*ModelResponse, []FusionFailedModel, *workflowToolCallInterrupt, error) {
	stepCtx, cancel := workflowRoundContext(ctx, cfg)
	defer cancel()
	responses := make([]*ModelResponse, 0, len(step.Models)-modelStartIndex)
	failed := make([]FusionFailedModel, 0)
	for modelIndex := modelStartIndex; modelIndex < len(step.Models); modelIndex++ {
		modelName := step.Models[modelIndex]
		resp, err := l.callWorkflowModel(stepCtx, stepReq, cfg, modelName, true, iterationStart+(modelIndex-modelStartIndex), req)
		if err != nil {
			failed = append(failed, FusionFailedModel{Model: modelName, Error: err.Error()})
			if stepCtx.Err() != nil && len(responses) > 0 && cfg.OnError != config.WorkflowOnErrorFail {
				return responses, failed, nil, nil
			}
			if cfg.OnError == config.WorkflowOnErrorFail {
				return responses, failed, nil, fmt.Errorf("workflow step %q failed for model %q: %w", step.ID, modelName, err)
			}
			continue
		}
		if resp.HasToolCalls {
			return nil, failed, &workflowToolCallInterrupt{
				resp: resp,
				state: &workflowPendingToolState{
					DecisionName:         req.DecisionName,
					Mode:                 cfg.Mode,
					Template:             cfg.Template,
					Plan:                 plan,
					SemanticRequest:      cloneSemanticRequest(req.SemanticRequest),
					Phase:                workflowToolPhaseStep,
					AgentID:              workflowAgentID(workflowToolPhaseStep, step, modelName, modelIndex),
					StepID:               step.ID,
					Role:                 step.Role,
					AccessList:           append([]string(nil), step.AccessList...),
					StepIndex:            stepIndex,
					ModelIndex:           modelIndex,
					Model:                modelName,
					StepRequest:          cloneRequest(stepReq),
					AgentRequest:         cloneRequest(stepReq),
					CurrentStepResponses: append([]*ModelResponse(nil), responses...),
					CurrentStepFailed:    append([]FusionFailedModel(nil), failed...),
					Iteration:            iterationStart + (modelIndex - modelStartIndex),
					Streaming:            req.IsStreaming,
					IncludeUsage:         streamUsageRequested(req),
				},
			}, nil
		}
		responses = append(responses, resp)
	}
	if len(responses) == 0 {
		return nil, failed, nil, fmt.Errorf("workflow step %q failed: all models failed", step.ID)
	}
	return responses, failed, nil, nil
}
