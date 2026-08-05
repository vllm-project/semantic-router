package looper

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func validateWorkflowStepResumeState(state *workflowPendingToolState) error {
	step, err := workflowCurrentStep(state)
	if err != nil {
		return err
	}
	if state.ModelIndex < 0 || state.ModelIndex >= len(step.Models) {
		return fmt.Errorf("workflow tool state model index %d out of range", state.ModelIndex)
	}
	if step.Models[state.ModelIndex] != state.Model {
		return fmt.Errorf("workflow tool state model %q does not match step model %q", state.Model, step.Models[state.ModelIndex])
	}
	expectedAgentID := workflowAgentID(workflowToolPhaseStep, step, state.Model, state.ModelIndex)
	if strings.TrimSpace(state.AgentID) != "" && state.AgentID != expectedAgentID {
		return fmt.Errorf("workflow tool state agent id %q does not match %q", state.AgentID, expectedAgentID)
	}
	if strings.TrimSpace(state.StepID) != "" && state.StepID != step.ID {
		return fmt.Errorf("workflow tool state step id %q does not match %q", state.StepID, step.ID)
	}
	if strings.TrimSpace(state.Role) != "" && state.Role != step.Role {
		return fmt.Errorf("workflow tool state role %q does not match %q", state.Role, step.Role)
	}
	return nil
}

func validateWorkflowFinalResumeState(state *workflowPendingToolState, cfg workflowsExecutionConfig) error {
	expected, err := resolveWorkflowFinalModel(cfg, state.Plan, state.StepResults)
	if err != nil {
		return err
	}
	if state.Model != expected {
		return fmt.Errorf("workflow final tool state model %q does not match final model %q", state.Model, expected)
	}
	if state.StepIndex != len(state.Plan.Steps) {
		return fmt.Errorf("workflow final tool state step index %d does not match final index %d", state.StepIndex, len(state.Plan.Steps))
	}
	expectedAgentID := workflowAgentID(workflowToolPhaseFinal, workflowPlanStep{ID: "final", Role: "final"}, state.Model, 0)
	if strings.TrimSpace(state.AgentID) != "" && state.AgentID != expectedAgentID {
		return fmt.Errorf("workflow final tool state agent id %q does not match %q", state.AgentID, expectedAgentID)
	}
	return nil
}

func (l *WorkflowsLooper) callWorkflowAgentAfterTool(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	state *workflowPendingToolState,
	toolMessages []map[string]interface{},
) (*ModelResponse, *openai.ChatCompletionNewParams, error) {
	assistantMessage, err := workflowAssistantMessageFromRaw(state.AssistantRaw)
	if err != nil {
		return nil, nil, err
	}
	resumeMessages := append([]map[string]interface{}{assistantMessage}, toolMessages...)
	agentReq, err := appendWorkflowRawMessages(state.AgentRequest, resumeMessages...)
	if err != nil {
		return nil, nil, err
	}
	resp, err := l.callWorkflowModel(ctx, agentReq, cfg, state.Model, true, state.Iteration+1, req)
	if err != nil {
		return nil, nil, fmt.Errorf("workflow tool resume failed for model %q: %w", state.Model, err)
	}
	state.ToolTrajectory = append(state.ToolTrajectory, workflowAgentToolTurnFromState(state, toolMessages))
	return resp, agentReq, nil
}

func workflowAgentToolTurnFromState(state *workflowPendingToolState, toolMessages []map[string]interface{}) workflowAgentToolTurn {
	if state == nil {
		return workflowAgentToolTurn{}
	}
	return workflowAgentToolTurn{
		AgentID:      state.AgentID,
		Phase:        workflowToolPhase(state),
		StepID:       state.StepID,
		Role:         state.Role,
		Model:        state.Model,
		ToolCallIDs:  append([]string(nil), state.ToolCallIDs...),
		AssistantRaw: append([]byte(nil), state.AssistantRaw...),
		ToolMessages: cloneWorkflowMessages(toolMessages),
	}
}

func cloneWorkflowMessages(messages []map[string]interface{}) []map[string]interface{} {
	if len(messages) == 0 {
		return nil
	}
	cloned := make([]map[string]interface{}, 0, len(messages))
	for _, message := range messages {
		cloned = append(cloned, cloneWorkflowMap(message))
	}
	return cloned
}

func (l *WorkflowsLooper) finishCurrentWorkflowStepAfterResume(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	state *workflowPendingToolState,
	firstResp *ModelResponse,
	originalRequest *openai.ChatCompletionNewParams,
) ([]workflowStepResult, *workflowToolCallInterrupt, error) {
	step, err := workflowCurrentStep(state)
	if err != nil {
		return nil, nil, err
	}
	currentResponses := append(append([]*ModelResponse(nil), state.CurrentStepResponses...), firstResp)
	currentFailed := append([]FusionFailedModel(nil), state.CurrentStepFailed...)
	currentToolTrajectories := workflowStepToolTrajectoriesWithAgent(
		state.CurrentStepToolTrajectories,
		state.AgentID,
		state.ToolTrajectory,
	)
	results := append([]workflowStepResult(nil), state.StepResults...)

	for modelIndex := state.ModelIndex + 1; modelIndex < len(step.Models); modelIndex++ {
		modelName := step.Models[modelIndex]
		nextResp, callErr := l.callWorkflowModel(ctx, state.StepRequest, cfg, modelName, true, workflowResumeModelIteration(state, modelIndex), req)
		if callErr != nil {
			currentFailed = append(currentFailed, FusionFailedModel{Model: modelName, Error: callErr.Error()})
			if cfg.OnError == config.WorkflowOnErrorFail {
				return nil, nil, fmt.Errorf("workflow step %q failed for model %q: %w", step.ID, modelName, callErr)
			}
			continue
		}
		if nextResp.HasToolCalls {
			nextState := workflowPendingStateForResumedModel(state, results, originalRequest, modelIndex, modelName, currentResponses, currentFailed, currentToolTrajectories, req.IsStreaming)
			return nil, &workflowToolCallInterrupt{resp: nextResp, state: nextState}, nil
		}
		currentResponses = append(currentResponses, nextResp)
	}

	results = append(results, workflowStepResult{
		step:             step,
		responses:        currentResponses,
		failed:           currentFailed,
		toolTrajectories: currentToolTrajectories,
	})
	return results, nil, nil
}

func workflowResumeModelIteration(state *workflowPendingToolState, modelIndex int) int {
	return state.Iteration + 1 + modelIndex - state.ModelIndex
}

func workflowPendingStateForResumedModel(
	state *workflowPendingToolState,
	results []workflowStepResult,
	originalRequest *openai.ChatCompletionNewParams,
	modelIndex int,
	modelName string,
	currentResponses []*ModelResponse,
	currentFailed []FusionFailedModel,
	currentToolTrajectories map[string][]workflowAgentToolTurn,
	streaming bool,
) *workflowPendingToolState {
	return &workflowPendingToolState{
		DecisionName:                state.DecisionName,
		Mode:                        state.Mode,
		Template:                    state.Template,
		Plan:                        state.Plan,
		PlannerResp:                 state.PlannerResp,
		WorkerModels:                append([]string(nil), state.WorkerModels...),
		StepResults:                 append([]workflowStepResult(nil), results...),
		OriginalRequest:             cloneRequest(originalRequest),
		Phase:                       workflowToolPhaseStep,
		AgentID:                     workflowAgentID(workflowToolPhaseStep, state.Plan.Steps[state.StepIndex], modelName, modelIndex),
		StepID:                      state.Plan.Steps[state.StepIndex].ID,
		Role:                        state.Plan.Steps[state.StepIndex].Role,
		AccessList:                  append([]string(nil), state.Plan.Steps[state.StepIndex].AccessList...),
		StepIndex:                   state.StepIndex,
		ModelIndex:                  modelIndex,
		Model:                       modelName,
		StepRequest:                 cloneRequest(state.StepRequest),
		AgentRequest:                cloneRequest(state.StepRequest),
		CurrentStepResponses:        append([]*ModelResponse(nil), currentResponses...),
		CurrentStepFailed:           append([]FusionFailedModel(nil), currentFailed...),
		CurrentStepToolTrajectories: cloneWorkflowToolTrajectories(currentToolTrajectories),
		Iteration:                   workflowResumeModelIteration(state, modelIndex),
		Streaming:                   streaming,
	}
}

func (l *WorkflowsLooper) executeRemainingWorkflowStepsAfterResume(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	state *workflowPendingToolState,
	originalRequest *openai.ChatCompletionNewParams,
	results []workflowStepResult,
) ([]workflowStepResult, *workflowToolCallInterrupt, error) {
	for stepIndex := state.StepIndex + 1; stepIndex < len(state.Plan.Steps); stepIndex++ {
		nextStep := state.Plan.Steps[stepIndex]
		prompt := buildWorkflowStepPrompt(originalRequest, nextStep, results)
		stepReq := appendFusionStageMessage(originalRequest, prompt)
		responses, failed, interrupt, err := l.executeWorkflowStep(ctx, req, cfg, state.Plan, nextStep, stepReq, stepIndex, 0, state.Iteration+1+len(results))
		if err != nil {
			return nil, nil, err
		}
		if interrupt != nil {
			hydrateResumedWorkflowInterrupt(interrupt, state, results)
			return results, interrupt, nil
		}
		results = append(results, workflowStepResult{step: nextStep, responses: responses, failed: failed})
	}
	return results, nil, nil
}

func hydrateResumedWorkflowInterrupt(
	interrupt *workflowToolCallInterrupt,
	state *workflowPendingToolState,
	results []workflowStepResult,
) {
	interrupt.state.Plan = state.Plan
	interrupt.state.PlannerResp = state.PlannerResp
	interrupt.state.DecisionName = state.DecisionName
	interrupt.state.Mode = state.Mode
	interrupt.state.Template = state.Template
	interrupt.state.WorkerModels = append([]string(nil), state.WorkerModels...)
	interrupt.state.StepResults = append([]workflowStepResult(nil), results...)
}

func (l *WorkflowsLooper) finishResumedWorkflow(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	state *workflowPendingToolState,
	originalRequest *openai.ChatCompletionNewParams,
	results []workflowStepResult,
) (*Response, error) {
	original := extractOriginalContent(originalRequest)
	finalResp, interrupt, err := l.synthesizeWorkflowFinal(ctx, req, cfg, state.Plan, original, results, state.PlannerResp, state.WorkerModels)
	if err != nil {
		return nil, err
	}
	if interrupt != nil {
		return l.formatWorkflowToolCallInterrupt(ctx, interrupt, cfg)
	}
	applyFinalOutputContract(req.OutputContractSpec, finalResp)
	summary := summarizeWorkflowExecution(cfg, state.PlannerResp, results, finalResp)
	trace := buildWorkflowTrace(cfg, state.WorkerModels, state.Plan, results, summary.failed)
	if req.IsStreaming {
		return formatWorkflowStreamingResponse(finalResp, summary.modelsUsed, summary.iterations, trace, summary.usage, cfg)
	}
	return formatWorkflowJSONResponse(finalResp, summary.modelsUsed, summary.iterations, trace, summary.usage, cfg)
}

func (l *WorkflowsLooper) finishResumedWorkflowFinal(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	state *workflowPendingToolState,
	originalRequest *openai.ChatCompletionNewParams,
	finalResp *ModelResponse,
) (*Response, error) {
	applyFinalOutputContract(req.OutputContractSpec, finalResp)
	summary := summarizeWorkflowExecution(cfg, state.PlannerResp, state.StepResults, finalResp)
	trace := buildWorkflowTrace(cfg, state.WorkerModels, state.Plan, state.StepResults, summary.failed)
	trace.FinalToolTrajectory = workflowToolTurnTraces(state.ToolTrajectory)
	if req.IsStreaming {
		return formatWorkflowStreamingResponse(finalResp, summary.modelsUsed, summary.iterations, trace, summary.usage, cfg)
	}
	return formatWorkflowJSONResponse(finalResp, summary.modelsUsed, summary.iterations, trace, summary.usage, cfg)
}
