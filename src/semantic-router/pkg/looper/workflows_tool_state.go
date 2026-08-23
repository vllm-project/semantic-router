package looper

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const (
	workflowToolCallIDPrefix    = "flowtool_"
	workflowToolCallIDSeparator = "__"
	workflowToolPhaseStep       = "step"
	workflowToolPhaseFinal      = "final"
)

type workflowPendingToolCallTrace struct {
	Phase       string   `json:"phase,omitempty"`
	StateID     string   `json:"state_id"`
	AgentID     string   `json:"agent_id,omitempty"`
	StepID      string   `json:"step_id,omitempty"`
	Role        string   `json:"role,omitempty"`
	Model       string   `json:"model"`
	ToolCallIDs []string `json:"tool_call_ids,omitempty"`
}

type workflowAgentToolTurn struct {
	AgentID      string                   `json:"agent_id,omitempty"`
	Phase        string                   `json:"phase,omitempty"`
	StepID       string                   `json:"step_id,omitempty"`
	Role         string                   `json:"role,omitempty"`
	Model        string                   `json:"model,omitempty"`
	ToolCallIDs  []string                 `json:"tool_call_ids,omitempty"`
	AssistantRaw []byte                   `json:"assistant_raw,omitempty"`
	ToolMessages []map[string]interface{} `json:"tool_messages,omitempty"`
}

type workflowPendingToolState struct {
	ID                          string                             `json:"id"`
	CreatedAt                   time.Time                          `json:"created_at"`
	DecisionName                string                             `json:"decision_name,omitempty"`
	Mode                        string                             `json:"mode,omitempty"`
	Template                    string                             `json:"template,omitempty"`
	Plan                        *workflowPlan                      `json:"plan,omitempty"`
	PlannerResp                 *ModelResponse                     `json:"planner_resp,omitempty"`
	WorkerModels                []string                           `json:"worker_models,omitempty"`
	StepResults                 []workflowStepResult               `json:"step_results,omitempty"`
	SemanticRequest             *llmprotocol.Request               `json:"semantic_request"`
	Phase                       string                             `json:"phase,omitempty"`
	AgentID                     string                             `json:"agent_id,omitempty"`
	StepID                      string                             `json:"step_id,omitempty"`
	Role                        string                             `json:"role,omitempty"`
	AccessList                  []string                           `json:"access_list,omitempty"`
	StepIndex                   int                                `json:"step_index"`
	ModelIndex                  int                                `json:"model_index"`
	Model                       string                             `json:"model"`
	StepRequest                 *openai.ChatCompletionNewParams    `json:"step_request,omitempty"`
	AgentRequest                *openai.ChatCompletionNewParams    `json:"agent_request,omitempty"`
	AssistantRaw                []byte                             `json:"assistant_raw,omitempty"`
	CurrentStepResponses        []*ModelResponse                   `json:"current_step_responses,omitempty"`
	CurrentStepFailed           []FusionFailedModel                `json:"current_step_failed,omitempty"`
	CurrentStepToolTrajectories map[string][]workflowAgentToolTurn `json:"current_step_tool_trajectories,omitempty"`
	Iteration                   int                                `json:"iteration"`
	ToolCallSeq                 int                                `json:"tool_call_seq,omitempty"`
	ToolCallIDs                 []string                           `json:"tool_call_ids,omitempty"`
	ToolTrajectory              []workflowAgentToolTurn            `json:"tool_trajectory,omitempty"`
	Streaming                   bool                               `json:"streaming"`
	IncludeUsage                bool                               `json:"include_usage"`
}

func workflowToolPhase(state *workflowPendingToolState) string {
	if state == nil || strings.TrimSpace(state.Phase) == "" {
		return workflowToolPhaseStep
	}
	return state.Phase
}

func newWorkflowToolStateID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

func (l *WorkflowsLooper) formatWorkflowToolCallInterrupt(
	ctx context.Context,
	interrupt *workflowToolCallInterrupt,
	cfg workflowsExecutionConfig,
) (*Response, error) {
	if interrupt == nil || interrupt.resp == nil || interrupt.state == nil {
		return nil, fmt.Errorf("workflow tool-call interrupt is incomplete")
	}
	state := interrupt.state
	if state.SemanticRequest == nil {
		return nil, fmt.Errorf("workflow tool-call interrupt is missing its neutral original request")
	}
	patchedRaw, toolCallIDs, formatWorkflowToolCallInterruptErr := patchWorkflowToolCallResponse(interrupt.resp.Raw, state)
	if formatWorkflowToolCallInterruptErr != nil {
		return nil, formatWorkflowToolCallInterruptErr
	}
	state.AssistantRaw = patchedRaw
	state.ToolCallIDs = append([]string(nil), toolCallIDs...)
	state.CreatedAt = time.Now().UTC()
	if _, err := l.toolStates.Put(ctx, state); err != nil {
		return nil, err
	}

	patchedResp := *interrupt.resp
	patchedResp.Raw = patchedRaw
	semantic, _, _, formatWorkflowToolCallInterruptErr := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, patchedRaw)
	if formatWorkflowToolCallInterruptErr != nil {
		return nil, fmt.Errorf("decode patched workflow tool-call response: %w", formatWorkflowToolCallInterruptErr)
	}
	patchedResp.Semantic = semanticModelResponse(semantic, patchedResp.Model)
	pendingStep := workflowPendingTraceStep(state)
	traceResults := workflowTraceResultsForPendingToolCall(state, &patchedResp)
	trace := buildWorkflowTrace(cfg, state.WorkerModels, state.Plan, traceResults, workflowFailedModels(traceResults))
	if workflowToolPhase(state) == workflowToolPhaseFinal {
		trace.FinalToolTrajectory = workflowToolTurnTraces(state.ToolTrajectory)
	}
	trace.PendingToolCall = &workflowPendingToolCallTrace{
		Phase:       workflowToolPhase(state),
		StateID:     state.ID,
		AgentID:     state.AgentID,
		StepID:      pendingStep.ID,
		Role:        pendingStep.Role,
		Model:       state.Model,
		ToolCallIDs: toolCallIDs,
	}
	extraProgress := workflowPendingProgressResponses(state, &patchedResp)
	usage := workflowProgressUsage(state.PlannerResp, traceResults, extraProgress...)
	modelsUsed := workflowProgressModels(cfg, state.PlannerResp, traceResults, extraProgress...)
	if state.Streaming {
		return formatWorkflowStreamingResponse(
			&patchedResp, modelsUsed, state.Iteration, trace, usage, cfg,
			state.IncludeUsage,
		)
	}
	return formatWorkflowJSONResponse(&patchedResp, modelsUsed, state.Iteration, trace, usage, cfg)
}

func workflowPendingTraceStep(state *workflowPendingToolState) workflowPlanStep {
	if workflowToolPhase(state) == workflowToolPhaseFinal {
		return workflowPlanStep{ID: "final", Role: "final", Models: []string{state.Model}}
	}
	if state != nil && state.Plan != nil && state.StepIndex >= 0 && state.StepIndex < len(state.Plan.Steps) {
		return state.Plan.Steps[state.StepIndex]
	}
	return workflowPlanStep{}
}

func workflowTraceResultsForPendingToolCall(state *workflowPendingToolState, patchedResp *ModelResponse) []workflowStepResult {
	traceResults := append([]workflowStepResult(nil), state.StepResults...)
	if workflowToolPhase(state) == workflowToolPhaseFinal {
		return traceResults
	}
	pendingStep := workflowPendingTraceStep(state)
	traceResults = append(traceResults, workflowStepResult{
		step:             pendingStep,
		responses:        append(append([]*ModelResponse(nil), state.CurrentStepResponses...), patchedResp),
		failed:           append([]FusionFailedModel(nil), state.CurrentStepFailed...),
		toolTrajectories: workflowPendingStepToolTrajectories(state),
	})
	return traceResults
}

func workflowPendingProgressResponses(state *workflowPendingToolState, patchedResp *ModelResponse) []*ModelResponse {
	if workflowToolPhase(state) == workflowToolPhaseFinal {
		return []*ModelResponse{patchedResp}
	}
	return nil
}

func workflowFailedModels(results []workflowStepResult) []FusionFailedModel {
	var failed []FusionFailedModel
	for _, result := range results {
		failed = append(failed, result.failed...)
	}
	return failed
}

func workflowProgressUsage(plannerResp *ModelResponse, results []workflowStepResult, extra ...*ModelResponse) TokenUsage {
	usage := SumUsage(plannerResp)
	for _, result := range results {
		usage = usage.Add(result.usageResponses()...)
	}
	usage = usage.Add(extra...)
	return usage
}

func workflowProgressModels(cfg workflowsExecutionConfig, plannerResp *ModelResponse, results []workflowStepResult, extra ...*ModelResponse) []string {
	var models []string
	models = appendUniqueWorkflowModel(models, cfg.PlannerModel)
	if plannerResp != nil {
		models = appendUniqueWorkflowModel(models, plannerResp.Model)
	}
	for _, result := range results {
		for _, resp := range result.usageResponses() {
			if resp != nil {
				models = appendUniqueWorkflowModel(models, resp.Model)
			}
		}
	}
	for _, resp := range extra {
		if resp != nil {
			models = appendUniqueWorkflowModel(models, resp.Model)
		}
	}
	return models
}

type workflowResumeRequestContext struct {
	originalRequest *openai.ChatCompletionNewParams
	looperRequest   Request
}

func (l *WorkflowsLooper) resumeWorkflowToolCall(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	workerModels []string,
	stateID string,
) (*Response, error) {
	state, err := l.takeWorkflowToolState(ctx, stateID)
	if err != nil {
		return nil, err
	}
	clearSettledWorkflowUsage(state)
	restoreState := true
	defer l.restoreWorkflowToolState(ctx, state, &restoreState)

	out, consumed, err := l.resumeWorkflowToolCallWithState(ctx, req, cfg, workerModels, state)
	if err != nil {
		return nil, err
	}
	restoreState = !consumed
	return out, nil
}

// clearSettledWorkflowUsage turns a persisted workflow continuation into a
// fresh accounting boundary. Responses retained in the state are execution
// context from earlier HTTP requests whose provider usage was already exposed
// and post-accounted with the tool-call interrupt response. Keeping their
// content, model, and trace fields preserves workflow continuity; clearing only
// Usage ensures every resume response reports the calls made by that request.
func clearSettledWorkflowUsage(state *workflowPendingToolState) {
	if state == nil {
		return
	}
	clearWorkflowResponseUsage(state.PlannerResp)
	for _, result := range state.StepResults {
		for _, response := range result.responses {
			clearWorkflowResponseUsage(response)
		}
		for _, response := range result.accountingResponses {
			clearWorkflowResponseUsage(response)
		}
	}
	for _, response := range state.CurrentStepResponses {
		clearWorkflowResponseUsage(response)
	}
}

func clearWorkflowResponseUsage(response *ModelResponse) {
	if response != nil {
		response.Usage = TokenUsage{}
	}
}

func (l *WorkflowsLooper) takeWorkflowToolState(ctx context.Context, stateID string) (*workflowPendingToolState, error) {
	state, ok, err := l.toolStates.Take(ctx, stateID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("workflow tool state %q not found or expired", stateID)
	}
	return state, nil
}

func (l *WorkflowsLooper) restoreWorkflowToolState(ctx context.Context, state *workflowPendingToolState, restore *bool) {
	if *restore {
		_, _ = l.toolStates.Put(ctx, state)
	}
}

func (l *WorkflowsLooper) resumeWorkflowToolCallWithState(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	workerModels []string,
	state *workflowPendingToolState,
) (*Response, bool, error) {
	state.Streaming = req.IsStreaming
	resumeCtx, err := newWorkflowResumeRequestContext(req, state)
	if err != nil {
		return nil, false, err
	}
	if validateErr := validateWorkflowResumeState(state, workerModels, cfg, req.DecisionName); validateErr != nil {
		return nil, false, validateErr
	}
	toolMessages, err := workflowToolMessagesForState(req.executionRequest, state)
	if err != nil {
		return nil, false, err
	}
	resp, agentReq, err := l.callWorkflowAgentAfterTool(ctx, req, cfg, state, toolMessages)
	if err != nil {
		return nil, false, err
	}
	if resp.HasToolCalls {
		state.AgentRequest = agentReq
		state.Iteration++
		return l.workflowToolInterruptResponse(ctx, cfg, &workflowToolCallInterrupt{resp: resp, state: state})
	}
	if workflowToolPhase(state) == workflowToolPhaseFinal {
		out, finishErr := l.finishResumedWorkflowFinal(ctx, &resumeCtx.looperRequest, cfg, state, resumeCtx.originalRequest, resp)
		if finishErr != nil {
			return nil, false, newPartialExecutionError(
				finishErr,
				workflowExecutionEvidence(cfg, state.PlannerResp, state.StepResults, resp),
			)
		}
		return out, true, nil
	}

	results, interrupt, err := l.continueWorkflowAfterResumedAgent(ctx, req, cfg, state, resp, resumeCtx)
	if err != nil {
		return nil, false, newPartialExecutionError(
			err,
			workflowExecutionEvidence(cfg, state.PlannerResp, results),
		)
	}
	if interrupt != nil {
		return l.workflowToolInterruptResponse(ctx, cfg, interrupt)
	}

	out, finishErr := l.finishResumedWorkflow(ctx, &resumeCtx.looperRequest, cfg, state, resumeCtx.originalRequest, results)
	if finishErr != nil {
		return nil, false, newPartialExecutionError(
			finishErr,
			workflowExecutionEvidence(cfg, state.PlannerResp, results),
		)
	}
	return out, true, nil
}

func (l *WorkflowsLooper) workflowToolInterruptResponse(
	ctx context.Context,
	cfg workflowsExecutionConfig,
	interrupt *workflowToolCallInterrupt,
) (*Response, bool, error) {
	out, err := l.formatWorkflowToolCallInterrupt(ctx, interrupt, cfg)
	if err != nil {
		var plannerResp *ModelResponse
		if interrupt != nil && interrupt.state != nil {
			plannerResp = interrupt.state.PlannerResp
		}
		return nil, false, newPartialExecutionError(
			err,
			workflowInterruptExecutionEvidence(cfg, plannerResp, interrupt),
		)
	}
	return out, true, nil
}

func (l *WorkflowsLooper) continueWorkflowAfterResumedAgent(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	state *workflowPendingToolState,
	resp *ModelResponse,
	resumeCtx workflowResumeRequestContext,
) ([]workflowStepResult, *workflowToolCallInterrupt, error) {
	results, interrupt, err := l.finishCurrentWorkflowStepAfterResume(ctx, req, cfg, state, resp, resumeCtx.originalRequest)
	if err != nil || interrupt != nil {
		return results, interrupt, err
	}
	return l.executeRemainingWorkflowStepsAfterResume(ctx, &resumeCtx.looperRequest, cfg, state, resumeCtx.originalRequest, results)
}

func newWorkflowResumeRequestContext(req *Request, state *workflowPendingToolState) (workflowResumeRequestContext, error) {
	if state == nil || state.SemanticRequest == nil {
		return workflowResumeRequestContext{}, fmt.Errorf("workflow tool state is missing its neutral original request")
	}
	originalSemantic := cloneSemanticRequest(state.SemanticRequest)
	original, err := NewRequestFromSemantic(originalSemantic)
	if err != nil {
		return workflowResumeRequestContext{}, fmt.Errorf("restore workflow original request: %w", err)
	}
	originalRequest := original.executionRequest
	resumeReq := *req
	resumeReq.SemanticRequest = cloneSemanticRequest(state.SemanticRequest)
	resumeReq.executionRequest = originalRequest
	return workflowResumeRequestContext{originalRequest: originalRequest, looperRequest: resumeReq}, nil
}

func validateWorkflowResumeState(
	state *workflowPendingToolState,
	workerModels []string,
	cfg workflowsExecutionConfig,
	decisionName string,
) error {
	if state == nil {
		return fmt.Errorf("workflow tool state missing")
	}
	if strings.TrimSpace(state.DecisionName) != "" && state.DecisionName != decisionName {
		return fmt.Errorf("workflow tool state belongs to decision %q, not %q", state.DecisionName, decisionName)
	}
	if strings.TrimSpace(state.Mode) != "" && state.Mode != cfg.Mode {
		return fmt.Errorf("workflow tool state mode %q does not match current mode %q", state.Mode, cfg.Mode)
	}
	if strings.TrimSpace(state.Template) != "" && state.Template != cfg.Template {
		return fmt.Errorf("workflow tool state template %q does not match current template %q", state.Template, cfg.Template)
	}
	if len(state.WorkerModels) > 0 && !workflowStringSlicesEqual(state.WorkerModels, workerModels) {
		return fmt.Errorf("workflow tool state worker model set changed")
	}
	if err := validateWorkflowPlan(state.Plan, workerModels, cfg); err != nil {
		return err
	}
	if workflowToolPhase(state) == workflowToolPhaseFinal {
		return validateWorkflowFinalResumeState(state, cfg)
	}
	return validateWorkflowStepResumeState(state)
}

func workflowStringSlicesEqual(left []string, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func workflowCurrentStep(state *workflowPendingToolState) (workflowPlanStep, error) {
	if state == nil || state.Plan == nil {
		return workflowPlanStep{}, fmt.Errorf("workflow tool state missing plan")
	}
	if state.StepIndex < 0 || state.StepIndex >= len(state.Plan.Steps) {
		return workflowPlanStep{}, fmt.Errorf("workflow tool state step index %d out of range", state.StepIndex)
	}
	return state.Plan.Steps[state.StepIndex], nil
}
