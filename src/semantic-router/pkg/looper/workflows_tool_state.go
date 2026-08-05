package looper

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go"
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
	OriginalRequest             *openai.ChatCompletionNewParams    `json:"original_request,omitempty"`
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

func findWorkflowToolStateID(req *openai.ChatCompletionNewParams) (string, bool) {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return "", false
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return "", false
	}
	for _, message := range trailingWorkflowToolMessages(messages) {
		if id, ok := message["tool_call_id"].(string); ok {
			if stateID, parsed := parseWorkflowToolStateID(id); parsed {
				return stateID, true
			}
		}
	}
	return "", false
}

func parseWorkflowToolStateID(toolCallID string) (string, bool) {
	if !strings.HasPrefix(toolCallID, workflowToolCallIDPrefix) {
		return "", false
	}
	rest := strings.TrimPrefix(toolCallID, workflowToolCallIDPrefix)
	idx := strings.Index(rest, workflowToolCallIDSeparator)
	if idx <= 0 {
		return "", false
	}
	return rest[:idx], true
}

func requestHasTools(req *openai.ChatCompletionNewParams) bool {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return false
	}
	tools, ok := reqMap["tools"].([]interface{})
	if ok && len(tools) > 0 {
		return true
	}
	functions, ok := reqMap["functions"].([]interface{})
	return ok && len(functions) > 0
}

func requestAsMap(req *openai.ChatCompletionNewParams) (map[string]interface{}, bool) {
	if req == nil {
		return nil, false
	}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, false
	}
	var reqMap map[string]interface{}
	if err := json.Unmarshal(data, &reqMap); err != nil {
		return nil, false
	}
	return reqMap, true
}

func workflowToolMessagesForState(req *openai.ChatCompletionNewParams, state *workflowPendingToolState) ([]map[string]interface{}, error) {
	stateID, err := workflowToolStateID(state)
	if err != nil {
		return nil, err
	}
	trailingTools, err := workflowTrailingToolMessagesForRequest(req, stateID)
	if err != nil {
		return nil, err
	}
	pending := workflowPendingToolCallIDSet(state.ToolCallIDs)
	return workflowValidatedToolMessages(trailingTools, stateID, pending)
}

func workflowToolStateID(state *workflowPendingToolState) (string, error) {
	if state == nil {
		return "", fmt.Errorf("workflow tool state missing")
	}
	if state.ID == "" {
		return "", fmt.Errorf("workflow tool state ID missing")
	}
	return state.ID, nil
}

func workflowAgentID(phase string, step workflowPlanStep, model string, modelIndex int) string {
	if phase == workflowToolPhaseFinal {
		return "final:" + strings.TrimSpace(model)
	}
	stepID := strings.TrimSpace(step.ID)
	if stepID == "" {
		stepID = "step"
	}
	modelName := strings.TrimSpace(model)
	if modelName == "" {
		modelName = "model"
	}
	return fmt.Sprintf("%s:%d:%s", stepID, modelIndex, modelName)
}

func workflowTrailingToolMessagesForRequest(req *openai.ChatCompletionNewParams, stateID string) ([]map[string]interface{}, error) {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return nil, fmt.Errorf("could not parse request messages")
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("request messages missing")
	}
	trailingTools := trailingWorkflowToolMessages(messages)
	if len(trailingTools) == 0 {
		return nil, fmt.Errorf("request does not end with workflow tool messages for state %q", stateID)
	}
	return trailingTools, nil
}

func workflowValidatedToolMessages(
	trailingTools []map[string]interface{},
	stateID string,
	pending map[string]bool,
) ([]map[string]interface{}, error) {
	seen := map[string]bool{}
	var toolMessages []map[string]interface{}
	for _, message := range trailingTools {
		id, ok := workflowToolMessageID(message)
		if !ok {
			continue
		}
		if err := validateWorkflowToolMessageID(id, stateID, pending); err != nil {
			return nil, err
		}
		if !seen[id] {
			toolMessages = append(toolMessages, cloneWorkflowMap(message))
			seen[id] = true
		}
	}
	if len(toolMessages) == 0 {
		return nil, fmt.Errorf("no tool messages found for workflow state %q", stateID)
	}
	if err := workflowRequirePendingToolMessages(pending, seen); err != nil {
		return nil, err
	}
	return toolMessages, nil
}

func workflowToolMessageID(message map[string]interface{}) (string, bool) {
	id, ok := message["tool_call_id"].(string)
	return id, ok && id != ""
}

func validateWorkflowToolMessageID(id string, stateID string, pending map[string]bool) error {
	parsed, isFlow := parseWorkflowToolStateID(id)
	if !isFlow {
		return fmt.Errorf("tool result %q is not a Router Flow tool_call_id", id)
	}
	if parsed != stateID {
		return fmt.Errorf("tool result %q belongs to workflow state %q, not %q", id, parsed, stateID)
	}
	if len(pending) > 0 && !pending[id] {
		return fmt.Errorf("tool result %q was not requested by workflow state %q", id, stateID)
	}
	return nil
}

func workflowRequirePendingToolMessages(pending map[string]bool, seen map[string]bool) error {
	for id := range pending {
		if !seen[id] {
			return fmt.Errorf("missing tool result for workflow tool_call_id %q", id)
		}
	}
	return nil
}

func trailingWorkflowToolMessages(messages []interface{}) []map[string]interface{} {
	var reversed []map[string]interface{}
	for i := len(messages) - 1; i >= 0; i-- {
		message, ok := messages[i].(map[string]interface{})
		if !ok || message["role"] != "tool" {
			break
		}
		reversed = append(reversed, message)
	}
	if len(reversed) == 0 {
		return nil
	}
	ordered := make([]map[string]interface{}, len(reversed))
	for i := range reversed {
		ordered[len(reversed)-1-i] = reversed[i]
	}
	return ordered
}

func workflowPendingToolCallIDSet(ids []string) map[string]bool {
	if len(ids) == 0 {
		return nil
	}
	pending := make(map[string]bool, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) != "" {
			pending[id] = true
		}
	}
	return pending
}

func patchWorkflowToolCallResponse(raw []byte, state *workflowPendingToolState) ([]byte, []string, error) {
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("empty tool-call response")
	}
	ensureWorkflowToolStateID(state)
	completion, message, err := workflowToolCallCompletionMessage(raw)
	if err != nil {
		return nil, nil, err
	}
	toolCalls, err := workflowToolCallsFromMessage(message)
	if err != nil {
		return nil, nil, err
	}
	ids := patchWorkflowToolCallIDs(toolCalls, state)
	normalizeWorkflowToolCallFinishReason(completion)
	patched, err := json.Marshal(completion)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal workflow tool-call response: %w", err)
	}
	return patched, ids, nil
}

func normalizeWorkflowToolCallFinishReason(completion map[string]interface{}) {
	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return
	}
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return
	}
	if toolCalls, ok := message["tool_calls"].([]interface{}); ok && len(toolCalls) > 0 {
		choice["finish_reason"] = "tool_calls"
	}
}

func ensureWorkflowToolStateID(state *workflowPendingToolState) string {
	if state.ID != "" {
		return state.ID
	}
	state.ID = newWorkflowToolStateID()
	return state.ID
}

func workflowToolCallCompletionMessage(raw []byte) (map[string]interface{}, map[string]interface{}, error) {
	var completion map[string]interface{}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, nil, fmt.Errorf("parse workflow tool-call response: %w", err)
	}
	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, nil, fmt.Errorf("workflow tool-call response missing choices")
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("workflow tool-call response choice is invalid")
	}
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("workflow tool-call response missing message")
	}
	return completion, message, nil
}

func workflowToolCallsFromMessage(message map[string]interface{}) ([]interface{}, error) {
	toolCalls, ok := message["tool_calls"].([]interface{})
	if !ok || len(toolCalls) == 0 {
		return workflowToolCallsFromLegacyFunctionCall(message)
	}
	return toolCalls, nil
}

func workflowToolCallsFromLegacyFunctionCall(message map[string]interface{}) ([]interface{}, error) {
	functionCall, ok := message["function_call"].(map[string]interface{})
	if !ok || len(functionCall) == 0 {
		return nil, fmt.Errorf("workflow tool-call response missing tool_calls")
	}
	name, _ := functionCall["name"].(string)
	if strings.TrimSpace(name) == "" {
		return nil, fmt.Errorf("workflow function_call response missing name")
	}
	arguments, _ := functionCall["arguments"].(string)
	toolCall := map[string]interface{}{
		"id":   newWorkflowLegacyFunctionToolCallID(name),
		"type": "function",
		"function": map[string]interface{}{
			"name":      name,
			"arguments": arguments,
		},
	}
	toolCalls := []interface{}{toolCall}
	message["tool_calls"] = toolCalls
	delete(message, "function_call")
	return toolCalls, nil
}

func newWorkflowLegacyFunctionToolCallID(name string) string {
	normalized := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= 'A' && r <= 'Z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '-' || r == '_':
			return r
		default:
			return '_'
		}
	}, name)
	normalized = strings.Trim(normalized, "_-")
	if normalized == "" {
		normalized = "function"
	}
	return "call_" + normalized + "_" + newWorkflowToolStateID()
}

func patchWorkflowToolCallIDs(toolCalls []interface{}, state *workflowPendingToolState) []string {
	stateID := ensureWorkflowToolStateID(state)
	ids := make([]string, 0, len(toolCalls))
	for _, rawCall := range toolCalls {
		call, ok := rawCall.(map[string]interface{})
		if !ok {
			continue
		}
		originalID, _ := call["id"].(string)
		call["id"] = workflowPatchedToolCallID(originalID, stateID, nextWorkflowToolCallSeq(state))
		if id, ok := call["id"].(string); ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func nextWorkflowToolCallSeq(state *workflowPendingToolState) int {
	if state == nil {
		return 0
	}
	seq := state.ToolCallSeq
	state.ToolCallSeq++
	return seq
}

func workflowPatchedToolCallID(originalID string, stateID string, sequence int) string {
	if originalID == "" {
		originalID = newWorkflowToolStateID()
	}
	if _, alreadyFlow := parseWorkflowToolStateID(originalID); alreadyFlow {
		return originalID
	}
	return fmt.Sprintf("%s%s%s%d%s%s", workflowToolCallIDPrefix, stateID, workflowToolCallIDSeparator, sequence, workflowToolCallIDSeparator, originalID)
}

func workflowAssistantMessageFromRaw(raw []byte) (map[string]interface{}, error) {
	var completion struct {
		Choices []struct {
			Message map[string]interface{} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, fmt.Errorf("parse workflow assistant tool-call message: %w", err)
	}
	if len(completion.Choices) == 0 || completion.Choices[0].Message == nil {
		return nil, fmt.Errorf("workflow assistant tool-call message missing")
	}
	message := cloneWorkflowMap(completion.Choices[0].Message)
	message["role"] = "assistant"
	return message, nil
}

func appendWorkflowRawMessages(req *openai.ChatCompletionNewParams, rawMessages ...map[string]interface{}) (*openai.ChatCompletionNewParams, error) {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return nil, fmt.Errorf("could not parse request")
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("request messages missing")
	}
	for _, message := range rawMessages {
		messages = append(messages, cloneWorkflowMap(message))
	}
	reqMap["messages"] = messages
	data, err := json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("marshal request with workflow tool messages: %w", err)
	}
	var appended openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &appended); err != nil {
		return nil, fmt.Errorf("parse request with workflow tool messages: %w", err)
	}
	return &appended, nil
}

func cloneWorkflowMap(src map[string]interface{}) map[string]interface{} {
	cloned := make(map[string]interface{}, len(src))
	for key, value := range src {
		cloned[key] = value
	}
	return cloned
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
	patchedRaw, toolCallIDs, err := patchWorkflowToolCallResponse(interrupt.resp.Raw, state)
	if err != nil {
		return nil, err
	}
	state.AssistantRaw = patchedRaw
	state.ToolCallIDs = append([]string(nil), toolCallIDs...)
	state.CreatedAt = time.Now().UTC()
	if _, err := l.toolStates.Put(ctx, state); err != nil {
		return nil, err
	}

	patchedResp := *interrupt.resp
	patchedResp.Raw = patchedRaw
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
		return formatWorkflowStreamingResponse(&patchedResp, modelsUsed, state.Iteration, trace, usage, cfg)
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
		usage = usage.Add(result.responses...)
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
		for _, resp := range result.responses {
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
	restoreState := true
	defer l.restoreWorkflowToolState(ctx, state, &restoreState)

	out, consumed, err := l.resumeWorkflowToolCallWithState(ctx, req, cfg, workerModels, state)
	if err != nil {
		return nil, err
	}
	restoreState = !consumed
	return out, nil
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
	resumeCtx := newWorkflowResumeRequestContext(req, state)
	if validateErr := validateWorkflowResumeState(state, workerModels, cfg, req.DecisionName); validateErr != nil {
		return nil, false, validateErr
	}
	toolMessages, err := workflowToolMessagesForState(req.OriginalRequest, state)
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
		return out, finishErr == nil, finishErr
	}

	results, interrupt, err := l.continueWorkflowAfterResumedAgent(ctx, req, cfg, state, resp, resumeCtx)
	if err != nil {
		return nil, false, err
	}
	if interrupt != nil {
		return l.workflowToolInterruptResponse(ctx, cfg, interrupt)
	}

	out, finishErr := l.finishResumedWorkflow(ctx, &resumeCtx.looperRequest, cfg, state, resumeCtx.originalRequest, results)
	return out, finishErr == nil, finishErr
}

func (l *WorkflowsLooper) workflowToolInterruptResponse(
	ctx context.Context,
	cfg workflowsExecutionConfig,
	interrupt *workflowToolCallInterrupt,
) (*Response, bool, error) {
	out, err := l.formatWorkflowToolCallInterrupt(ctx, interrupt, cfg)
	return out, err == nil, err
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

func newWorkflowResumeRequestContext(req *Request, state *workflowPendingToolState) workflowResumeRequestContext {
	originalRequest := state.OriginalRequest
	if originalRequest == nil {
		originalRequest = req.OriginalRequest
	}
	resumeReq := *req
	resumeReq.OriginalRequest = originalRequest
	return workflowResumeRequestContext{originalRequest: originalRequest, looperRequest: resumeReq}
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
