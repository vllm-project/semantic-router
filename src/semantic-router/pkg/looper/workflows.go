package looper

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type WorkflowsLooper struct {
	*BaseLooper
	toolStates workflowToolStateStore
}

func NewWorkflowsLooper(cfg *config.LooperConfig) *WorkflowsLooper {
	return &WorkflowsLooper{
		BaseLooper: NewBaseLooper(cfg),
		toolStates: newWorkflowToolStateStoreFromConfig(
			workflowFlowRuntimeConfig(cfg),
		),
	}
}

func workflowFlowRuntimeConfig(cfg *config.LooperConfig) config.FlowRuntimeConfig {
	if cfg == nil {
		return config.FlowRuntimeConfig{}
	}
	return cfg.Flow
}

type workflowsExecutionConfig struct {
	Mode                         string
	Template                     string
	PlannerModel                 string
	PlannerMaxCompletionTokens   int
	Roles                        []config.WorkflowRoleConfig
	Final                        config.WorkflowFinalConfig
	MaxSteps                     int
	MaxParallel                  int
	MaxCompletionTokens          int
	RoundTimeoutSeconds          int
	MinSuccessfulResponses       int
	Temperature                  *float64
	IncludeIntermediateResponses bool
	OnError                      string
}

type workflowPlan struct {
	Steps []workflowPlanStep `json:"steps"`
	Final *workflowFinalStep `json:"final,omitempty"`
}

type workflowPlanStep struct {
	ID         string   `json:"id,omitempty"`
	Role       string   `json:"role,omitempty"`
	Models     []string `json:"models,omitempty"`
	Prompt     string   `json:"prompt,omitempty"`
	AccessList []string `json:"access_list,omitempty"`
}

type workflowFinalStep struct {
	Model  string `json:"model,omitempty"`
	Prompt string `json:"prompt,omitempty"`
}

type workflowStepTrace struct {
	ID         string                  `json:"id,omitempty"`
	Role       string                  `json:"role,omitempty"`
	Models     []string                `json:"models,omitempty"`
	Prompt     string                  `json:"prompt,omitempty"`
	AccessList []string                `json:"access_list,omitempty"`
	Responses  []workflowResponseTrace `json:"responses,omitempty"`
}

type workflowResponseTrace struct {
	AgentID        string                  `json:"agent_id,omitempty"`
	Model          string                  `json:"model"`
	Content        string                  `json:"content"`
	Reasoning      string                  `json:"reasoning,omitempty"`
	ToolTrajectory []workflowToolTurnTrace `json:"tool_trajectory,omitempty"`
}

type workflowToolTurnTrace struct {
	ToolCallIDs []string `json:"tool_call_ids,omitempty"`
}

type workflowTrace struct {
	Mode                string                        `json:"mode"`
	Template            string                        `json:"template,omitempty"`
	PlannerModel        string                        `json:"planner_model,omitempty"`
	WorkerModels        []string                      `json:"worker_models,omitempty"`
	Plan                *workflowPlan                 `json:"plan,omitempty"`
	Steps               []workflowStepTrace           `json:"steps,omitempty"`
	FinalToolTrajectory []workflowToolTurnTrace       `json:"final_tool_trajectory,omitempty"`
	PendingToolCall     *workflowPendingToolCallTrace `json:"pending_tool_call,omitempty"`
	FailedModels        []FusionFailedModel           `json:"failed_models,omitempty"`
}

type workflowStepResult struct {
	step             workflowPlanStep
	responses        []*ModelResponse
	failed           []FusionFailedModel
	toolTrajectories map[string][]workflowAgentToolTurn
}

type workflowModelResult struct {
	index int
	model string
	resp  *ModelResponse
	err   error
}

type workflowToolCallInterrupt struct {
	resp  *ModelResponse
	state *workflowPendingToolState
}

type workflowExecutionSummary struct {
	usage      TokenUsage
	modelsUsed []string
	failed     []FusionFailedModel
	iterations int
}

func (l *WorkflowsLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	l.client.SetDecisionName(req.DecisionName)

	cfg := resolveWorkflowsExecutionConfig(req)
	if len(req.ModelRefs) == 0 {
		return nil, fmt.Errorf("workflows requires decision modelRefs")
	}
	if err := l.validateWorkflowControlModels(cfg); err != nil {
		return nil, err
	}

	workerModels := modelRefsToNames(req.ModelRefs)
	original := extractOriginalContent(req.OriginalRequest)
	if stateID, ok := findWorkflowToolStateID(req.OriginalRequest); ok {
		return l.resumeWorkflowToolCall(ctx, req, cfg, workerModels, stateID)
	}
	logging.ComponentEvent("looper", "workflows_execution_started", map[string]interface{}{
		"decision":     req.DecisionName,
		"mode":         cfg.Mode,
		"planner":      cfg.PlannerModel,
		"workers":      len(workerModels),
		"max_steps":    cfg.MaxSteps,
		"max_parallel": cfg.MaxParallel,
		"streaming":    req.IsStreaming,
	})

	plan, plannerResp, err := l.buildWorkflowPlan(ctx, req, cfg, original, workerModels)
	if err != nil {
		return nil, err
	}

	stepResults, interrupt, err := l.executeWorkflowSteps(ctx, req, cfg, plan, plannerResp, workerModels)
	if err != nil {
		return nil, err
	}
	if interrupt != nil {
		return l.formatWorkflowToolCallInterrupt(ctx, interrupt, cfg)
	}

	finalResp, interrupt, err := l.synthesizeWorkflowFinal(ctx, req, cfg, plan, original, stepResults, plannerResp, workerModels)
	if err != nil {
		if cfg.OnError != config.WorkflowOnErrorSkip {
			return nil, err
		}
		finalResp = workflowFallbackFinalResponse(
			req.OutputContractSpec,
			stepResults,
		)
		if finalResp == nil {
			return nil, err
		}
		logging.Warnf("[Workflows] Final synthesis failed; using worker response fallback because on_error=skip: %v", err)
	}
	if interrupt != nil {
		return l.formatWorkflowToolCallInterrupt(ctx, interrupt, cfg)
	}
	applyJSONActionOutputContract(req.OutputContractSpec, finalResp, workflowStepModelResponses(stepResults))
	applyFinalOutputContract(req.OutputContractSpec, finalResp)
	applyWorkflowSingleChoiceFallback(req.OutputContractSpec, finalResp, stepResults)
	applyFinalOutputContract(req.OutputContractSpec, finalResp)

	summary := summarizeWorkflowExecution(cfg, plannerResp, stepResults, finalResp)
	trace := buildWorkflowTrace(cfg, workerModels, plan, stepResults, summary.failed)
	if req.IsStreaming {
		return formatWorkflowStreamingResponse(finalResp, summary.modelsUsed, summary.iterations, trace, summary.usage, cfg)
	}
	return formatWorkflowJSONResponse(finalResp, summary.modelsUsed, summary.iterations, trace, summary.usage, cfg)
}

func (l *WorkflowsLooper) buildWorkflowPlan(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	original string,
	workerModels []string,
) (*workflowPlan, *ModelResponse, error) {
	var plan *workflowPlan
	var plannerResp *ModelResponse
	var err error
	if cfg.Mode == config.WorkflowModeDynamic {
		plan, plannerResp, err = l.generateDynamicWorkflowPlan(ctx, req, cfg, original, workerModels)
	} else {
		plan, err = buildStaticWorkflowPlan(cfg)
	}
	if err != nil {
		return nil, nil, err
	}
	applyConfiguredWorkflowFinal(plan, cfg)
	if validateErr := validateWorkflowPlan(plan, workerModels, cfg); validateErr != nil {
		if shouldUseDynamicWorkflowFallback(cfg) {
			logging.Warnf("[Workflows] Planner returned invalid workflow plan (%v); using fallback workflow because on_error=skip", validateErr)
			fallbackPlan := buildDynamicWorkflowFallbackPlan(workerModels, cfg)
			applyConfiguredWorkflowFinal(fallbackPlan, cfg)
			if fallbackErr := validateWorkflowPlan(fallbackPlan, workerModels, cfg); fallbackErr != nil {
				return nil, nil, fallbackErr
			}
			return fallbackPlan, plannerResp, nil
		}
		return nil, nil, validateErr
	}
	return plan, plannerResp, nil
}

func applyConfiguredWorkflowFinal(plan *workflowPlan, cfg workflowsExecutionConfig) {
	if plan == nil || cfg.Final.IsZero() {
		return
	}
	if plan.Final == nil {
		plan.Final = &workflowFinalStep{}
	}
	if strings.TrimSpace(cfg.Final.Model) != "" {
		plan.Final.Model = strings.TrimSpace(cfg.Final.Model)
	}
	if strings.TrimSpace(cfg.Final.Prompt) != "" {
		plan.Final.Prompt = strings.TrimSpace(cfg.Final.Prompt)
	}
}

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
		prompt := buildWorkflowStepPrompt(req.OriginalRequest, step, previous)
		stepReq := appendFusionStageMessage(req.OriginalRequest, prompt)
		responses, failed, interrupt, err := l.executeWorkflowStep(ctx, req, cfg, plan, step, stepReq, idx, 0, idx+2)
		if err != nil {
			return nil, nil, err
		}
		if interrupt != nil {
			interrupt.state.Plan = plan
			interrupt.state.PlannerResp = plannerResp
			interrupt.state.WorkerModels = append([]string(nil), workerModels...)
			interrupt.state.StepResults = append([]workflowStepResult(nil), previous...)
			return nil, interrupt, nil
		}
		result := workflowStepResult{step: step, responses: responses, failed: failed}
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
) ([]*ModelResponse, []FusionFailedModel, *workflowToolCallInterrupt, error) {
	if requestHasTools(stepReq) {
		return l.executeWorkflowStepSequential(ctx, req, cfg, plan, step, stepReq, stepIndex, modelStartIndex, iterationStart)
	}
	stepCtx, cancel := workflowRoundContext(ctx, cfg)
	defer cancel()
	models := step.Models[modelStartIndex:]
	results := l.startWorkflowStepWorkers(stepCtx, req, cfg, stepReq, models, modelStartIndex, iterationStart)

	collector := newWorkflowStepCollector(step, cfg, len(models), cancel)
	for range models {
		select {
		case result := <-results:
			responses, err, done := collector.handleResult(result)
			if done {
				return responses, collector.failed, nil, err
			}
		case <-stepCtx.Done():
			responses, err := collector.handleTimeout(stepCtx.Err())
			return responses, collector.failed, nil, err
		}
	}

	responses, err := collector.finalize()
	return responses, collector.failed, nil, err
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
	failed        []FusionFailedModel
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

func (c *workflowStepCollector) handleResult(result workflowModelResult) ([]*ModelResponse, error, bool) {
	if result.err != nil {
		c.failed = append(c.failed, FusionFailedModel{Model: result.model, Error: result.err.Error()})
		if c.cfg.OnError == config.WorkflowOnErrorFail {
			return nil, fmt.Errorf("workflow step %q failed for model %q: %w", c.step.ID, result.model, result.err), true
		}
		return nil, nil, false
	}
	c.ordered[result.index] = result.resp
	responses := c.responses()
	if len(responses) < c.minSuccessful {
		return nil, nil, false
	}
	c.cancel()
	return responses, nil, true
}

func (c *workflowStepCollector) handleTimeout(err error) ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) > 0 && c.cfg.OnError != config.WorkflowOnErrorFail {
		logging.Warnf("[Workflows] Step %q timed out with %d partial responses; continuing because on_error=skip", c.step.ID, len(responses))
		return responses, nil
	}
	return nil, err
}

func (c *workflowStepCollector) finalize() ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) == 0 {
		return nil, fmt.Errorf("workflow step %q failed: all models failed", c.step.ID)
	}
	return responses, nil
}

func (c *workflowStepCollector) responses() []*ModelResponse {
	return workflowResponsesFromOrdered(c.ordered)
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
				return nil, failed, nil, fmt.Errorf("workflow step %q failed for model %q: %w", step.ID, modelName, err)
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
					OriginalRequest:      cloneRequest(req.OriginalRequest),
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

func (l *WorkflowsLooper) synthesizeWorkflowFinal(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	plan *workflowPlan,
	original string,
	stepResults []workflowStepResult,
	plannerResp *ModelResponse,
	workerModels []string,
) (*ModelResponse, *workflowToolCallInterrupt, error) {
	modelName, err := resolveWorkflowFinalModel(cfg, plan, stepResults)
	if err != nil {
		return nil, nil, err
	}
	outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
	prompt := buildWorkflowFinalPrompt(plan, original, outputContract, stepResults)
	finalReq := appendFusionStageMessage(req.OriginalRequest, prompt)
	finalCtx, cancel := workflowRoundContext(ctx, cfg)
	defer cancel()
	resp, err := l.callWorkflowModel(finalCtx, finalReq, cfg, modelName, true, workflowFinalIteration(stepResults), req)
	if err != nil {
		return nil, nil, fmt.Errorf("workflow final synthesis failed for model %q: %w", modelName, err)
	}
	if resp.HasToolCalls {
		return nil, &workflowToolCallInterrupt{
			resp: resp,
			state: &workflowPendingToolState{
				DecisionName:    req.DecisionName,
				Mode:            cfg.Mode,
				Template:        cfg.Template,
				Plan:            plan,
				PlannerResp:     plannerResp,
				WorkerModels:    append([]string(nil), workerModels...),
				StepResults:     append([]workflowStepResult(nil), stepResults...),
				OriginalRequest: cloneRequest(req.OriginalRequest),
				Phase:           workflowToolPhaseFinal,
				AgentID:         workflowAgentID(workflowToolPhaseFinal, workflowPlanStep{ID: "final", Role: "final"}, modelName, 0),
				StepID:          "final",
				Role:            "final",
				StepIndex:       len(plan.Steps),
				ModelIndex:      0,
				Model:           modelName,
				StepRequest:     cloneRequest(finalReq),
				AgentRequest:    cloneRequest(finalReq),
				Iteration:       workflowFinalIteration(stepResults),
				Streaming:       req.IsStreaming,
			},
		}, nil
	}
	return resp, nil, nil
}

func resolveWorkflowFinalModel(cfg workflowsExecutionConfig, plan *workflowPlan, stepResults []workflowStepResult) (string, error) {
	modelName := cfg.PlannerModel
	if plan != nil && plan.Final != nil && strings.TrimSpace(plan.Final.Model) != "" {
		modelName = strings.TrimSpace(plan.Final.Model)
	} else if strings.TrimSpace(cfg.Final.Model) != "" {
		modelName = strings.TrimSpace(cfg.Final.Model)
	}
	if modelName == "" {
		modelName = firstWorkflowResponseModel(stepResults)
	}
	if modelName == "" {
		return "", fmt.Errorf("workflows could not resolve final synthesis model")
	}
	return modelName, nil
}

func workflowStepModelResponses(stepResults []workflowStepResult) []*ModelResponse {
	var responses []*ModelResponse
	for _, step := range stepResults {
		responses = append(responses, step.responses...)
	}
	return responses
}

func buildWorkflowFinalPrompt(plan *workflowPlan, original string, outputContract string, stepResults []workflowStepResult) string {
	instruction := "Synthesize the workflow outputs into the best final answer for the user."
	if plan != nil && plan.Final != nil && strings.TrimSpace(plan.Final.Prompt) != "" {
		instruction = strings.TrimSpace(plan.Final.Prompt)
	}
	prompt := fmt.Sprintf(`You are the Router Flow final synthesizer.

Instruction:
%s

	Rules:
	- Answer the original user request directly; do not describe internal workflow execution.
	- Treat workflow outputs as evidence, not authority. Check math, code, and logic
	  against the original request and correct contradictions.
	- Preserve any constrained output format exactly. If the user asks for only a
	  letter, option, JSON object, code block, patch, or other strict format, output
	  only that format.
	- Do not reveal hidden reasoning, scratch work, panel reasoning, tool traces,
	  workflow traces, or internal deliberation. Provide a concise explanation only
	  when the original output contract asks for one.
	- Preserve requested deliverables such as code, tests, diagnosis, interval
	  reasoning, or a workflow design.
- If a worker output is truncated or incomplete, complete the answer from the
  original request and the usable evidence.
- Do not mention internal model names or workflow steps unless the user asks.

Original user request:
%s

Workflow outputs:
%s

Final answer:`, instruction, original, formatWorkflowStepResults(stepResults))

	return appendOutputContractForPrompt(prompt, outputContract)
}

func formatWorkflowStepResults(stepResults []workflowStepResult) string {
	var b strings.Builder
	for _, result := range stepResults {
		fmt.Fprintf(&b, "Step %s (%s):\n", result.step.ID, result.step.Role)
		for _, resp := range result.responses {
			if resp == nil {
				continue
			}
			fmt.Fprintf(&b, "- %s [%s]: %s\n", workflowResponseAgentID(result.step, resp), resp.Model, resp.Content)
			if resp.ReasoningContent != "" {
				fmt.Fprintf(&b, "  reasoning: %s\n", resp.ReasoningContent)
			}
		}
	}
	return strings.TrimSpace(b.String())
}

func firstWorkflowResponseModel(stepResults []workflowStepResult) string {
	for _, result := range stepResults {
		for _, resp := range result.responses {
			if resp != nil && resp.Model != "" {
				return resp.Model
			}
		}
	}
	return ""
}

func workflowRoundContext(ctx context.Context, cfg workflowsExecutionConfig) (context.Context, context.CancelFunc) {
	if cfg.RoundTimeoutSeconds <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, time.Duration(cfg.RoundTimeoutSeconds)*time.Second)
}

func workflowRoundMinSuccessful(numCalls int, configured int) int {
	if configured > 0 && configured < numCalls {
		return configured
	}
	return numCalls
}

func workflowResponsesFromOrdered(ordered []*ModelResponse) []*ModelResponse {
	responses := make([]*ModelResponse, 0, len(ordered))
	for _, resp := range ordered {
		if resp != nil {
			responses = append(responses, resp)
		}
	}
	return responses
}

func workflowFallbackFinalResponse(spec *config.OutputContractSpec, stepResults []workflowStepResult) *ModelResponse {
	if requestsSingleChoice(spec) {
		if resp := workflowSingleChoiceFallbackResponse(stepResults, spec); resp != nil {
			return resp
		}
	}
	for i := len(stepResults) - 1; i >= 0; i-- {
		responses := stepResults[i].responses
		for j := len(responses) - 1; j >= 0; j-- {
			resp := responses[j]
			if resp == nil || (strings.TrimSpace(resp.Content) == "" && strings.TrimSpace(resp.ReasoningContent) == "") {
				continue
			}
			fallback := *resp
			return &fallback
		}
	}
	return nil
}

func workflowSingleChoiceFallbackResponse(stepResults []workflowStepResult, spec *config.OutputContractSpec) *ModelResponse {
	answer, ok := workflowMajoritySingleChoiceAnswer(stepResults, spec)
	if !ok {
		return nil
	}
	for _, result := range stepResults {
		for _, resp := range result.responses {
			if resp == nil {
				continue
			}
			respAnswer, ok := extractSingleChoiceAnswer(resp.Content, spec)
			if !ok && strings.TrimSpace(resp.Content) == "" {
				respAnswer, ok = extractSingleChoiceAnswer(resp.ReasoningContent, spec)
			}
			if ok && respAnswer == answer {
				fallback := *resp
				fallback.Content = renderSingleChoiceAnswer(answer, spec)
				fallback.ReasoningContent = ""
				fallback.HasToolCalls = false
				return &fallback
			}
		}
	}
	return &ModelResponse{Model: firstWorkflowResponseModel(stepResults), Content: renderSingleChoiceAnswer(answer, spec)}
}

func workflowFinalIteration(stepResults []workflowStepResult) int {
	iteration := 2
	for _, result := range stepResults {
		iteration += len(result.responses) + len(result.failed)
	}
	return iteration
}

func (l *WorkflowsLooper) callWorkflowModel(
	ctx context.Context,
	req *openai.ChatCompletionNewParams,
	cfg workflowsExecutionConfig,
	modelName string,
	allowTools bool,
	iteration int,
	baseReq *Request,
) (*ModelResponse, error) {
	callReq := cloneRequest(req)
	if !allowTools {
		callReq = stripFusionToolUse(callReq)
	}
	if modelName == cfg.PlannerModel {
		applyWorkflowChatTemplateKwargs(callReq, workflowPlannerChatTemplateKwargs(modelName))
	}
	applyWorkflowModelReasoningControl(callReq, modelName, baseReq)
	if cfg.Temperature != nil {
		callReq.Temperature = openai.Float(*cfg.Temperature)
	}
	if cfg.MaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
	}
	if modelName == cfg.PlannerModel && cfg.PlannerMaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(cfg.PlannerMaxCompletionTokens))
	}
	return l.client.CallModel(ctx, callReq, modelName, false, iteration, nil, accessKeyForModel(baseReq, modelName))
}
