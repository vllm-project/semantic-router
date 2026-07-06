package looper

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (l *WorkflowsLooper) generateDynamicWorkflowPlan(
	ctx context.Context,
	req *Request,
	cfg workflowsExecutionConfig,
	original string,
	workerModels []string,
) (*workflowPlan, *ModelResponse, error) {
	if cfg.PlannerModel == "" {
		return nil, nil, fmt.Errorf("workflows dynamic mode requires planner.model")
	}
	plannerOriginal := requestTextWithOutputContract(original, req.OriginalRequest, req.OutputContract)
	prompt := buildWorkflowPlannerPrompt(plannerOriginal, workerModels, cfg, req.OutputContractSpec)
	planReq := appendFusionStageMessage(stripFusionToolUse(req.OriginalRequest), prompt)
	configureWorkflowPlannerRequest(planReq, cfg.PlannerModel)
	resp, err := l.callWorkflowModel(ctx, planReq, cfg, cfg.PlannerModel, false, 1, req)
	if err != nil {
		return nil, resp, fmt.Errorf("workflow planner %q failed: %w", cfg.PlannerModel, err)
	}
	plan, err := parseWorkflowPlanFromResponse(resp)
	if err != nil {
		if shouldUseDynamicWorkflowFallback(cfg) {
			logging.Warnf("[Workflows] Planner %q returned invalid JSON plan (%v); using fallback workflow because on_error=skip", cfg.PlannerModel, err)
			return buildDynamicWorkflowFallbackPlan(workerModels, cfg), resp, nil
		}
		return nil, resp, fmt.Errorf("workflow planner %q returned invalid plan: %w", cfg.PlannerModel, err)
	}
	return plan, resp, nil
}

func shouldUseDynamicWorkflowFallback(cfg workflowsExecutionConfig) bool {
	return cfg.Mode == config.WorkflowModeDynamic && cfg.OnError == config.WorkflowOnErrorSkip
}

func buildDynamicWorkflowFallbackPlan(workerModels []string, cfg workflowsExecutionConfig) *workflowPlan {
	models := append([]string(nil), workerModels...)
	if cfg.MaxParallel > 0 && len(models) > cfg.MaxParallel {
		models = models[:cfg.MaxParallel]
	}
	return &workflowPlan{
		Steps: []workflowPlanStep{{
			ID:     "fallback_solve",
			Role:   "worker",
			Models: models,
			Prompt: "Solve the original user request directly. Preserve any required output format exactly.",
		}},
		Final: &workflowFinalStep{
			Prompt: "Synthesize the worker answer into the final response. Preserve any constrained output format exactly.",
		},
	}
}

func buildWorkflowPlannerPrompt(original string, workerModels []string, cfg workflowsExecutionConfig, outputContractSpec *config.OutputContractSpec) string {
	choicePlanningRule := ""
	if requestsSingleChoice(outputContractSpec) {
		choicePlanningRule = `
- For multiple-choice benchmark prompts or requests that require a final
  answer choice, prefer one parallel independent-solver step using the
  strongest available workers, followed by final synthesis that checks the
  solver outputs against the original question. The final response must preserve
  the requested answer format exactly.`
	}
	return fmt.Sprintf(`You are the Router Flow planner. Create a compact execution plan for a bounded multi-agent workflow.

Return only valid JSON. Do not include markdown, prose, XML tags, or chain-of-thought.
The first non-whitespace character must be "{" and the last non-whitespace character must be "}".

Available worker models, and the only worker models you may use:
%s

Limits:
- steps: 1 to %d
- models per step: 1 to %d
- every step model must exactly match one available worker model

Planning rules:
- Preserve the user's requested deliverable. If the user asks to design,
  explain, diagnose, or write code, plan work that produces that deliverable;
  do not claim that actions were already executed.
- A step may include "access_list" with earlier step ids whose outputs it may
  read. It may also include earlier agent ids using the format
  "<step-id>:<model-index>:<model-name>" when only one worker output should be
  exposed from a parallel step. Omit access_list to read all earlier step
  outputs. Use [] to isolate a step from prior outputs.
- For coding, debugging, math, or contested reasoning tasks, include a focused
  verification or synthesis instruction unless the task is trivial.
%s
- The final synthesis instruction should tell the final model to check the
  worker outputs against the original request and correct contradictions.

JSON schema:
{
  "steps": [
    {
      "id": "short-id",
      "role": "thinker|worker|verifier",
      "models": ["one-or-more-available-worker-models"],
      "prompt": "focused instruction for this step",
      "access_list": ["optional-earlier-step-or-agent-id"]
    }
  ],
  "final": {
    "prompt": "instruction for final synthesis"
  }
}

Original user request:
%s`, strings.Join(workerModels, "\n"), cfg.MaxSteps, cfg.MaxParallel, choicePlanningRule, original)
}

func configureWorkflowPlannerRequest(req *openai.ChatCompletionNewParams, plannerModel string) {
	if req == nil {
		return
	}
	jsonObjectFormat := shared.NewResponseFormatJSONObjectParam()
	req.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONObject: &jsonObjectFormat}

	applyWorkflowChatTemplateKwargs(req, workflowPlannerChatTemplateKwargs(plannerModel))
}

func applyWorkflowChatTemplateKwargs(req *openai.ChatCompletionNewParams, kwargs map[string]any) {
	if req == nil || len(kwargs) == 0 {
		return
	}
	extras := cloneWorkflowPlannerExtraFields(req.ExtraFields())
	if existing, ok := extras["chat_template_kwargs"].(map[string]any); ok {
		for key, value := range kwargs {
			existing[key] = value
		}
		extras["chat_template_kwargs"] = existing
	} else {
		extras["chat_template_kwargs"] = kwargs
	}
	req.SetExtraFields(extras)
}

func cloneWorkflowPlannerExtraFields(fields map[string]any) map[string]any {
	cloned := make(map[string]any, len(fields)+1)
	for key, value := range fields {
		cloned[key] = value
	}
	return cloned
}

func workflowPlannerChatTemplateKwargs(plannerModel string) map[string]any {
	normalized := strings.ToLower(plannerModel)
	kwargs := map[string]any{}
	if strings.Contains(normalized, "qwen") || strings.Contains(normalized, "qwq") {
		kwargs["enable_thinking"] = false
	}
	if strings.Contains(normalized, "deepseek") {
		kwargs["thinking"] = false
	}
	return kwargs
}
