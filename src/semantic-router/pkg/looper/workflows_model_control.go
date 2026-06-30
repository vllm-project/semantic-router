package looper

import (
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func applyWorkflowModelReasoningControl(req *openai.ChatCompletionNewParams, modelName string, baseReq *Request) {
	if req == nil || baseReq == nil {
		return
	}
	ref, ok := workflowModelRef(baseReq.ModelRefs, modelName)
	if !ok || ref.UseReasoning == nil {
		return
	}
	parameter := workflowReasoningTemplateParameter(modelName, baseReq)
	if parameter == "" {
		return
	}
	extras := cloneWorkflowPlannerExtraFields(req.ExtraFields())
	kwargs, _ := extras["chat_template_kwargs"].(map[string]any)
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	value, ok := workflowReasoningControlValue(parameter, ref)
	if !ok {
		return
	}
	kwargs[parameter] = value
	extras["chat_template_kwargs"] = kwargs
	req.SetExtraFields(extras)
}

func workflowReasoningControlValue(parameter string, ref config.ModelRef) (any, bool) {
	if parameter != "reasoning_effort" {
		return *ref.UseReasoning, true
	}
	if !*ref.UseReasoning {
		return nil, false
	}
	effort := strings.TrimSpace(ref.ReasoningEffort)
	if effort == "" {
		effort = "medium"
	}
	return effort, true
}

func workflowModelRef(refs []config.ModelRef, modelName string) (config.ModelRef, bool) {
	for _, ref := range refs {
		if ref.Model == modelName {
			return ref, true
		}
	}
	return config.ModelRef{}, false
}

func workflowReasoningTemplateParameter(modelName string, req *Request) string {
	if req != nil && req.ModelParams != nil {
		if params, ok := req.ModelParams[modelName]; ok {
			if parameter := workflowReasoningFamilyParameter(params.ReasoningFamily); parameter != "" {
				return parameter
			}
		}
	}
	return workflowReasoningFamilyParameter(modelName)
}

func workflowReasoningFamilyParameter(value string) string {
	normalized := strings.ToLower(value)
	switch {
	case strings.Contains(normalized, "qwen") || strings.Contains(normalized, "qwq"):
		return "enable_thinking"
	case strings.Contains(normalized, "deepseek"):
		return "thinking"
	case strings.Contains(normalized, "gpt-oss"):
		return "reasoning_effort"
	default:
		return ""
	}
}
