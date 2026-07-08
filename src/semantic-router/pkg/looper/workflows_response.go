package looper

import (
	"encoding/json"
	"fmt"
	"time"
)

func buildWorkflowTrace(
	cfg workflowsExecutionConfig,
	workerModels []string,
	plan *workflowPlan,
	stepResults []workflowStepResult,
	failed []FusionFailedModel,
) *workflowTrace {
	trace := &workflowTrace{
		Mode:         cfg.Mode,
		Template:     cfg.Template,
		PlannerModel: cfg.PlannerModel,
		WorkerModels: append([]string(nil), workerModels...),
		FailedModels: failed,
	}
	if cfg.IncludeIntermediateResponses {
		trace.Plan = plan
		trace.Steps = make([]workflowStepTrace, 0, len(stepResults))
		for _, result := range stepResults {
			stepTrace := workflowStepTrace{
				ID:         result.step.ID,
				Role:       result.step.Role,
				Models:     append([]string(nil), result.step.Models...),
				Prompt:     result.step.Prompt,
				AccessList: append([]string(nil), result.step.AccessList...),
			}
			for _, resp := range result.responses {
				if resp == nil {
					continue
				}
				agentID := workflowResponseAgentID(result.step, resp)
				stepTrace.Responses = append(stepTrace.Responses, workflowResponseTrace{
					AgentID:        agentID,
					Model:          resp.Model,
					Content:        resp.Content,
					Reasoning:      resp.ReasoningContent,
					ToolTrajectory: workflowToolTurnTraces(result.toolTrajectories[agentID]),
				})
			}
			trace.Steps = append(trace.Steps, stepTrace)
		}
	}
	return trace
}

func formatWorkflowJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	trace *workflowTrace,
	usage TokenUsage,
	cfg workflowsExecutionConfig,
) (*Response, error) {
	if finalResp.HasToolCalls {
		return formatWorkflowToolCallJSONResponse(finalResp, modelsUsed, iterations, trace, usage, cfg)
	}
	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-flow-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   finalResp.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": finalResp.Content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": usage.Map(),
	}
	if cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 {
		completion["flow"] = trace
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal workflow response: %w", err)
	}
	return &Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "workflows",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, nil
}

func formatWorkflowToolCallJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	trace *workflowTrace,
	usage TokenUsage,
	cfg workflowsExecutionConfig,
) (*Response, error) {
	var completion map[string]interface{}
	if err := json.Unmarshal(finalResp.Raw, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse workflow tool-call response: %w", err)
	}
	completion["id"] = fmt.Sprintf("chatcmpl-flow-%d", time.Now().UnixNano())
	completion["model"] = finalResp.Model
	completion["usage"] = usage.Map()
	if cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 {
		completion["flow"] = trace
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal workflow tool-call response: %w", err)
	}
	return &Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "workflows",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, nil
}

func formatWorkflowStreamingResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	trace *workflowTrace,
	usage TokenUsage,
	cfg workflowsExecutionConfig,
) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-flow-%d", timestamp)
	var (
		body []byte
		err  error
	)
	if finalResp.HasToolCalls {
		body, err = buildWorkflowStreamingToolCallSSE(id, timestamp, finalResp.Model, finalResp.Raw, trace, cfg)
		if err != nil {
			return nil, err
		}
	} else {
		body = buildWorkflowStreamingSSE(id, timestamp, finalResp.Model, finalResp.Content, trace, cfg)
	}
	resp := streamingLooperResponse(body, finalResp.Model, modelsUsed, iterations, "workflows")
	resp.IntermediateResponses = trace
	resp.Usage = usage
	return resp, nil
}

func buildWorkflowStreamingToolCallSSE(
	id string,
	created int64,
	model string,
	raw []byte,
	trace *workflowTrace,
	cfg workflowsExecutionConfig,
) ([]byte, error) {
	toolCalls, err := fusionToolCallDeltasFromRaw(raw)
	if err != nil {
		return nil, fmt.Errorf("failed to parse workflow streaming tool-call response: %w", err)
	}

	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	var extra map[string]interface{}
	if cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 {
		extra = map[string]interface{}{"flow": trace}
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, extra))
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"tool_calls": toolCalls},
		"finish_reason": nil,
	}, nil))
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": "tool_calls",
	}, nil))
	return appendSSEDone(body), nil
}

func buildWorkflowStreamingSSE(
	id string,
	created int64,
	model string,
	content string,
	trace *workflowTrace,
	cfg workflowsExecutionConfig,
) []byte {
	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	var extra map[string]interface{}
	if cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 {
		extra = map[string]interface{}{"flow": trace}
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, extra))
	for _, chunk := range splitIntoChunks(content, 50) {
		body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, map[string]interface{}{
			"index":         0,
			"delta":         map[string]interface{}{"content": chunk},
			"finish_reason": nil,
		}, nil))
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": "stop",
	}, nil))
	return appendSSEDone(body)
}
