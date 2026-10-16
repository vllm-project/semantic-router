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
				"finish_reason": workflowFinalFinishReason(finalResp),
			},
		},
		"usage": usage.Map(),
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal workflow response: %w", err)
	}
	return withWorkflowExtension(&Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "workflows",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, cfg, trace)
}

func workflowFinalFinishReason(resp *ModelResponse) string {
	if resp.Parsed != nil && len(resp.Parsed.Choices) > 0 && resp.Parsed.Choices[0].FinishReason != "" {
		return resp.Parsed.Choices[0].FinishReason
	}
	// Locally assembled responses may not carry a backend completion.
	return "stop"
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
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal workflow tool-call response: %w", err)
	}
	return withWorkflowExtension(&Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "workflows",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, cfg, trace)
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
		body, err = buildWorkflowStreamingToolCallSSE(id, timestamp, finalResp.Model, finalResp.Raw)
		if err != nil {
			return nil, err
		}
	} else {
		body = buildWorkflowStreamingSSE(id, timestamp, finalResp)
	}
	resp := streamingLooperResponse(body, finalResp.Model, modelsUsed, iterations, "workflows")
	resp.IntermediateResponses = trace
	resp.Usage = usage
	return withWorkflowExtension(resp, cfg, trace)
}

func buildWorkflowStreamingToolCallSSE(
	id string,
	created int64,
	model string,
	raw []byte,
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
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, nil))
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
	finalResp *ModelResponse,
) []byte {
	model := finalResp.Model
	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, nil))
	for _, chunk := range splitIntoChunks(finalResp.Content, 50) {
		body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, map[string]interface{}{
			"index":         0,
			"delta":         map[string]interface{}{"content": chunk},
			"finish_reason": nil,
		}, nil))
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": workflowFinalFinishReason(finalResp),
	}, nil))
	return appendSSEDone(body)
}
