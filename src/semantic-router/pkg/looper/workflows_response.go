package looper

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	_ = cfg
	semantic := newTextSemanticResponse("response-workflow", finalResp.Model, finalResp.Content, usage)
	return newLooperResponse(semantic, false, true, finalResp.Model, modelsUsed, iterations, "workflows", usage, trace), nil
}

func formatWorkflowToolCallJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	trace *workflowTrace,
	usage TokenUsage,
	cfg workflowsExecutionConfig,
) (*Response, error) {
	_ = cfg
	semantic, err := newModelSemanticResponse("response-workflow", finalResp, finalResp.Model, usage)
	if err != nil {
		return nil, fmt.Errorf("build neutral workflow tool-call response: %w", err)
	}
	return newLooperResponse(semantic, false, true, finalResp.Model, modelsUsed, iterations, "workflows", usage, trace), nil
}

func formatWorkflowStreamingResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	trace *workflowTrace,
	usage TokenUsage,
	cfg workflowsExecutionConfig,
	includeUsage bool,
) (*Response, error) {
	_ = cfg
	var semantic llmprotocol.Response
	if finalResp.HasToolCalls {
		var err error
		semantic, err = newModelSemanticResponse("response-workflow", finalResp, finalResp.Model, usage)
		if err != nil {
			return nil, err
		}
	} else {
		semantic = newTextSemanticResponse("response-workflow", finalResp.Model, finalResp.Content, usage)
	}
	return newLooperResponse(semantic, true, includeUsage, finalResp.Model, modelsUsed, iterations, "workflows", usage, trace), nil
}
