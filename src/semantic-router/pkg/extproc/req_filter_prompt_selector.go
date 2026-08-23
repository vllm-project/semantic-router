package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const promptSelectorMaxCompletionTokens = 512

func (r *OpenAIRouter) newDecisionPromptSelector(
	cfg config.PromptSelectionConfig,
) selection.Selector {
	descriptions := make(map[string]string, len(r.Config.ModelConfig))
	for model, params := range r.Config.ModelConfig {
		descriptions[model] = params.Description
	}
	client := looper.NewClient(&r.Config.Looper)

	invoke := func(
		ctx context.Context,
		model string,
		systemPrompt string,
		input string,
	) (selection.PromptInvocationResult, error) {
		timeout := cfg.TimeoutSeconds
		if timeout <= 0 {
			timeout = 5
		}
		callCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
		defer cancel()

		maxOutput := int64(promptSelectorMaxCompletionTokens)
		temperature := float64(0)
		request := &llmprotocol.Request{
			Generation: 1,
			Model:      model,
			Instructions: []llmprotocol.InstructionBlock{{
				Role:    llmprotocol.RoleSystem,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: systemPrompt}},
			}},
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: input}},
			}},
			Sampling: llmprotocol.Sampling{
				MaxOutputTokens: &maxOutput,
				Temperature:     &temperature,
			},
			OutputFormat: llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONObject},
		}
		response, err := client.CallSemanticModel(
			callCtx,
			request,
			model,
			false,
			1,
			nil,
		)
		if err != nil {
			return selection.PromptInvocationResult{}, err
		}
		return selection.PromptInvocationResult{
			Content:          response.Content,
			Model:            model,
			PromptTokens:     response.Usage.PromptTokens,
			CompletionTokens: response.Usage.CompletionTokens,
			TotalTokens:      response.Usage.TotalTokens,
			LatencyMs:        response.LatencyMs,
		}, nil
	}
	return selection.NewPromptSelector(cfg, invoke, descriptions)
}
