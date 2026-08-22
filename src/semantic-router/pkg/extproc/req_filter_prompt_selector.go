package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

		request := &openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage(systemPrompt),
				openai.UserMessage(input),
			},
			MaxCompletionTokens: openai.Int(promptSelectorMaxCompletionTokens),
			Temperature:         openai.Float(0),
		}
		jsonObjectFormat := shared.NewResponseFormatJSONObjectParam()
		request.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &jsonObjectFormat,
		}
		disablePromptHelperReasoning(request, model)
		response, err := client.CallModel(
			callCtx,
			request,
			model,
			false,
			1,
			nil,
			"",
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

func disablePromptHelperReasoning(
	request *openai.ChatCompletionNewParams,
	model string,
) {
	normalized := strings.ToLower(model)
	if !strings.Contains(normalized, "qwen") &&
		!strings.Contains(normalized, "qwq") {
		return
	}
	request.SetExtraFields(map[string]interface{}{
		"chat_template_kwargs": map[string]interface{}{
			"enable_thinking": false,
		},
	})
}
