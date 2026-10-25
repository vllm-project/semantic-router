package selection

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// EffectiveCandidateRequest returns a detached text-edit view with deterministic
// decision mutations. Tool schemas and metadata remain immutable shared values.
// The ingress request stays untouched for signals, retention and actual plugins.
func EffectiveCandidateRequest(request *llmprotocol.Request, decision *config.Decision) (*llmprotocol.Request, error) {
	if request == nil {
		return nil, nil
	}
	view := *request
	view.Instructions = append([]llmprotocol.InstructionBlock(nil), request.Instructions...)
	for i := range view.Instructions {
		view.Instructions[i].Content = cloneCandidateContents(request.Instructions[i].Content)
	}
	view.Messages = append([]llmprotocol.Message(nil), request.Messages...)
	for i := range view.Messages {
		view.Messages[i].Content = cloneCandidateContents(request.Messages[i].Content)
	}
	if decision != nil {
		if prompt := decision.GetSystemPromptConfig(); prompt != nil && decision.IsSystemPromptEnabled() {
			llmprotocol.SetSystemInstruction(&view, prompt.SystemPrompt, decision.GetSystemPromptMode())
		}
		if tools := decision.GetToolsConfig(); tools != nil && tools.Enabled && tools.EffectiveMode() == config.ToolsPluginModeNone {
			llmprotocol.StripTools(&view, tools.StripToolHistory)
		}
		if params := decision.GetRequestParamsConfig(); params != nil {
			for _, field := range params.BlockedParams {
				if _, err := llmprotocol.BlockRequestField(&view, strings.TrimSpace(field)); err != nil {
					return nil, err
				}
			}
			if params.DefaultMaxTokens.IsAuto() {
				llmprotocol.DefaultAutomaticOutput(&view)
			} else {
				llmprotocol.DefaultOutputTokens(&view, params.DefaultMaxTokens.Fixed())
			}
			llmprotocol.CapOutputTokens(&view, params.MaxTokensLimit)
			llmprotocol.CapCandidateCount(&view, params.MaxN)
		}
	}
	return &view, nil
}

func cloneCandidateContents(contents []llmprotocol.Content) []llmprotocol.Content {
	cloned := append([]llmprotocol.Content(nil), contents...)
	for i := range cloned {
		if result := cloned[i].ToolResult; result != nil {
			detached := *result
			detached.Content = cloneCandidateContents(result.Content)
			cloned[i].ToolResult = &detached
		}
	}
	return cloned
}
