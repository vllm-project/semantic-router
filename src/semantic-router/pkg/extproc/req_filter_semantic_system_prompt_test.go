package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAddSemanticSystemPromptIfConfigured(t *testing.T) {
	router := &OpenAIRouter{}

	t.Run("insert preserves ordered structured instructions", func(t *testing.T) {
		decision := semanticSystemPromptDecision("support", "Treat inputs as confidential.", "insert", nil)
		request := &llmprotocol.Request{Instructions: []llmprotocol.InstructionBlock{{
			Role: llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "Existing instruction."},
				{Kind: llmprotocol.ContentFile, FileID: "policy-file"},
			},
		}}}
		requestContext := &RequestContext{
			TraceContext: context.Background(), VSRSelectedDecision: &decision,
		}

		changed, err := router.addSemanticSystemPromptIfConfigured(
			request, decision.Name, "test-model", requestContext,
		)
		if err != nil {
			t.Fatalf("add semantic system prompt: %v", err)
		}
		if !changed || !requestContext.VSRInjectedSystemPrompt {
			t.Fatal("configured system prompt was not recorded as injected")
		}
		content := request.Instructions[0].Content
		if len(content) != 3 || content[0].Text != "Treat inputs as confidential." ||
			content[1].Text != "Existing instruction." || content[2].FileID != "policy-file" {
			t.Fatalf("ordered instruction content = %#v", content)
		}
	})

	t.Run("replace removes prior system content only", func(t *testing.T) {
		decision := semanticSystemPromptDecision("support", "Use the approved policy.", "replace", nil)
		request := &llmprotocol.Request{Instructions: []llmprotocol.InstructionBlock{
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "old"}}},
			{Role: llmprotocol.RoleDeveloper, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "keep"}}},
		}}
		requestContext := &RequestContext{
			TraceContext: context.Background(), VSRSelectedDecision: &decision,
		}

		changed, err := router.addSemanticSystemPromptIfConfigured(
			request, decision.Name, "test-model", requestContext,
		)
		if err != nil || !changed {
			t.Fatalf("replace system prompt: changed=%t err=%v", changed, err)
		}
		if len(request.Instructions) != 2 || len(request.Instructions[0].Content) != 1 ||
			request.Instructions[0].Content[0].Text != "Use the approved policy." ||
			request.Instructions[1].Role != llmprotocol.RoleDeveloper ||
			request.Instructions[1].Content[0].Text != "keep" {
			t.Fatalf("replaced instructions = %#v", request.Instructions)
		}
	})

	t.Run("missing system instruction is prepended", func(t *testing.T) {
		decision := semanticSystemPromptDecision("support", "Lead with the answer.", "insert", nil)
		request := testNeutralRequest("test-model", "hello")
		requestContext := &RequestContext{
			TraceContext: context.Background(), VSRSelectedDecision: &decision,
		}

		changed, err := router.addSemanticSystemPromptIfConfigured(
			request, decision.Name, "test-model", requestContext,
		)
		if err != nil || !changed {
			t.Fatalf("prepend system prompt: changed=%t err=%v", changed, err)
		}
		if len(request.Instructions) != 1 || request.Instructions[0].Role != llmprotocol.RoleSystem ||
			request.Instructions[0].Content[0].Text != "Lead with the answer." {
			t.Fatalf("prepended instructions = %#v", request.Instructions)
		}
	})

	t.Run("disabled plugin leaves neutral request unchanged", func(t *testing.T) {
		disabled := false
		decision := semanticSystemPromptDecision("support", "Do not add me.", "insert", &disabled)
		request := testNeutralRequest("test-model", "hello")
		requestContext := &RequestContext{
			TraceContext: context.Background(), VSRSelectedDecision: &decision,
		}

		changed, err := router.addSemanticSystemPromptIfConfigured(
			request, decision.Name, "test-model", requestContext,
		)
		if err != nil || changed || len(request.Instructions) != 0 || requestContext.VSRInjectedSystemPrompt {
			t.Fatalf("disabled plugin mutated request: changed=%t err=%v request=%#v", changed, err, request)
		}
	})
}

func semanticSystemPromptDecision(name string, prompt string, mode string, enabled *bool) config.Decision {
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginSystemPrompt,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": enabled, "system_prompt": prompt, "mode": mode,
			}),
		}},
	}
}
