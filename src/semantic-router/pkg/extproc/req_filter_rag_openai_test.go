package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIRAGConfigurationModes(t *testing.T) {
	tests := []struct {
		name string
		mode string
	}{
		{name: "direct search", mode: "direct_search"},
		{name: "tool based", mode: "tool_based"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ragConfig := &config.RAGPluginConfig{
				Backend: "openai",
				BackendConfig: config.MustStructuredPayload(&config.OpenAIRAGConfig{
					VectorStoreID: "vs_test123",
					WorkflowMode:  test.mode,
					MaxNumResults: intPtr(10),
				}),
			}
			got, err := ragConfig.OpenAIBackendConfig()
			if err != nil {
				t.Fatalf("decode backend config: %v", err)
			}
			if got.VectorStoreID != "vs_test123" || got.WorkflowMode != test.mode || got.MaxNumResults == nil || *got.MaxNumResults != 10 {
				t.Fatalf("unexpected backend config: %+v", got)
			}
		})
	}
}

func TestServerHostedFileSearchIsExplicitlyUnsupported(t *testing.T) {
	ctx := &RequestContext{SemanticRequest: testNeutralRequest("model", "find a file")}
	err := (&OpenAIRouter{}).addFileSearchToolToRequest(ctx, &config.OpenAIRAGConfig{VectorStoreID: "vs_test123"})
	if err == nil {
		t.Fatal("server-hosted tool was accepted without a representable neutral contract")
	}
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Fatalf("expected typed unsupported-feature error, got %T: %v", err, err)
	}
	if len(ctx.SemanticRequest.Tools) != 0 {
		t.Fatal("unsupported tool mutated the neutral request")
	}
}

func TestRAGToolRoleInjectionMutatesNeutralRequest(t *testing.T) {
	request := testNeutralRequest("model", "test query")
	ctx := &RequestContext{RequestID: "request-1", SemanticRequest: request}
	ragConfig := &config.RAGPluginConfig{InjectionMode: "tool_role"}

	if err := (&OpenAIRouter{}).injectRAGContext(ctx, "grounded context", ragConfig); err != nil {
		t.Fatalf("inject RAG context: %v", err)
	}
	if len(request.Messages) != 3 || request.Messages[1].Role != llmprotocol.RoleAssistant || request.Messages[2].Role != llmprotocol.RoleTool {
		t.Fatalf("unexpected neutral tool exchange: %#v", request.Messages)
	}
	call := request.Messages[1].Content[0].ToolCall
	result := request.Messages[2].Content[0].ToolResult
	if call == nil || result == nil || call.ID == "" || result.CallID != call.ID || result.Content[0].Text != "grounded context" {
		t.Fatalf("RAG tool lifecycle is not linked: call=%#v result=%#v", call, result)
	}
	if _, ok := ctx.RAGToolCallIDs[call.ID]; !ok {
		t.Fatal("RAG provenance did not retain the generated call ID")
	}
	if request.Generation != 2 || !ctx.HasToolsForFactCheck {
		t.Fatalf("semantic mutation metadata missing: generation=%d context=%#v", request.Generation, ctx)
	}
}

func TestRAGSystemInstructionPreservesAuthority(t *testing.T) {
	request := testNeutralRequest("model", "test query")
	request.Instructions = []llmprotocol.InstructionBlock{{
		Role:    llmprotocol.RoleDeveloper,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "developer instruction"}},
	}}
	ctx := &RequestContext{SemanticRequest: request}

	if err := (&OpenAIRouter{}).injectRAGContext(ctx, "grounded context", &config.RAGPluginConfig{InjectionMode: "system_prompt"}); err != nil {
		t.Fatalf("inject RAG system context: %v", err)
	}
	if len(request.Instructions) != 2 || request.Instructions[0].Role != llmprotocol.RoleSystem || request.Instructions[1].Role != llmprotocol.RoleDeveloper {
		t.Fatalf("instruction authority/order changed: %#v", request.Instructions)
	}
	if request.Instructions[0].Content[0].Text == "" || request.Generation != 2 {
		t.Fatalf("RAG system instruction was not applied: %#v", request.Instructions[0])
	}
}

func intPtr(value int) *int { return &value }
