package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRetrieveFromOpenAIResponseLimit(t *testing.T) {
	responseBody, err := json.Marshal(map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{{
			"content":  "retrieved context",
			"filename": "doc.txt",
		}},
	})
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(responseBody)
	}))
	defer server.Close()

	tests := []struct {
		name             string
		maxResponseBytes int64
		wantErr          bool
	}{
		{name: "default"},
		{name: "at limit", maxResponseBytes: int64(len(responseBody))},
		{name: "one byte over", maxResponseBytes: int64(len(responseBody)) - 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ragConfig := &config.RAGPluginConfig{
				Enabled: true,
				Backend: "openai",
				BackendConfig: config.MustStructuredPayload(&config.OpenAIRAGConfig{
					VectorStoreID:    "vs_123",
					BaseURL:          server.URL,
					APIKey:           "secret",
					MaxResponseBytes: tt.maxResponseBytes,
				}),
			}

			contextText, err := (&OpenAIRouter{}).retrieveFromOpenAI(
				context.Background(),
				&RequestContext{UserContent: "hello"},
				ragConfig,
			)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
					t.Fatalf("retrieveFromOpenAI() error = %v, want response limit error", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("retrieveFromOpenAI() error = %v", err)
			}
			if contextText != "retrieved context" {
				t.Fatalf("context = %q, want retrieved context", contextText)
			}
		})
	}
}

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

//nolint:cyclop // The table verifies every role and content variant in one contract test.
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
