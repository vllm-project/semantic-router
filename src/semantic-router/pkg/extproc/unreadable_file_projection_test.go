package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A model that declares its capabilities without file input refuses a
// conversation that holds a PDF, and every later call of the session carries
// it. With content.file accepted the file reaches such a model as a note,
// while a model that declares file input still receives the file.
func TestUnreadableFileOptInQualifiesModelsWithoutFileInput(t *testing.T) {
	request := llmprotocol.Request{
		Trusted: llmprotocol.TrustedMetadata{SourceFormat: llmprotocol.AnthropicMessagesV1},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "read report.pdf"}}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
				CallID: "toolu_1", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentFile, MediaType: "application/pdf", Data: "JVBERi0="}},
			}}}},
		},
	}
	router := func(lossy ...string) *OpenAIRouter {
		return &OpenAIRouter{Config: &config.RouterConfig{
			BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
				"no-files": {APIFormat: "anthropic", Capabilities: []string{"image_input", "tools"}},
				"files":    {APIFormat: "anthropic", Capabilities: []string{"image_input", "file_input"}},
			}},
			RouterOptions: config.RouterOptions{AllowLossyTranslation: lossy},
		}}
	}
	qualify := func(r *OpenAIRouter, model string) (llmprotocol.Request, llmprotocol.Diagnostics, error) {
		projected, diagnostics, err := r.projectRequestForBackendWithDiagnostics(request, model, llmprotocol.AnthropicMessagesV1)
		if err != nil {
			return projected, diagnostics, err
		}
		return projected, diagnostics, r.providerCapabilityMismatch(model, llmprotocol.AnthropicMessagesV1, llmprotocol.RequiredCapabilities(projected))
	}

	if _, _, err := qualify(router(), "no-files"); err == nil || !strings.Contains(err.Error(), "file_input") {
		t.Fatalf("strict qualification error = %v, want the file_input refusal", err)
	}
	projected, diagnostics, err := qualify(router(llmprotocol.LossyFieldUnreadableFile), "no-files")
	if err != nil {
		t.Fatalf("opted-in qualification failed: %v", err)
	}
	if note := projected.Messages[1].Content[0].ToolResult.Content[0]; note.Kind != llmprotocol.ContentText {
		t.Fatalf("tool result still carries %+v", note)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != llmprotocol.LossyFieldUnreadableFile {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	projected, diagnostics, err = qualify(router(llmprotocol.LossyFieldUnreadableFile), "files")
	if err != nil || len(diagnostics) != 0 || projected.Messages[1].Content[0].ToolResult.Content[0].Kind != llmprotocol.ContentFile {
		t.Fatalf("a model with file input lost the file: err=%v diagnostics=%+v", err, diagnostics)
	}
}
