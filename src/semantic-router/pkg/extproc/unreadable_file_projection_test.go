package extproc

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
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

// A fallback candidate is qualified and encoded through the same projection,
// so a Chat candidate without file input that qualified on the note never
// receives the file itself.
func TestUnreadableFileOptInReachesFallbackCandidates(t *testing.T) {
	router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	candidate := cfg.ModelConfig["model-fallback-1"]
	candidate.APIFormat = config.APIFormatOpenAI
	candidate.Capabilities = []string{"image_input"}
	cfg.ModelConfig["model-fallback-1"] = candidate
	cfg.AllowLossyTranslation = []string{llmprotocol.LossyFieldUnreadableFile}

	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(`{
		"model":"model-primary","max_tokens":64,
		"messages":[{"role":"user","content":[
			{"type":"text","text":"summarize this"},
			{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
		]}]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = &request
	ctx.ProtocolEnvelope = envelope
	ctx.Headers = map[string]string{}
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable

	router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
		if bytes.Contains(body, []byte("JVBERi0=")) {
			return nil, 0, fmt.Errorf("%s received the file it cannot read: %s", model, body)
		}
		return []byte(`{
			"id":"chatcmpl-note","object":"chat.completion","model":"model-fallback-1",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
		}`), http.StatusOK, nil
	}
	response := router.handleUpstreamTransportError([]byte(`{"error":{"message":"primary unavailable","type":"server_error"}}`), ctx)
	if response.GetImmediateResponse().GetStatus().GetCode() != http.StatusOK || ctx.VSRSelectedModel != "model-fallback-1" {
		t.Fatalf("fallback did not answer from the candidate: selected=%q response=%+v", ctx.VSRSelectedModel, response)
	}
	noted := false
	for _, diagnostic := range ctx.ProtocolDiagnostics {
		noted = noted || diagnostic.Field == llmprotocol.LossyFieldUnreadableFile
	}
	if !noted {
		t.Fatalf("the note was sent without its warning: %+v", ctx.ProtocolDiagnostics)
	}
}
