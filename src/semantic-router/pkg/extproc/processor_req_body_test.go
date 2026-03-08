package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaybeForceImageGenerationModalitySetsBothWhenUserContentPresent(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{HasImageGenerationTool: true},
	}

	router.maybeForceImageGenerationModality("draw a cat", ctx)

	if ctx.ModalityClassification == nil {
		t.Fatal("expected modality classification to be set")
	}
	if ctx.ModalityClassification.Modality != ModalityBoth {
		t.Fatalf("expected modality %q, got %q", ModalityBoth, ctx.ModalityClassification.Modality)
	}
	if ctx.ModalityClassification.Method != "image_generation_tool" {
		t.Fatalf("unexpected modality method: %q", ctx.ModalityClassification.Method)
	}
}

func TestRefreshResponseAPITranslatedBodyStoresBody(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{IsResponseAPIRequest: true},
	}

	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	router.refreshResponseAPITranslatedBody(ctx, body)

	if string(ctx.ResponseAPICtx.TranslatedBody) != string(body) {
		t.Fatalf("translated body not stored: got %q", string(ctx.ResponseAPICtx.TranslatedBody))
	}
}

func TestMaybeForceImageGenerationModalityRespectsExistingNonARDecision(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{HasImageGenerationTool: true},
		ModalityClassification: &ModalityClassificationResult{
			Modality: ModalityDiffusion,
			Method:   "existing",
		},
	}

	router.maybeForceImageGenerationModality("draw a cat", ctx)

	if ctx.ModalityClassification.Method != "existing" {
		t.Fatalf("expected existing modality classification to be preserved, got %q", ctx.ModalityClassification.Method)
	}
}

func TestPrepareAnthropicRoutingRequestSetsAnthropicContext(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(false)

	request, err := parseOpenAIRequest([]byte(`{"model":"claude","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("parseOpenAIRequest failed: %v", err)
	}

	ctx := &RequestContext{Headers: map[string]string{"x-user-anthropic-key": "anthropic-test-key"}}
	_, body, response := router.prepareAnthropicRoutingRequest(request, "claude-3-5-sonnet", "", ctx)
	if response != nil {
		t.Fatal("did not expect immediate error response")
	}
	if len(body) == 0 {
		t.Fatal("expected anthropic request body to be generated")
	}
	if ctx.APIFormat != config.APIFormatAnthropic {
		t.Fatalf("expected API format %q, got %q", config.APIFormatAnthropic, ctx.APIFormat)
	}
	if ctx.RequestModel != "claude-3-5-sonnet" {
		t.Fatalf("expected request model to be updated, got %q", ctx.RequestModel)
	}
}
