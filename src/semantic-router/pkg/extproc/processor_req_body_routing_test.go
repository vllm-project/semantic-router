package extproc

import (
	"context"
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestHandleAutoModelRoutingPreservesSelectedModelHeaderAndRewritesUpstreamModel(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "qwen14b-dev",
			ModelConfig: map[string]config.ModelParams{
				"qwen14b-dev": {
					PreferredEndpoints: []string{"qwen14b-dev_vllm"},
					ExternalModelIDs: map[string]string{
						"vllm": "Qwen/Qwen2.5-14B-Instruct",
					},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "qwen14b-dev_vllm",
					Address: "127.0.0.1",
					Port:    8000,
					Type:    "vllm",
					Weight:  1,
				},
			},
		},
	}

	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	ctx := &RequestContext{
		Headers:      map[string]string{},
		TraceContext: context.Background(),
	}
	openAIRequest := &openai.ChatCompletionNewParams{
		Model: "MoM",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hello from multi-endpoint"),
		},
	}
	baseResponse := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	response, err := router.handleAutoModelRouting(
		openAIRequest,
		"MoM",
		"",
		entropy.ReasoningDecision{},
		"qwen14b-dev",
		ctx,
		baseResponse,
	)
	if err != nil {
		t.Fatalf("handleAutoModelRouting returned error: %v", err)
	}

	requestBodyResponse := response.GetRequestBody()
	if requestBodyResponse == nil {
		t.Fatal("expected request body response")
	}
	headerMap := headerValuesByName(requestBodyResponse.Response.HeaderMutation.SetHeaders)
	if got := headerMap[headers.SelectedModel]; got != "qwen14b-dev" {
		t.Fatalf("expected %s header to preserve router alias, got %q", headers.SelectedModel, got)
	}
	if got := headerMap[headers.GatewayDestinationEndpoint]; got != "127.0.0.1:8000" {
		t.Fatalf("expected %s header to contain resolved endpoint, got %q", headers.GatewayDestinationEndpoint, got)
	}

	var body map[string]any
	if err := json.Unmarshal(requestBodyResponse.Response.BodyMutation.GetBody(), &body); err != nil {
		t.Fatalf("failed to decode mutated body: %v", err)
	}
	if got := body["model"]; got != "Qwen/Qwen2.5-14B-Instruct" {
		t.Fatalf("expected upstream body model rewrite, got %#v", got)
	}
}
