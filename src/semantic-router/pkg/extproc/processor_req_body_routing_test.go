package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestProviderDispatchEncodesEveryClientBackendProtocolPair(t *testing.T) {
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, source := range formats {
		for _, target := range formats {
			t.Run(string(source)+"_to_"+string(target), func(t *testing.T) {
				assertProviderDispatchPair(t, source, target)
			})
		}
	}
}

func assertProviderDispatchPair(t *testing.T, source, target llmprotocol.WireFormat) {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(target)
	ctx := &RequestContext{
		Headers: map[string]string{}, SourceFormat: source,
		RequestID: "protocol-matrix-request", TraceContext: context.Background(),
	}
	request, immediate := router.prepareProtocolRequest(protocolRequestFixture(source), ctx)
	if immediate != nil {
		t.Fatalf("prepareProtocolRequest(%s) returned immediate response", source)
	}
	response, err := router.handleSpecifiedModelRouting(request, logicalModel, "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting(%s -> %s): %v", source, target, err)
	}
	assertDispatchedRequest(t, response, target)
}

func assertDispatchedRequest(t *testing.T, response *ext_proc.ProcessingResponse, target llmprotocol.WireFormat) {
	t.Helper()
	common := response.GetRequestBody().GetResponse()
	body := common.GetBodyMutation().GetBody()
	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(target, body)
	if err != nil {
		t.Fatalf("decode dispatched %s body: %v\n%s", target, err, body)
	}
	if decoded.Model != "provider-model" || len(decoded.Messages) != 1 ||
		len(decoded.Messages[0].Content) != 1 || decoded.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("dispatched request changed semantics: %+v", decoded)
	}
	wantPath := requestWirePath(target)
	gotPath := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]
	if gotPath != wantPath {
		t.Fatalf("dispatch path = %q, want %q", gotPath, wantPath)
	}
}

func protocolRequestFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"virtual","messages":[{"role":"user","content":"hello"}],"max_tokens":8}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"virtual","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}],"max_output_tokens":8}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"virtual","max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)
	default:
		panic(fmt.Sprintf("unsupported fixture format %q", format))
	}
}

func TestProviderProtocolPath(t *testing.T) {
	tests := []struct {
		name         string
		baseURL      string
		protocolPath string
		want         string
	}{
		{name: "root", baseURL: "https://api.example.com", protocolPath: "/v1/messages", want: "/v1/messages"},
		{name: "version root", baseURL: "https://api.example.com/v1", protocolPath: "/v1/responses", want: "/v1/responses"},
		{name: "nested version root", baseURL: "https://api.example.com/openai/v1", protocolPath: "/v1/responses", want: "/openai/v1/responses"},
		{name: "custom base", baseURL: "https://api.example.com/proxy", protocolPath: "/v1/messages", want: "/proxy/v1/messages"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := providerProtocolPath(test.baseURL, test.protocolPath); got != test.want {
				t.Fatalf("providerProtocolPath(%q, %q) = %q, want %q", test.baseURL, test.protocolPath, got, test.want)
			}
		})
	}
}

func routingTestRouterForFormat(format llmprotocol.WireFormat) (*OpenAIRouter, string) {
	logicalModel := "target-" + string(format)
	apiFormat := config.APIFormatOpenAI
	providerType := "openai"
	baseURL := "http://127.0.0.1:8000/v1"
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		apiFormat = "responses"
	case llmprotocol.AnthropicMessagesV1:
		apiFormat = config.APIFormatAnthropic
		providerType = "anthropic"
		baseURL = "http://127.0.0.1:8000"
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: logicalModel,
			ModelConfig: map[string]config.ModelParams{
				logicalModel: {
					PreferredEndpoints: []string{"backend"},
					APIFormat:          apiFormat,
					ExternalModelIDs:   map[string]string{"vllm": "provider-model"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: providerType, BaseURL: baseURL},
			},
		},
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}, logicalModel
}

func TestHandleAutoModelRoutingEmitsSelectedModelAndEncodedBody(t *testing.T) {
	router := routingTestRouter("qwen14b-dev")
	request := testNeutralRequest("MoM", "hello from routed model")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ModalityClassification = &ModalityClassificationResult{
		Modality: ModalityBoth, Confidence: 0.97, Method: "signal",
	}

	response, err := router.handleEntrypointModelRouting(
		request, "MoM", "", entropy.ReasoningDecision{}, "qwen14b-dev", ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	headersByName := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if got := headersByName[headers.SelectedModel]; got != "qwen14b-dev" {
		t.Fatalf("selected model header = %q", got)
	}
	var body map[string]any
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode routed request: %v", err)
	}
	if got := body["model"]; got != "qwen14b-dev" {
		t.Fatalf("logical model = %#v", got)
	}
	if request.Model != "qwen14b-dev" || ctx.SemanticRequest != request {
		t.Fatalf("neutral request was not updated in place: %+v", request)
	}
	if ctx.ModalityClassification.Modality != ModalityBoth {
		t.Fatalf("backend dispatch lost modality evidence: %+v", ctx.ModalityClassification)
	}
}

func TestHandleAutoModelRoutingSameModelEncodesCurrentSemanticRequest(t *testing.T) {
	router := routingTestRouter("auto")
	request := testNeutralRequest("auto", "short semantic content")
	request.Generation = 4
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	response, err := router.handleEntrypointModelRouting(
		request, "auto", "", entropy.ReasoningDecision{}, "auto", ctx,
	)
	if err != nil {
		t.Fatalf("handleEntrypointModelRouting returned error: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if body.Model != "auto" || len(body.Messages) != 1 || body.Messages[0].Content != "short semantic content" {
		t.Fatalf("unexpected encoded semantic request: %+v", body)
	}
}

func TestSpecifiedModelPreservesAnthropicWireAtExtProcBoundary(t *testing.T) {
	router := routingTestRouter("test-model")
	request := testNeutralRequest("test-model", "Explain the incident.")
	request.Stream = true
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(256)
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, request)
	ctx.ExpectStreamingResponse = true

	response, err := router.handleSpecifiedModelRouting(request, "test-model", "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatalf("decode Anthropic request: %v", err)
	}
	if _, ok := body["messages"]; !ok {
		t.Fatalf("Anthropic request omitted messages: %s", common.GetBodyMutation().GetBody())
	}
	if string(body["stream"]) != "true" || string(body["max_tokens"]) != "256" {
		t.Fatalf("Anthropic options were not preserved: %s", common.GetBodyMutation().GetBody())
	}
	if got := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]; got != "/v1/messages" {
		t.Fatalf("ExtProc source-format path = %q", got)
	}
}

func routingTestRouter(model string) *OpenAIRouter {
	apiFormat := config.APIFormatOpenAI
	profileType := "openai"
	baseURL := "http://127.0.0.1:8000/v1"
	if model == "test-model" {
		apiFormat = config.APIFormatAnthropic
		profileType = "anthropic"
		baseURL = "http://127.0.0.1:8000"
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: model,
			ModelConfig: map[string]config.ModelParams{
				model: {PreferredEndpoints: []string{"backend"}, APIFormat: apiFormat},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "backend", Address: "127.0.0.1", Port: 8000,
				ProviderProfileName: "provider",
			}},
			ProviderProfiles: map[string]config.ProviderProfile{
				"provider": {Type: profileType, BaseURL: baseURL},
			},
		},
	}
	return &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
}

func routingTestContext(format llmprotocol.WireFormat, request *llmprotocol.Request) *RequestContext {
	return &RequestContext{
		Headers:         map[string]string{},
		SourceFormat:    format,
		SemanticRequest: request,
		RequestID:       "routing-test-request",
		TraceContext:    context.Background(),
	}
}
