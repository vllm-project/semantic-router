package extproc

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func outputContractTestRouter(t *testing.T, output ...string) (*OpenAIRouter, *config.Decision, string) {
	t.Helper()
	router, chat := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	painter := "painter"
	router.Config.ModelConfig[painter] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatImages,
		Capabilities:       []string{"image_generation"},
		ExternalModelIDs:   map[string]string{"vllm": "provider-image"},
	}
	decision := &config.Decision{
		Name:      "image",
		Output:    &config.DecisionOutput{Kind: config.OutputKinds(output)},
		ModelRefs: []config.ModelRef{{Model: chat}, {Model: painter}},
	}
	return router, decision, painter
}

// A text-only request entering an image decision carries the hosted
// image_generation operation, so selection skips the chat-wire candidate and
// the images model receives the request's text as its prompt.
func TestATextOnlyRequestEnteringAnImageDecisionGeneratesTheImage(t *testing.T) {
	router, decision, painter := outputContractTestRouter(t, config.OutputKindImage)
	request := testNeutralRequest("vllm-sr/auto", "draw a lighthouse at dusk")
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
	ctx.VSRSelectedDecision = decision

	if err := router.applyDecisionOutput(ctx); err != nil {
		t.Fatalf("applyDecisionOutput: %v", err)
	}
	if request.ImageGeneration == nil || request.ToolChoice.Mode != llmprotocol.ToolChoiceImageGeneration || request.Generation != 2 {
		t.Fatalf("the request does not carry the forced operation: %+v", request)
	}
	dispatch, err := selectCapabilityTestDispatch(router, request, decision, ctx)
	if err != nil {
		t.Fatalf("selection: %v", err)
	}
	if dispatch.logicalModel != painter || dispatch.targetFormat != llmprotocol.OpenAIImagesV1 {
		t.Fatalf("selected %s on %s, want %s on the images API", dispatch.logicalModel, dispatch.targetFormat, painter)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode the images request: %v", err)
	}
	if !bytes.Contains(body, []byte(`"prompt":"draw a lighthouse at dusk"`)) {
		t.Fatalf("images request = %s", body)
	}
}

// Text with an image leaves the operation to the model, which answers on the
// Responses API with a message and the image.
func TestTextWithAnImageLeavesTheOperationToTheModel(t *testing.T) {
	router, decision, _ := outputContractTestRouter(t, config.OutputKindText, config.OutputKindImage)
	request := testNeutralRequest("vllm-sr/auto", "explain a lighthouse and draw one")
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)
	ctx.VSRSelectedDecision = decision
	if err := router.applyDecisionOutput(ctx); err != nil {
		t.Fatalf("applyDecisionOutput: %v", err)
	}
	if request.ImageGeneration == nil || request.ToolChoice.Mode != "" {
		t.Fatalf("the operation must be offered, not forced: %+v", request)
	}
	engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, *request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode for the omni model: %v", err)
	}
	if !bytes.Contains(encoded.Body, []byte(`"type":"image_generation"`)) || bytes.Contains(encoded.Body, []byte(`"tool_choice"`)) {
		t.Fatalf("omni request = %s", encoded.Body)
	}
	answer := []byte(`{"id":"resp_1","object":"response","created_at":1,"model":"omni","status":"completed","error":null,"incomplete_details":null,"output":[` +
		`{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"A lighthouse guides ships.","annotations":[]}]},` +
		`{"type":"image_generation_call","id":"ig_1","status":"completed","result":"aGVsbG8="}],` +
		`"parallel_tool_calls":false,"tool_choice":"auto","tools":[{"type":"image_generation"}],` +
		`"usage":{"input_tokens":7,"input_tokens_details":{"cached_tokens":0},"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18}}`)
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, answer, nil)
	if err != nil {
		t.Fatalf("render the answer: %v", err)
	}
	if !bytes.Contains(translated.Body, []byte(`"type":"message"`)) || !bytes.Contains(translated.Body, []byte(`"type":"image_generation_call"`)) {
		t.Fatalf("client body = %s", translated.Body)
	}
}

// A client whose API cannot return an image, or which forbids the
// operation, is refused before any model is called.
func TestAnImageDecisionRefusesWhatCannotReturnTheImage(t *testing.T) {
	for _, tc := range []struct {
		client     llmprotocol.WireFormat
		toolChoice llmprotocol.ToolChoiceMode
		code, want string
	}{
		{llmprotocol.OpenAIChatV1, "", "unsupported_capability", "the matched route answers with a generated image, which the Chat Completions API cannot return; call the Responses API or the Images API"},
		{llmprotocol.AnthropicMessagesV1, "", "unsupported_capability", "which the Anthropic Messages API cannot return"},
		{llmprotocol.OpenAIResponsesV1, llmprotocol.ToolChoiceNone, "image_generation_forbidden", "the request's tool_choice none forbids the image_generation operation"},
	} {
		router, decision, _ := outputContractTestRouter(t, config.OutputKindImage)
		request := testNeutralRequest("vllm-sr/auto", "draw a lighthouse")
		request.ToolChoice.Mode = tc.toolChoice
		ctx := routingTestContext(tc.client, request)
		ctx.VSRSelectedDecision = decision
		err := router.applyDecisionOutput(ctx)
		var protocolErr *llmprotocol.ProtocolError
		if !errors.As(err, &protocolErr) || protocolErr.Code != tc.code || !strings.Contains(protocolErr.Message, tc.want) {
			t.Errorf("%s: got %v, want %s: %s", tc.client, err, tc.code, tc.want)
		}
		if request.ImageGeneration != nil || request.Generation != 1 {
			t.Errorf("%s: a refused request changed: %+v", tc.client, request)
		}
	}
}

func imagesEntryTestRequest(t *testing.T, prompt string) *llmprotocol.Request {
	t.Helper()
	engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	body := fmt.Sprintf(`{"model":"vllm-sr/auto","prompt":%q,"size":"512x512"}`, prompt)
	request, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIImagesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode the images request: %v", err)
	}
	return &request
}

// The Images API entry is an inference path whose requests are decoded as the
// images dialect.
func TestTheImagesAPIEntryIsAnInferencePath(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	if response, requiresBody := router.classifyRequestHeaders("POST", "/v1/images/generations"); response != nil || !requiresBody {
		t.Fatalf("POST /v1/images/generations = %+v, %v; want an inference request", response, requiresBody)
	}
	if response, _ := router.classifyRequestHeaders("GET", "/v1/images/generations"); response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_MethodNotAllowed {
		t.Fatalf("GET /v1/images/generations = %+v, want 405", response)
	}
	ctx := &RequestContext{}
	detectSourceFormat("/v1/images/generations?trace=1", ctx)
	if ctx.SourceFormat != llmprotocol.OpenAIImagesV1 {
		t.Fatalf("source format = %q, want the images dialect", ctx.SourceFormat)
	}
}

// An Images API request routes on its prompt into an image decision, whose
// model generates the image the client receives in the images dialect, from
// an images model or a Responses model alike.
func TestAnImagesAPIRequestGetsTheImageDecisionsImage(t *testing.T) {
	router, decision, painter := outputContractTestRouter(t, config.OutputKindImage)
	request := imagesEntryTestRequest(t, "a lighthouse at dusk")
	generation := request.Generation
	ctx := routingTestContext(llmprotocol.OpenAIImagesV1, request)
	ctx.VSRSelectedDecision = decision
	if err := router.applyDecisionOutput(ctx); err != nil {
		t.Fatalf("applyDecisionOutput: %v", err)
	}
	if request.ToolChoice.Mode != llmprotocol.ToolChoiceImageGeneration || request.Generation != generation+1 {
		t.Fatalf("the request does not force the image: %+v", request)
	}
	if rejected := router.imagesEntryRejection(ctx); rejected != nil {
		t.Fatalf("the image decision was refused: %+v", rejected)
	}
	dispatch, err := selectCapabilityTestDispatch(router, request, decision, ctx)
	if err != nil {
		t.Fatalf("selection: %v", err)
	}
	if dispatch.logicalModel != painter || dispatch.targetFormat != llmprotocol.OpenAIImagesV1 {
		t.Fatalf("selected %s on %s, want %s on the images API", dispatch.logicalModel, dispatch.targetFormat, painter)
	}
	body, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatalf("encode the images request: %v", err)
	}
	if !bytes.Contains(body, []byte(`"prompt":"a lighthouse at dusk"`)) || !bytes.Contains(body, []byte(`"size":"512x512"`)) {
		t.Fatalf("images request = %s", body)
	}

	engine, err := protocolcodec.NewEngine(protocolcodec.NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	for format, answer := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIImagesV1: `{"created":1,"data":[{"b64_json":"aGVsbG8="}]}`,
		llmprotocol.OpenAIResponsesV1: `{"id":"resp_1","object":"response","created_at":1,"model":"omni","status":"completed","error":null,"incomplete_details":null,"output":[` +
			`{"type":"image_generation_call","id":"ig_1","status":"completed","result":"aGVsbG8="}],` +
			`"parallel_tool_calls":false,"tool_choice":"auto","tools":[{"type":"image_generation"}],` +
			`"usage":{"input_tokens":7,"input_tokens_details":{"cached_tokens":0},"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":18}}`,
	} {
		translated, err := engine.TranslateResponse(format, llmprotocol.OpenAIImagesV1, []byte(answer), nil)
		if err != nil || !bytes.Contains(translated.Body, []byte(`"data":[{"b64_json":"aGVsbG8="`)) {
			t.Fatalf("%s answer for the images client = %v, %s", format, err, translated.Body)
		}
	}
}

// An Images API request that reaches a decision answering in text, or no
// decision at all, is refused before any model is called, without naming the
// decision.
func TestAnImagesAPIRequestOutsideAnImageDecisionIsRefused(t *testing.T) {
	router, decision, _ := outputContractTestRouter(t, config.OutputKindText)
	decision.Name = "lighthouse-facts"
	request := imagesEntryTestRequest(t, "what is a lighthouse?")
	ctx := routingTestContext(llmprotocol.OpenAIImagesV1, request)
	ctx.VSRSelectedDecision = decision
	err := router.applyDecisionOutput(ctx)
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "image_route_required" || strings.Contains(protocolErr.Message, decision.Name) {
		t.Fatalf("a text decision = %v, want image_route_required without the decision's name", err)
	}

	ctx = routingTestContext(llmprotocol.OpenAIImagesV1, imagesEntryTestRequest(t, "draw a lighthouse"))
	response := router.imagesEntryRejection(ctx)
	if response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_BadRequest ||
		ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "image_route_required" {
		t.Fatalf("a request without a decision = %+v, %+v; want image_route_required", response, ctx.ImmediateProtocolError)
	}
	if router.imagesEntryRejection(routingTestContext(llmprotocol.OpenAIChatV1, testNeutralRequest("vllm-sr/auto", "hello"))) != nil {
		t.Fatal("a Chat Completions request was refused as an Images API request")
	}
}

func TestATextDecisionLeavesTheRequestAsItIs(t *testing.T) {
	router, decision, _ := outputContractTestRouter(t, config.OutputKindText)
	request := testNeutralRequest("vllm-sr/auto", "what is a lighthouse?")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision
	if err := router.applyDecisionOutput(ctx); err != nil || request.ImageGeneration != nil || request.Generation != 1 {
		t.Fatalf("a text decision changed the request: %v %+v", err, request)
	}
}
