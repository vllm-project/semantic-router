package extproc

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// A request that needs image_generation routed to an openai.chat.v1 backend
// must fail fast at dispatch with a capability ProtocolError instead of
// surfacing later as a generic 500 internal error from encode.
func TestPrepareProviderDispatchRejectsImageGenerationOnChatWire(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(logicalModel, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	_, err := router.prepareProviderDispatch(request, logicalModel, "", false, ctx)
	if err == nil {
		t.Fatalf("prepareProviderDispatch: expected capability mismatch error, got nil")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("expected *llmprotocol.ProtocolError, got %T: %v", err, err)
	}
	if protocolError.Code != "unsupported_capability" {
		t.Fatalf("protocol error code = %q, want unsupported_capability", protocolError.Code)
	}
	if ctx.ImmediateProtocolError == nil {
		t.Fatalf("expected ctx.ImmediateProtocolError to be populated")
	}
}

// The same request on a backend whose wire can express image_generation
// (OpenAI Responses) must dispatch without error.
func TestPrepareProviderDispatchAllowsImageGenerationOnResponsesWire(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIResponsesV1)
	request := testNeutralRequest(logicalModel, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIResponsesV1, request)

	if _, err := router.prepareProviderDispatch(request, logicalModel, "", false, ctx); err != nil {
		t.Fatalf("prepareProviderDispatch on responses wire: %v", err)
	}
}

// processBodyRoutingError must convert a ProtocolError into an immediate 400
// response whose body is encoded in the client's source wire format, and must
// leave non-protocol errors untouched for the caller to surface as-is.
func TestProcessBodyRoutingErrorConvertsProtocolErrorToImmediateResponse(t *testing.T) {
	request := testNeutralRequest("m", "draw a cat")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	router := &OpenAIRouter{}

	cause := llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"unsupported_capability",
		"request requires capability image_generation which wire openai.chat.v1 cannot express",
		nil,
	)
	response, converted := router.processBodyRoutingError(cause, ctx)
	if !converted {
		t.Fatalf("expected ProtocolError to be converted")
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "unsupported_capability" {
		t.Fatalf("ctx.ImmediateProtocolError not populated with the capability error")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != 400 {
		t.Fatalf("immediate status = %d, want 400", got)
	}

	encoded := router.encodeImmediateResponseForClient(response, ctx)
	body := encoded.GetImmediateResponse().GetBody()
	if len(body) == 0 {
		t.Fatalf("expected a wire-encoded error body")
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("decode error envelope: %v (body=%s)", err, body)
	}
	if envelope.Error.Message == "" {
		t.Fatalf("error envelope message is empty: %s", body)
	}
}

func TestProcessBodyRoutingErrorLeavesNonProtocolErrorUntouched(t *testing.T) {
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, testNeutralRequest("m", "hi"))
	router := &OpenAIRouter{}

	sentinel := errors.New("some internal failure")
	if response, converted := router.processBodyRoutingError(sentinel, ctx); converted || response != nil {
		t.Fatalf("non-protocol error must not be converted (response=%v converted=%v)", response, converted)
	}
}

// End-to-end through handleEntrypointModelRouting: the capability mismatch must
// surface as a clean ProtocolError (not a generic internal failure) at the
// routing seam; the RPC boundary converts it into the client 400 immediate
// response (covered by TestProcessBodyRoutingError*).
func TestEntrypointRoutingSurfacesCapabilityMismatchAsProtocolError(t *testing.T) {
	router, logicalModel := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	request := testNeutralRequest(logicalModel, "draw a cat")
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	_, err := router.handleEntrypointModelRouting(
		request, logicalModel, "dispatch", entropy.ReasoningDecision{UseReasoning: false}, logicalModel, ctx,
	)
	if err == nil {
		t.Fatalf("handleEntrypointModelRouting: expected capability mismatch error, got nil")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("expected *llmprotocol.ProtocolError, got %T: %v", err, err)
	}
	if protocolError.Code != "unsupported_capability" {
		t.Fatalf("protocol error code = %q, want unsupported_capability", protocolError.Code)
	}
	if ctx.ImmediateProtocolError == nil {
		t.Fatalf("expected ImmediateProtocolError to be set on capability mismatch")
	}
}

// When candidate modelRefs declare capabilities, a candidate is only taken if
// its declared capabilities cover the request's task (modality) capabilities.
func TestPrepareProviderDispatchPrefersCapabilityAnnotatedModelRef(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	visionOnly := "fallback-vision"
	generator := "fallback-generator"
	router.Config.ModelConfig[visionOnly] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		Capabilities:       []string{"image_input"},
		ExternalModelIDs:   map[string]string{"vllm": "provider-vision"},
	}
	router.Config.ModelConfig[generator] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		Capabilities:       []string{"image_generation"},
		ExternalModelIDs:   map[string]string{"vllm": "provider-generator"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary},
			{Model: visionOnly},
			{Model: generator},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err != nil {
		t.Fatalf("expected reroute to generator, got error: %v", err)
	}
	if dispatch.logicalModel != generator {
		t.Fatalf("logical model = %s, want %s (must skip declared vision-only candidate)", dispatch.logicalModel, generator)
	}
}

// A declared candidate that cannot cover the required task capability is
// skipped, and an undeclared candidate (wire-only qualification) is taken.
func TestPrepareProviderDispatchSkipsDeclaredMismatchAndFallsBackToWireOnly(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	visionOnly := "fallback-vision"
	undeclared := "fallback-undeclared"
	router.Config.ModelConfig[visionOnly] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		Capabilities:       []string{"image_input"},
		ExternalModelIDs:   map[string]string{"vllm": "provider-vision"},
	}
	router.Config.ModelConfig[undeclared] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		ExternalModelIDs:   map[string]string{"vllm": "provider-undeclared"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary},
			{Model: visionOnly},
			{Model: undeclared},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err != nil {
		t.Fatalf("expected reroute to undeclared candidate, got error: %v", err)
	}
	if dispatch.logicalModel != undeclared {
		t.Fatalf("logical model = %s, want %s", dispatch.logicalModel, undeclared)
	}
}

// When every wire-qualified candidate declares capabilities that don't cover
// the required task, no candidate qualifies and the mismatch surfaces.
func TestPrepareProviderDispatchRejectsWhenAllDeclaredCandidatesMismatch(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	visionOnly := "fallback-vision"
	router.Config.ModelConfig[visionOnly] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		Capabilities:       []string{"image_input"},
		ExternalModelIDs:   map[string]string{"vllm": "provider-vision"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary},
			{Model: visionOnly},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	_, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err == nil {
		t.Fatalf("expected capability mismatch error, got nil")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("expected *llmprotocol.ProtocolError, got %T: %v", err, err)
	}
	if protocolError.Code != "unsupported_capability" {
		t.Fatalf("protocol error code = %q, want unsupported_capability", protocolError.Code)
	}
}

// When the selected decision offers another modelRef whose wire format can
// express the required capabilities, dispatch must be re-routed to it instead
// of surfacing a capability error.
func TestPrepareProviderDispatchReroutesToCapabilityQualifiedModelRef(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	fallback := "fallback-responses"
	router.Config.ModelConfig[fallback] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatResponses,
		ExternalModelIDs:   map[string]string{"vllm": "provider-fallback"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary},
			{Model: fallback},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err != nil {
		t.Fatalf("expected reroute to qualified modelRef, got error: %v", err)
	}
	if dispatch.logicalModel != fallback {
		t.Fatalf("logical model = %s, want %s (rerouted)", dispatch.logicalModel, fallback)
	}
	if dispatch.targetFormat != llmprotocol.OpenAIResponsesV1 {
		t.Fatalf("target format = %s, want %s", dispatch.targetFormat, llmprotocol.OpenAIResponsesV1)
	}
	if request.Model != "provider-fallback" {
		t.Fatalf("upstream model = %q, want provider-fallback", request.Model)
	}
	if ctx.TargetFormat != llmprotocol.OpenAIResponsesV1 {
		t.Fatalf("ctx.TargetFormat = %s, want %s", ctx.TargetFormat, llmprotocol.OpenAIResponsesV1)
	}
	if ctx.ImmediateProtocolError != nil {
		t.Fatalf("ImmediateProtocolError must be cleared after a successful reroute")
	}
}

// When the decision offers no modelRef that can express the required
// capability, the capability mismatch must still surface as a ProtocolError.
func TestPrepareProviderDispatchRejectsWhenNoQualifiedModelRef(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	decision := &config.Decision{
		Name: "ChatOnly",
		ModelRefs: []config.ModelRef{
			{Model: primary},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	_, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err == nil {
		t.Fatalf("expected capability mismatch error, got nil")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("expected *llmprotocol.ProtocolError, got %T: %v", err, err)
	}
	if protocolError.Code != "unsupported_capability" {
		t.Fatalf("protocol error code = %q, want unsupported_capability", protocolError.Code)
	}
}

// A DALL-E (images) sibling is a qualified reroute target for a hosted
// image_generation request: its wire advertises image_generation and the
// hosted-tool tools requirement.
func TestPrepareProviderDispatchReroutesToImagesWireSibling(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	imageBackend := "image-backend"
	router.Config.ModelConfig[imageBackend] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatImages,
		ExternalModelIDs:   map[string]string{"vllm": "provider-image"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary},
			{Model: imageBackend},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{Size: "1024x1024"}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err != nil {
		t.Fatalf("expected reroute to images sibling, got error: %v", err)
	}
	if dispatch.logicalModel != imageBackend {
		t.Fatalf("logical model = %s, want %s (rerouted to images backend)", dispatch.logicalModel, imageBackend)
	}
	if dispatch.targetFormat != llmprotocol.OpenAIImagesV1 {
		t.Fatalf("target format = %s, want %s", dispatch.targetFormat, llmprotocol.OpenAIImagesV1)
	}
	if ctx.TargetFormat != llmprotocol.OpenAIImagesV1 {
		t.Fatalf("ctx.TargetFormat = %s, want %s", ctx.TargetFormat, llmprotocol.OpenAIImagesV1)
	}
}

// An explicit tool_choice: none forbids all tools, including the hosted
// image_generation operation, so the request must NOT be rerouted to an
// images sibling: no image backend call may occur for a request whose caller
// forbade all tools (Xun review 5119851642).
func TestPrepareProviderDispatchNoneToolChoiceDoesNotRerouteToImages(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	imageBackend := "image-backend"
	router.Config.ModelConfig[imageBackend] = config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatImages,
		ExternalModelIDs:   map[string]string{"vllm": "provider-image"},
	}
	decision := &config.Decision{
		Name: "Omni",
		ModelRefs: []config.ModelRef{
			{Model: primary},
			{Model: imageBackend},
		},
	}
	request := testNeutralRequest(primary, "draw a cat")
	request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}
	request.ImageGeneration = &llmprotocol.ImageGenerationOptions{Size: "1024x1024"}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = decision

	dispatch, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	if err != nil {
		t.Fatalf("tool_choice none must dispatch on the primary chat wire, got error: %v", err)
	}
	if dispatch.logicalModel != primary {
		t.Fatalf("logical model = %s, want %s (must not reroute to images backend)", dispatch.logicalModel, primary)
	}
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		t.Fatalf("target format = %s, want %s", dispatch.targetFormat, llmprotocol.OpenAIChatV1)
	}
	if request.ToolChoice.Mode != llmprotocol.ToolChoiceNone {
		t.Fatalf("explicit no-tool choice must be preserved, got %v", request.ToolChoice.Mode)
	}
}

func TestWireFormatForImagesAPIFormat(t *testing.T) {
	format, err := wireFormatForModel(config.APIFormatImages)
	if err != nil {
		t.Fatalf("wireFormatForModel(images): %v", err)
	}
	if format != llmprotocol.OpenAIImagesV1 {
		t.Fatalf("format = %s, want %s", format, llmprotocol.OpenAIImagesV1)
	}
}

func TestRequestWirePathForImagesFormat(t *testing.T) {
	if path := requestWirePath(llmprotocol.OpenAIImagesV1); path != "/v1/images/generations" {
		t.Fatalf("wire path = %q, want /v1/images/generations", path)
	}
}
