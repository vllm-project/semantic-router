package extproc

import (
	"encoding/json"
	"errors"
	"testing"

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
