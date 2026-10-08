package protocolcodec

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Rebuild only the supported metadata, using public identity and this encoder's
// sequence space. Raw provider lifecycle resources must not bypass projection.
func (encoder *responsesStreamEncoder) encodeDynamoResponsesLifecycle(event llmprotocol.Event) ([][]byte, error) {
	if encoder.context.Source != llmprotocol.OpenAIResponsesV1 || encoder.context.Target != llmprotocol.OpenAIResponsesV1 {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_nvext_translation",
			"Dynamo nvext stream chunks cannot be translated across wire formats", nil)
	}
	switch event.DynamoResponsesLifecycle {
	case "response.created", "response.queued", "response.in_progress":
	default:
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_dynamo_lifecycle_event",
			"Dynamo lifecycle metadata requires a Responses start-phase event", nil)
	}
	if len(event.Opaque) != 0 {
		return nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "invalid_dynamo_lifecycle_event",
			"Dynamo lifecycle metadata cannot contain an opaque frame", nil)
	}
	status := strings.TrimPrefix(event.DynamoResponsesLifecycle, "response.")
	if status == "created" {
		status = "in_progress"
	}
	response := newResponsesResponseWire(event.ResponseID, event.Model, status, 0, encoder.context.PreviousResponseID)
	nvext, err := encodeDynamoResponseNVExt(event.DynamoNVExt, encoder.policy)
	if err != nil {
		return nil, err
	}
	response.NVExt = nvext
	frame, err := encoder.encodeResponsesStreamFrame(responsesEventWire{
		Type: event.DynamoResponsesLifecycle, Sequence: encoder.nextWireSequence(), Response: &response,
	})
	if err != nil {
		return nil, err
	}
	return [][]byte{frame}, nil
}
