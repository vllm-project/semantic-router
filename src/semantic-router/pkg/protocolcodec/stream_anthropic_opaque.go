package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func (encoder *anthropicStreamEncoder) encodeAnthropicOpaque(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.DynamoNVExt != nil {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_nvext_translation",
			"Dynamo nvext stream chunks cannot be translated across wire formats", nil,
		)
	}
	if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
	}
	return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
}
