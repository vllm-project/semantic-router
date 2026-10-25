package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// Compatible Anthropic providers sometimes omit the required, nullable
// stop_sequence field. All three response surfaces interpret that omission as
// null, but retain a bounded diagnostic rather than silently changing policy.
// Call only after strict provider decoding and stop-reason validation.
func anthropicStopSequenceDiagnostics(
	body []byte,
	container string,
	policy llmprotocol.Policy,
) (llmprotocol.Diagnostics, error) {
	fields, err := providerEventFields(body)
	if err != nil {
		return nil, err
	}
	field := "stop_sequence"
	if container != "" {
		fields, err = providerNestedEventFields(fields, container)
		if err != nil {
			return nil, err
		}
		field = container + "." + field
	}
	if _, present := fields["stop_sequence"]; present {
		return nil, nil
	}
	return appendDiagnostics(nil, llmprotocol.Diagnostics{{
		Source: llmprotocol.AnthropicMessagesV1,
		Field:  field,
		Action: llmprotocol.DiagnosticApproximated,
		Reason: "absent nullable stop_sequence interpreted as null for provider compatibility",
	}}, policy.Limits.Diagnostics), nil
}

func anthropicStreamStopSequenceDiagnostics(
	body []byte,
	eventType string,
	policy llmprotocol.Policy,
) (llmprotocol.Diagnostics, error) {
	switch eventType {
	case "message_start":
		return anthropicStopSequenceDiagnostics(body, "message", policy)
	case "message_delta":
		return anthropicStopSequenceDiagnostics(body, "delta", policy)
	default:
		return nil, nil
	}
}
