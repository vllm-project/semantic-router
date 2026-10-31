package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type responsesProviderDefault struct {
	path     string
	value    json.RawMessage
	expected string
}

func responsesProviderDefaults(wire responsesResponseWire) []responsesProviderDefault {
	fields := []responsesProviderDefault{
		{"kv_transfer_params", wire.KVTransferParams, "null"},
		{"ec_transfer_params", wire.ECTransferParams, "null"},
		{"input_messages", wire.InputMessages, "null"},
		{"output_messages", wire.OutputMessages, "null"},
	}
	if wire.Usage == nil {
		return fields
	}
	if details := wire.Usage.InputTokensDetails; details != nil {
		fields = append(fields,
			responsesProviderDefault{"usage.input_tokens_details.input_tokens_per_turn", details.InputTokensPerTurn, "[]"},
			responsesProviderDefault{"usage.input_tokens_details.cached_tokens_per_turn", details.CachedTokensPerTurn, "[]"},
		)
	}
	if details := wire.Usage.OutputTokensDetails; details != nil {
		fields = append(fields,
			responsesProviderDefault{"usage.output_tokens_details.output_tokens_per_turn", details.OutputTokensPerTurn, "[]"},
			responsesProviderDefault{"usage.output_tokens_details.tool_output_tokens_per_turn", details.ToolOutputTokensPerTurn, "[]"},
			responsesProviderDefault{"usage.output_tokens_details.tool_output_tokens", details.ToolOutputTokens, "0"},
		)
	}
	return fields
}

func validateResponsesProviderDefaults(wire responsesResponseWire) error {
	for _, field := range responsesProviderDefaults(wire) {
		if len(field.value) > 0 && !matchesResponsesProviderDefault(field.value, field.expected) {
			return invalidProviderResponse("unsupported_response_extension",
				fmt.Sprintf("Responses field %s only supports %s", field.path, field.expected))
		}
	}
	return nil
}

func matchesResponsesProviderDefault(raw json.RawMessage, expected string) bool {
	switch expected {
	case "[]":
		var values []json.RawMessage
		return json.Unmarshal(raw, &values) == nil && values != nil && len(values) == 0
	case "0":
		var value *int64
		return json.Unmarshal(raw, &value) == nil && value != nil && *value == 0
	default:
		return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
	}
}
