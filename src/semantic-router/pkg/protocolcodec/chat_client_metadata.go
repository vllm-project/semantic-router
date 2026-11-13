package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// clientMetadataReason reports a member that a client library adds to what it
// sends and that no backend reads.
const clientMetadataReason = "client metadata is not model input"

// chatClientMetadata is what a Chat request decode drops from the messages a
// client replays as its library returned them: LiteLLM's
// provider_specific_fields, which holds the reply members its message type
// has no attribute for, and the parse results that the openai SDK's stream
// and parse helpers add, parsed on the message and parsed_arguments on each
// function.
type chatClientMetadata struct {
	// body is the request to decode, without the parse results.
	body []byte
	// removed reports whether the decode drops a member, null or not, so the
	// source bytes no longer describe the request.
	removed bool
	// fields maps the path of each dropped member to whether one carried a
	// value. A null member carries nothing and is not reported.
	fields map[string]bool
}

// withoutChatParseResults removes the openai SDK's parse results from the
// messages, which the strict decode would refuse as members the wire contract
// does not declare. A body it cannot rewrite is left to that decode.
func withoutChatParseResults(body []byte, policy llmprotocol.Policy) (chatClientMetadata, error) {
	metadata := chatClientMetadata{body: body, fields: map[string]bool{}}
	if !bytes.Contains(body, []byte(`"parsed`)) {
		return metadata, nil
	}
	// The body is checked as received, so a duplicate member cannot hide
	// behind the rewrite.
	if err := validateClientJSONDocument(body, policy, true, nil); err != nil {
		return metadata, err
	}
	var document map[string]json.RawMessage
	var messages []map[string]json.RawMessage
	if json.Unmarshal(body, &document) != nil || json.Unmarshal(document["messages"], &messages) != nil {
		return metadata, nil
	}
	stripped := chatClientMetadata{fields: map[string]bool{}}
	for _, message := range messages {
		if err := stripped.removeParseResults(message); err != nil {
			return metadata, nil
		}
	}
	if !stripped.removed {
		return metadata, nil
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return metadata, nil
	}
	document["messages"] = encoded
	if stripped.body, err = json.Marshal(document); err != nil {
		return metadata, nil
	}
	return stripped, nil
}

func (metadata *chatClientMetadata) removeParseResults(message map[string]json.RawMessage) error {
	metadata.take(message, "parsed", "messages[].parsed")
	var calls []map[string]json.RawMessage
	if json.Unmarshal(message["tool_calls"], &calls) != nil {
		return nil
	}
	changed := false
	for _, call := range calls {
		var function map[string]json.RawMessage
		if json.Unmarshal(call["function"], &function) != nil ||
			!metadata.take(function, "parsed_arguments", "messages[].tool_calls[].function.parsed_arguments") {
			continue
		}
		encoded, err := json.Marshal(function)
		if err != nil {
			return err
		}
		call["function"] = encoded
		changed = true
	}
	if !changed {
		return nil
	}
	encoded, err := json.Marshal(calls)
	message["tool_calls"] = encoded
	return err
}

func (metadata *chatClientMetadata) take(object map[string]json.RawMessage, name, path string) bool {
	value, found := object[name]
	if !found {
		return false
	}
	delete(object, name)
	metadata.removed = true
	metadata.fields[path] = metadata.fields[path] || rawJSONNonNull(value)
	return true
}

// noteProviderSpecificFields drops LiteLLM's provider_specific_fields, which
// the message wire declares for upstream replies.
func (metadata *chatClientMetadata) noteProviderSpecificFields(messages []chatMessageWire) {
	for _, message := range messages {
		if len(message.ProviderSpecificFields) == 0 {
			continue
		}
		metadata.removed = true
		path := "messages[].provider_specific_fields"
		metadata.fields[path] = metadata.fields[path] || rawJSONNonNull(message.ProviderSpecificFields)
	}
}

func (metadata *chatClientMetadata) diagnostics(policy llmprotocol.Policy) llmprotocol.Diagnostics {
	var diagnostics llmprotocol.Diagnostics
	appendProviderFieldOmissions(&diagnostics, policy, llmprotocol.OpenAIChatV1, metadata.fields, clientMetadataReason)
	return diagnostics
}
