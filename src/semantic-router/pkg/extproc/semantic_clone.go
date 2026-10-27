package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// cloneSemanticRequestForReplay detaches every mutable field before a request is
// retained for a later provider attempt. JSON round-tripping deliberately uses
// the protocol-neutral type itself, so new semantic fields are copied without a
// second hand-maintained clone implementation.
func cloneSemanticRequestForReplay(request *llmprotocol.Request) (*llmprotocol.Request, error) {
	if request == nil {
		return nil, fmt.Errorf("semantic request is unavailable")
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("clone semantic request: %w", err)
	}
	var cloned llmprotocol.Request
	if err := json.Unmarshal(encoded, &cloned); err != nil {
		return nil, fmt.Errorf("clone semantic request: %w", err)
	}
	// A nil json.RawMessage marshals as JSON null, then unmarshals as the
	// non-nil byte slice "null". Preserve absence explicitly: protocol
	// validation treats a present raw field as caller-provided data.
	if request.OutputFormat.Schema == nil {
		cloned.OutputFormat.Schema = nil
	}
	if request.ChatTemplateKwargs == nil {
		cloned.ChatTemplateKwargs = nil
	}
	if request.ContextManagement == nil {
		cloned.ContextManagement = nil
	}
	for i := range request.Tools {
		if request.Tools[i].InputSchema == nil {
			cloned.Tools[i].InputSchema = nil
		}
	}
	// Preserve-mode passthrough fields ride the messages with json:"-", so
	// the round-trip above drops them. Copy them per message: a later
	// provider attempt re-encodes from this clone and must still forward
	// every captured field on its own message.
	for i := range request.Messages {
		cloned.Messages[i].Unknown = clonePassthroughFields(request.Messages[i].Unknown)
	}
	for i := range request.Instructions {
		cloned.Instructions[i].Unknown = clonePassthroughFields(request.Instructions[i].Unknown)
	}
	return &cloned, nil
}

func clonePassthroughFields(fields map[string]json.RawMessage) map[string]json.RawMessage {
	if fields == nil {
		return nil
	}
	cloned := make(map[string]json.RawMessage, len(fields))
	for name, value := range fields {
		cloned[name] = append(json.RawMessage(nil), value...)
	}
	return cloned
}

func cloneSemanticResponseForCommit(response *llmprotocol.Response) (*llmprotocol.Response, error) {
	if response == nil {
		return nil, fmt.Errorf("semantic response is unavailable")
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		return nil, fmt.Errorf("clone semantic response: %w", err)
	}
	var cloned llmprotocol.Response
	if err := json.Unmarshal(encoded, &cloned); err != nil {
		return nil, fmt.Errorf("clone semantic response: %w", err)
	}
	return &cloned, nil
}
