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
	for i := range request.Tools {
		if request.Tools[i].InputSchema == nil {
			cloned.Tools[i].InputSchema = nil
		}
	}
	return &cloned, nil
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
