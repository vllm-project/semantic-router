package protocolcodec

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A malformed tool-call completion in the Responses stream must carry the
// offending arguments on the error's cause, like the Chat/Anthropic paths.
// See issue #7302.
func TestResponsesToolDoneErrorCarriesArgumentEvidence(t *testing.T) {
	decoder := &responsesStreamDecoder{
		streamState: streamState{
			policy:          llmprotocol.DefaultPolicy(),
			toolArguments:   make(map[int][]byte),
			toolCalls:       map[int]llmprotocol.ToolCall{0: {Name: "lookup"}},
			toolCallIndexes: make(map[string]int),
		},
		toolArgumentsDone: make(map[int]bool),
	}
	index := 0
	err := decoder.validateResponsesToolDone(responsesEventWire{
		Type:        "response.function_call_arguments.done",
		OutputIndex: &index,
		Name:        "lookup",
		Arguments:   `not json at all`,
	})
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "invalid_stream_tool_arguments" {
		t.Fatalf("error = %v, want invalid_stream_tool_arguments", err)
	}
	cause := errors.Unwrap(protocolErr)
	if cause == nil || !strings.Contains(cause.Error(), "not json at all") {
		t.Fatalf("cause = %v, want the offending arguments summary", cause)
	}
}
