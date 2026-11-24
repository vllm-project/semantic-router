package extproc

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A stream that failed with a protocol error must surface that error, with its
// code and bounded cause, from response() so the reconstruction log can name
// the failure. The client frame is built elsewhere and stays unchanged.
// See issue #7302.
func TestSemanticStreamResponseSurfacesProtocolFailure(t *testing.T) {
	failure := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable,
		"invalid_stream_tool_arguments", "upstream streamed tool arguments are not a JSON object",
		errors.New(`rejected tool arguments: "not json at all"`))

	state := &semanticResponseStreamState{
		terminal: true,
		failed:   failure,
	}
	_, err := state.response()
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) {
		t.Fatalf("response error = %v (%T), want a ProtocolError", err, err)
	}
	if protocolErr.Code != "invalid_stream_tool_arguments" {
		t.Fatalf("error code = %q, want invalid_stream_tool_arguments", protocolErr.Code)
	}
	cause := errors.Unwrap(protocolErr)
	if cause == nil || !strings.Contains(cause.Error(), "not json at all") {
		t.Fatalf("cause = %v, want the offending arguments summary", cause)
	}
}

// A generic incomplete stream still reports the original generic message; the
// change only surfaces a protocol failure when the stream actually had one.
func TestSemanticStreamResponseKeepsGenericIncompleteError(t *testing.T) {
	state := &semanticResponseStreamState{terminal: false}
	_, err := state.response()
	if err == nil || err.Error() != "semantic stream did not complete successfully" {
		t.Fatalf("response error = %v, want the generic incomplete error", err)
	}
}
