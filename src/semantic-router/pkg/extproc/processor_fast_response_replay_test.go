package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestAddRouterReplayHeaderToImmediateResponse(t *testing.T) {
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{},
		},
	}

	addRouterReplayHeaderToImmediateResponse(response, "replay-123")

	immediate := response.GetImmediateResponse()
	if immediate.Headers == nil || len(immediate.Headers.SetHeaders) != 1 {
		t.Fatalf("headers = %#v", immediate.Headers)
	}
	header := immediate.Headers.SetHeaders[0].Header
	if header.Key != headers.RouterReplayID ||
		string(header.RawValue) != "replay-123" {
		t.Fatalf("header = %#v", header)
	}
}
