package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func addRouterReplayHeaderToImmediateResponse(
	response *ext_proc.ProcessingResponse,
	replayID string,
) {
	appendImmediateResponseHeader(response, headers.RouterReplayID, replayID)
}
