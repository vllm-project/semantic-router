package extproc

import (
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func addRouterReplayHeaderToImmediateResponse(
	response *ext_proc.ProcessingResponse,
	replayID string,
) {
	if response == nil || replayID == "" {
		return
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return
	}
	if immediate.Headers == nil {
		immediate.Headers = &ext_proc.HeaderMutation{}
	}
	immediate.Headers.SetHeaders = append(
		immediate.Headers.SetHeaders,
		&core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.RouterReplayID,
				RawValue: []byte(replayID),
			},
		},
	)
}
