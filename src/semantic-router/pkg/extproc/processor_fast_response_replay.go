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
	addImmediateResponseHeader(response, headers.RouterReplayID, replayID)
}

func addImmediateResponseHeader(response *ext_proc.ProcessingResponse, key, value string) {
	if response == nil || value == "" {
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
				Key:      key,
				RawValue: []byte(value),
			},
		},
	)
}
