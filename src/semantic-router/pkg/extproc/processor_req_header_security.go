package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// clientReservedRoutingHeaders are internal routing signals that must never be
// accepted from the downstream client. The request-header response removes the
// canonical name from Envoy's HeaderMap; captureRequestHeaders separately drops
// every case variant and duplicate before generic request context is built.
var clientReservedRoutingHeaders = []string{
	headers.SelectedModel,
}

func isClientReservedRoutingHeader(name string) bool {
	for _, reserved := range clientReservedRoutingHeaders {
		if strings.EqualFold(name, reserved) {
			return true
		}
	}
	return false
}

func requestHeaderSanitizationRemoveList() []string {
	remove := make([]string, 0, len(looperInternalRequestHeaders)+len(clientReservedRoutingHeaders))
	remove = append(remove, looperInternalRequestHeaders...)
	remove = append(remove, clientReservedRoutingHeaders...)
	return remove
}

func buildRequestHeaderSanitizationMutation() *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		RemoveHeaders: requestHeaderSanitizationRemoveList(),
	}
}
