package routing

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// proxyControlHeaders are the proxy-control headers only a trusted proxy may
// set: the internal-request marker and the retry, timeout and tracing
// controls. An edge drops them from client requests, because Envoy-based
// layers behind the Router (sidecars, gateways in front of model servers)
// obey them from a caller they trust, which the Router is; dropping them keeps
// the Router's reliability policy the one retry and timeout authority.
//
// The list is the one Envoy's HTTP connection manager strips from every
// external request (Envoy v1.35, mutateRequestHeaders and
// cleanInternalHeaders), and stays that narrow. The local template leaves
// use_remote_address off, so Envoy's edge-only list (x-envoy-decorator-operation,
// x-envoy-downstream-service-cluster, x-envoy-downstream-service-node,
// x-envoy-original-path and x-envoy-original-host) passes in both modes.
var proxyControlHeaders = map[string]bool{
	headers.EnvoyInternal:                     true,
	headers.EnvoyRetriableStatusCodes:         true,
	headers.EnvoyRetriableHeaderNames:         true,
	headers.EnvoyRetryOn:                      true,
	headers.EnvoyRetryGRPCOn:                  true,
	headers.EnvoyMaxRetries:                   true,
	headers.EnvoyUpstreamAltStatName:          true,
	headers.EnvoyUpstreamRqTimeoutMs:          true,
	headers.EnvoyUpstreamRqPerTryTimeoutMs:    true,
	headers.EnvoyUpstreamRqTimeoutAltResponse: true,
	headers.EnvoyExpectedRqTimeoutMs:          true,
	headers.EnvoyForceTrace:                   true,
	headers.EnvoyIPTags:                       true,
	headers.EnvoyOriginalURL:                  true,
	headers.EnvoyHedgeOnPerTryTimeout:         true,
}

// IsProxyControlHeader reports whether name is a proxy-control header that
// the edge drops from client requests; Envoy does it before ext_proc sees a
// request, and the standalone frontend before the engine does.
func IsProxyControlHeader(name string) bool {
	return proxyControlHeaders[strings.ToLower(name)]
}
