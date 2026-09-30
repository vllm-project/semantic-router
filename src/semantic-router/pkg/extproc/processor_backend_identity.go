package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

const (
	extProcAttributesNamespace   = "envoy.filters.http.ext_proc"
	upstreamBackendNameAttribute = `xds.upstream_host_metadata.filter_metadata["semantic-router"]["backend_name"]`
	upstreamBackendTypeAttribute = `xds.upstream_host_metadata.filter_metadata["semantic-router"]["backend_type"]`
)

// captureUpstreamBackendIdentity records trusted metadata for the endpoint
// Envoy actually selected. Response attributes are emitted only on the first
// response-path message, so the identity remains request-scoped in ctx for
// buffered and streaming body validation.
func captureUpstreamBackendIdentity(req *ext_proc.ProcessingRequest, ctx *RequestContext) {
	if req == nil || ctx == nil || req.GetResponseHeaders() == nil {
		return
	}
	// Response attributes are authoritative for the endpoint Envoy actually
	// selected. Clear any earlier value before parsing so missing or malformed
	// metadata fails closed, including after an upstream retry.
	ctx.UpstreamBackendName = ""
	ctx.UpstreamBackendType = ""
	ctx.AllowDynamoExtensions = false

	attributes := req.GetAttributes()[extProcAttributesNamespace]
	if attributes == nil {
		return
	}
	backendName := strings.TrimSpace(
		attributes.GetFields()[upstreamBackendNameAttribute].GetStringValue(),
	)
	backendType := strings.ToLower(strings.TrimSpace(
		attributes.GetFields()[upstreamBackendTypeAttribute].GetStringValue(),
	))
	if backendName == "" || backendType == "" {
		return
	}
	ctx.UpstreamBackendName = backendName
	ctx.UpstreamBackendType = backendType
	ctx.AllowDynamoExtensions = backendType == "dynamo"
}
