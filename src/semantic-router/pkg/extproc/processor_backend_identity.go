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
	setUpstreamBackendIdentity(ctx, "", "")

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
	setUpstreamBackendIdentity(ctx, backendName, backendType)
}

// captureNativeBackendIdentity resolves a transport-selected endpoint against
// trusted router configuration. Client/provider headers cannot grant permission.
func (r *OpenAIRouter) captureNativeBackendIdentity(name string, ctx *RequestContext) {
	setUpstreamBackendIdentity(ctx, "", "")
	if r == nil || r.Config == nil || name == "" {
		return
	}
	endpoint, found := r.Config.GetEndpointByName(name)
	if !found {
		return
	}
	setUpstreamBackendIdentity(ctx, endpoint.Name, endpoint.Type)
}

func setUpstreamBackendIdentity(ctx *RequestContext, name, backendType string) {
	if ctx == nil {
		return
	}
	ctx.UpstreamBackendName = ""
	ctx.UpstreamBackendType = ""
	ctx.AllowDynamoExtensions = false
	name = strings.TrimSpace(name)
	backendType = strings.ToLower(strings.TrimSpace(backendType))
	if name == "" || backendType == "" {
		return
	}
	ctx.UpstreamBackendName = name
	ctx.UpstreamBackendType = backendType
	ctx.AllowDynamoExtensions = backendType == "dynamo"
}
