package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	extProcAttributesNamespace    = "envoy.filters.http.ext_proc"
	upstreamHostMetadataAttribute = "xds.upstream_host_metadata"
	backendIdentityNamespace      = "semantic-router"
)

// captureUpstreamBackendIdentity records trusted metadata for the endpoint
// Envoy actually selected. Response attributes are emitted only on the first
// response-path message, so the identity remains request-scoped in ctx for
// buffered and streaming body validation.
func captureUpstreamBackendIdentity(req *ext_proc.ProcessingRequest, ctx *RequestContext) {
	if req == nil || ctx == nil || req.GetResponseHeaders() == nil {
		return
	}
	attributes := req.GetAttributes()[extProcAttributesNamespace]
	metadata := structField(attributes, upstreamHostMetadataAttribute)
	filterMetadata := structField(metadata, "filter_metadata")
	identity := structField(filterMetadata, backendIdentityNamespace)
	if identity == nil {
		return
	}
	ctx.UpstreamBackendName = strings.TrimSpace(stringField(identity, "backend_name"))
	ctx.UpstreamBackendType = strings.ToLower(strings.TrimSpace(stringField(identity, "backend_type")))
	ctx.AllowDynamoExtensions = ctx.UpstreamBackendType == "dynamo"
}

func structField(value *structpb.Struct, name string) *structpb.Struct {
	if value == nil {
		return nil
	}
	return value.GetFields()[name].GetStructValue()
}

func stringField(value *structpb.Struct, name string) string {
	if value == nil {
		return ""
	}
	return value.GetFields()[name].GetStringValue()
}
