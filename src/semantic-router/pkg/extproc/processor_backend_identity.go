package extproc

import (
	"strings"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/encoding/prototext"
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
	metadataText := attributes.GetFields()[upstreamHostMetadataAttribute].GetStringValue()
	if metadataText == "" {
		return
	}
	var metadata corev3.Metadata
	if err := prototext.Unmarshal([]byte(metadataText), &metadata); err != nil {
		return
	}
	identity := metadata.GetFilterMetadata()[backendIdentityNamespace]
	if identity == nil {
		return
	}
	ctx.UpstreamBackendName = strings.TrimSpace(stringField(identity, "backend_name"))
	ctx.UpstreamBackendType = strings.ToLower(strings.TrimSpace(stringField(identity, "backend_type")))
	ctx.AllowDynamoExtensions = ctx.UpstreamBackendType == "dynamo"
}

func stringField(value *structpb.Struct, name string) string {
	if value == nil {
		return ""
	}
	return value.GetFields()[name].GetStringValue()
}
