package extproc

import (
	"context"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
)

func TestCaptureUpstreamEndpointAddressFromResponseHeader(t *testing.T) {
	ctx := &RequestContext{}
	actualPod := "10.0.1.7:8000"

	captureUpstreamEndpointAddress(ctx, &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: headers.VSRUpstreamHost, Value: actualPod},
	}}, nil)

	if ctx.UpstreamBackendAddress != actualPod {
		t.Fatalf("UpstreamBackendAddress = %q, want %q", ctx.UpstreamBackendAddress, actualPod)
	}
}

func TestCaptureUpstreamEndpointAddressPrefersAttributesOverHeader(t *testing.T) {
	ctx := &RequestContext{}
	attrPod := "10.0.2.9:8000"
	headerPod := "10.0.1.7:8000"
	attrs := map[string]*structpb.Struct{
		envoyUpstreamAddressAttribute: {
			Fields: map[string]*structpb.Value{
				"value": structpb.NewStringValue(attrPod),
			},
		},
	}

	captureUpstreamEndpointAddress(ctx, &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: headers.VSRUpstreamHost, Value: headerPod},
	}}, attrs)

	if ctx.UpstreamBackendAddress != attrPod {
		t.Fatalf("UpstreamBackendAddress = %q, want attribute %q", ctx.UpstreamBackendAddress, attrPod)
	}
}

func TestShouldWriteKVAddressRegistryRequiresCapturedUpstreamEndpoint(t *testing.T) {
	ctx := testKVAddressRoutingContext(t)
	ctx.SessionID = "sess-1"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.RequestModel = "qwen3-14b"
	ctx.UpstreamStatusCode = 200

	if shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = true, want false without captured upstream endpoint")
	}

	ctx.UpstreamBackendAddress = "10.0.1.7:8000"
	if !shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = false, want true after Envoy-selected endpoint is captured")
	}
}

func TestUpdateKVAddressRegistryUsesCapturedPodNotConfiguredService(t *testing.T) {
	reg := kvtransfer.NewMemoryAddressRegistry()
	router := &OpenAIRouter{kvAddressRegistryStore: reg}
	ctx := testKVAddressRoutingContext(t)
	ctx.RequestID = "req-lb"
	ctx.SessionID = "sess-lb"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.RequestModel = "qwen3-14b"
	ctx.TurnIndex = 0
	ctx.UpstreamStatusCode = 200
	ctx.UpstreamBackendAddress = "10.0.1.7:8000"

	router.updateKVAddressRegistry(ctx)

	got, err := reg.Lookup(context.Background(), kvAddressNamespace(ctx), routingSessionStateKey(ctx))
	if err != nil {
		t.Fatalf("Lookup() error = %v", err)
	}
	if got == nil {
		t.Fatal("Lookup() = nil, want KV address record")
	}
	if got.SourcePod != "10.0.1.7:8000" {
		t.Fatalf("SourcePod = %q, want load-balanced pod address", got.SourcePod)
	}
	if got.SourcePod == "qwen3-14b.svc.cluster.local:8000" {
		t.Fatal("SourcePod must not be the configured service / primary-backend address")
	}
}
