package extproc

import (
	"bytes"
	"context"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type dynamoBackendConfig struct {
	endpoints []config.VLLMEndpoint
}

func (cfg dynamoBackendConfig) GetEndpointsForModel(string) []config.VLLMEndpoint {
	return cfg.endpoints
}

func TestValidateDynamoRoutingHeadersAcceptsDocumentedHeadersCaseInsensitively(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"X-Tenant-Id":                       "tenant-a",
		headers.DynamoWorkerInstanceID:      "18446744073709551615",
		headers.DynamoPrefillInstanceID:     "1",
		headers.DynamoDPRank:                "4294967295",
		headers.DynamoPrefillDPRankLegacy:   "2",
		headers.DynamoRequestPriority:       "-7",
		headers.DynamoRequestStrictPriority: "3",
	}}
	if err := validateDynamoRoutingHeaders(ctx, llmprotocol.DefaultPolicy().Limits); err != nil {
		t.Fatalf("validateDynamoRoutingHeaders() error = %v", err)
	}
}

func TestValidateDynamoRoutingHeadersRejectsInvalidUnsignedValuesAndOversizedTenant(t *testing.T) {
	for _, test := range []struct {
		name    string
		headers map[string]string
		limits  llmprotocol.Limits
		code    string
	}{
		{"negative", map[string]string{headers.DynamoDPRank: "-1"}, llmprotocol.DefaultPolicy().Limits, "invalid_dynamo_routing_header"},
		{"overflow", map[string]string{headers.DynamoDPRank: "4294967296"}, llmprotocol.DefaultPolicy().Limits, "invalid_dynamo_routing_header"},
		{"not decimal", map[string]string{headers.DynamoWorkerInstanceID: "0x10"}, llmprotocol.DefaultPolicy().Limits, "invalid_dynamo_routing_header"},
		{"tenant", map[string]string{headers.DynamoTenantID: "12345"}, func() llmprotocol.Limits {
			limits := llmprotocol.DefaultPolicy().Limits
			limits.DynamoNVExtStringBytes = 4
			return limits
		}(), "dynamo_tenant_header_limit"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateDynamoRoutingHeaders(&RequestContext{Headers: test.headers}, test.limits)
			if err == nil || !strings.Contains(err.Error(), test.code) {
				t.Fatalf("error = %v, want code %q", err, test.code)
			}
		})
	}
}

func TestValidateDynamoBackendPoolRequiresEveryCandidateToBeDynamo(t *testing.T) {
	dynamoEnvelope := llmprotocol.Envelope{Dynamo: &llmprotocol.DynamoEnvelope{
		RequestNVExt: &llmprotocol.DynamoRequestNVExt{GreedSampling: llmprotocol.Bool(true)},
	}}
	for _, test := range []struct {
		name      string
		endpoints []config.VLLMEndpoint
		wantCode  string
	}{
		{"all dynamo", []config.VLLMEndpoint{{Name: "a", Type: "dynamo"}, {Name: "b", Type: " DYNAMO "}}, ""},
		{"mixed", []config.VLLMEndpoint{{Name: "a", Type: "dynamo"}, {Name: "b", Type: "vllm"}}, "unsupported_dynamo_nvext_backend"},
		{"unmarked", []config.VLLMEndpoint{{Name: "a"}}, "unsupported_dynamo_nvext_backend"},
		{"empty", nil, "unsupported_dynamo_nvext_backend"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateDynamoBackendPool(dynamoBackendConfig{endpoints: test.endpoints}, "model-a", nil, dynamoEnvelope)
			if test.wantCode == "" && err != nil {
				t.Fatalf("validateDynamoBackendPool() error = %v", err)
			}
			if test.wantCode != "" && (err == nil || !strings.Contains(err.Error(), test.wantCode)) {
				t.Fatalf("error = %v, want code %q", err, test.wantCode)
			}
		})
	}
}

func TestValidateDynamoBackendPoolDoesNotAffectOrdinaryRequests(t *testing.T) {
	config := dynamoBackendConfig{endpoints: []config.VLLMEndpoint{{Name: "vllm", Type: "vllm"}}}
	if err := validateDynamoBackendPool(config, "model-a", nil, llmprotocol.Envelope{}); err != nil {
		t.Fatalf("ordinary request rejected: %v", err)
	}
}

func TestValidateDynamoBackendPoolIncludesHeaderOnlyExtensions(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{headers.DynamoDPRank: "1"}}
	for _, test := range []struct {
		name      string
		endpoints []config.VLLMEndpoint
		wantError bool
	}{
		{"dynamo", []config.VLLMEndpoint{{Name: "dynamo", Type: "dynamo"}}, false},
		{"vllm", []config.VLLMEndpoint{{Name: "vllm", Type: "vllm"}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateDynamoBackendPool(dynamoBackendConfig{endpoints: test.endpoints}, "model-a", ctx, llmprotocol.Envelope{})
			if (err != nil) != test.wantError {
				t.Fatalf("validateDynamoBackendPool() error = %v, wantError = %v", err, test.wantError)
			}
		})
	}
}

func TestValidateDynamoResponseBackendRequiresActualDynamoEndpoint(t *testing.T) {
	envelope := llmprotocol.Envelope{Dynamo: &llmprotocol.DynamoEnvelope{
		ResponseNVExt: &llmprotocol.DynamoResponseNVExt{TokenIDs: []uint32{1}},
	}}
	for _, test := range []struct {
		name        string
		allowDynamo bool
		wantError   bool
	}{
		{"dynamo", true, false},
		{"vllm", false, true},
		{"missing identity", false, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateDynamoResponseBackend(&RequestContext{
				RequestModel: "model-a", AllowDynamoExtensions: test.allowDynamo,
			}, envelope)
			if (err != nil) != test.wantError {
				t.Fatalf("validateDynamoResponseBackend() error = %v, wantError = %v", err, test.wantError)
			}
		})
	}
}

func TestCaptureUpstreamBackendIdentityFromResponseAttributes(t *testing.T) {
	identity, err := structpb.NewStruct(map[string]any{
		upstreamHostMetadataAttribute: map[string]any{
			"filter_metadata": map[string]any{
				backendIdentityNamespace: map[string]any{
					"backend_name": "dynamo-a",
					"backend_type": " DYNAMO ",
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{},
		},
		Attributes: map[string]*structpb.Struct{extProcAttributesNamespace: identity},
	}
	ctx := &RequestContext{}

	captureUpstreamBackendIdentity(req, ctx)

	if ctx.UpstreamBackendName != "dynamo-a" || ctx.UpstreamBackendType != "dynamo" {
		t.Fatalf("captured backend identity = %q/%q", ctx.UpstreamBackendName, ctx.UpstreamBackendType)
	}
	if !ctx.AllowDynamoExtensions {
		t.Fatal("Dynamo endpoint should allow response extensions")
	}
}

func TestCaptureUpstreamBackendIdentityDisallowsNonDynamoExtensions(t *testing.T) {
	identity, err := structpb.NewStruct(map[string]any{
		upstreamHostMetadataAttribute: map[string]any{
			"filter_metadata": map[string]any{
				backendIdentityNamespace: map[string]any{
					"backend_name": "vllm-a",
					"backend_type": "vllm",
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := &ext_proc.ProcessingRequest{
		Request: &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{},
		},
		Attributes: map[string]*structpb.Struct{extProcAttributesNamespace: identity},
	}
	ctx := &RequestContext{AllowDynamoExtensions: true}

	captureUpstreamBackendIdentity(req, ctx)

	if ctx.AllowDynamoExtensions {
		t.Fatal("non-Dynamo endpoint should not allow response extensions")
	}
}

func TestDecodeClientResponseRejectsDynamoNVExtFromNonDynamoBackend(t *testing.T) {
	router := dynamoBoundaryTestRouter("vllm")
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.OpenAIChatV1,
		RequestModel:        "model-a",
		UpstreamBackendName: "vllm-a",
		UpstreamBackendType: "vllm",
	}
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"model-a","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"nvext":{"token_ids":[1]}}`)

	_, err := router.decodeClientResponse(body, ctx)
	if err == nil || !strings.Contains(err.Error(), "unexpected_dynamo_nvext_backend") {
		t.Fatalf("decodeClientResponse() error = %v, want unexpected_dynamo_nvext_backend", err)
	}
}

func TestSemanticStreamRejectsDynamoNVExtFromNonDynamoBackend(t *testing.T) {
	router := dynamoBoundaryTestRouter("vllm")
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.OpenAIChatV1,
		RequestModel:        "model-a",
		UpstreamBackendName: "vllm-a",
		UpstreamBackendType: "vllm",
		TraceContext:        context.Background(),
		SemanticRequest:     &llmprotocol.Request{},
	}
	body := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}],\"nvext\":{\"token_ids\":[1]}}\n\n")

	response := router.handleSemanticStreamingResponseBody(body, false, ctx)
	if !ctx.StreamingAborted {
		t.Fatal("stream carrying Dynamo nvext from a non-Dynamo backend was not aborted")
	}
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("aborted stream did not suppress the invalid upstream frame")
	}
	if strings.Contains(string(mutation.GetBody()), `"nvext"`) {
		t.Fatalf("invalid upstream nvext leaked to the client: %s", mutation.GetBody())
	}
}

func TestSemanticStreamPreservesDynamoRequestIDFromDynamoBackend(t *testing.T) {
	router := dynamoBoundaryTestRouter("dynamo")
	ctx := &RequestContext{
		SourceFormat:          llmprotocol.OpenAIChatV1,
		TargetFormat:          llmprotocol.OpenAIChatV1,
		RequestModel:          "model-a",
		UpstreamBackendName:   "dynamo-a",
		UpstreamBackendType:   "dynamo",
		AllowDynamoExtensions: true,
		TraceContext:          context.Background(),
		SemanticRequest:       &llmprotocol.Request{},
	}
	body := []byte("event: request_id\n: \"req-123\"\n\n")

	response := router.handleSemanticStreamingResponseBody(body, false, ctx)
	if ctx.StreamingAborted {
		t.Fatal("Dynamo request_id annotation was rejected")
	}
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil || !bytes.Equal(mutation.GetBody(), body) {
		t.Fatalf("forwarded request_id frame = %q, want %q", mutation.GetBody(), body)
	}
}

func TestSemanticStreamRejectsDynamoRequestIDFromNonDynamoBackend(t *testing.T) {
	router := dynamoBoundaryTestRouter("vllm")
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        llmprotocol.OpenAIChatV1,
		RequestModel:        "model-a",
		UpstreamBackendName: "vllm-a",
		UpstreamBackendType: "vllm",
		TraceContext:        context.Background(),
		SemanticRequest:     &llmprotocol.Request{},
	}
	body := []byte("event: request_id\n: \"req-123\"\n\n")

	response := router.handleSemanticStreamingResponseBody(body, false, ctx)
	if !ctx.StreamingAborted {
		t.Fatal("Dynamo request_id annotation from a non-Dynamo backend was not rejected")
	}
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("aborted stream did not suppress the request_id frame")
	}
	if bytes.Contains(mutation.GetBody(), []byte("req-123")) {
		t.Fatalf("invalid request_id annotation leaked to the client: %s", mutation.GetBody())
	}
}

func dynamoBoundaryTestRouter(backendType string) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"model-a": {PreferredEndpoints: []string{"backend"}},
		},
		VLLMEndpoints: []config.VLLMEndpoint{{Name: "backend", Type: backendType}},
	}}}
}
