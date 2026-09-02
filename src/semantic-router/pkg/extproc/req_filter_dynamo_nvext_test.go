package extproc

import (
	"strings"
	"testing"

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
			err := validateDynamoBackendPool(dynamoBackendConfig{endpoints: test.endpoints}, "model-a", dynamoEnvelope)
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
	if err := validateDynamoBackendPool(config, "model-a", llmprotocol.Envelope{}); err != nil {
		t.Fatalf("ordinary request rejected: %v", err)
	}
}
