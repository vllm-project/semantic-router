package extproc

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// validateDynamoRoutingHeaders validates the documented Dynamo routing header
// types without folding their values into the request body. ExtProc forwards
// the headers unchanged, so the Dynamo frontend remains responsible for its
// documented header-over-body precedence. In particular, x-tenant-id remains
// routing input and is never promoted to trusted authentication state here.
func validateDynamoRoutingHeaders(ctx *RequestContext, limits llmprotocol.Limits) error {
	if tenantID := strings.TrimSpace(headerValueCI(ctx, headers.DynamoTenantID)); tenantID != "" &&
		limits.DynamoNVExtStringBytes > 0 && len(tenantID) > limits.DynamoNVExtStringBytes {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "dynamo_tenant_header_limit",
			"Dynamo x-tenant-id exceeds the configured limit", nil,
		)
	}
	for _, header := range []string{
		headers.DynamoWorkerInstanceID, headers.DynamoPrefillInstanceID,
		headers.DynamoWorkerInstanceIDLegacy, headers.DynamoPrefillInstanceIDLegacy,
	} {
		if err := validateDynamoUnsignedHeader(ctx, header, 64); err != nil {
			return err
		}
	}
	for _, header := range []string{
		headers.DynamoDPRank, headers.DynamoPrefillDPRank,
		headers.DynamoDPRankLegacy, headers.DynamoDataParallelRankLegacy,
		headers.DynamoPrefillDPRankLegacy,
	} {
		if err := validateDynamoUnsignedHeader(ctx, header, 32); err != nil {
			return err
		}
	}
	if err := validateDynamoSignedHeader(ctx, headers.DynamoRequestPriority, 32); err != nil {
		return err
	}
	if err := validateDynamoUnsignedHeader(ctx, headers.DynamoRequestStrictPriority, 32); err != nil {
		return err
	}
	return nil
}

func validateDynamoSignedHeader(ctx *RequestContext, name string, bits int) error {
	value := strings.TrimSpace(headerValueCI(ctx, name))
	if value == "" {
		return nil
	}
	if _, err := strconv.ParseInt(value, 10, bits); err != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "invalid_dynamo_routing_header",
			fmt.Sprintf("Dynamo routing header %s must be a signed base-10 integer", name), err,
		)
	}
	return nil
}

func validateDynamoUnsignedHeader(ctx *RequestContext, name string, bits int) error {
	value := strings.TrimSpace(headerValueCI(ctx, name))
	if value == "" {
		return nil
	}
	if _, err := strconv.ParseUint(value, 10, bits); err != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorInvalidRequest, "invalid_dynamo_routing_header",
			fmt.Sprintf("Dynamo routing header %s must be an unsigned base-10 integer", name), err,
		)
	}
	return nil
}

func validateDynamoBackendPool(
	cfg interface {
		GetEndpointsForModel(string) []config.VLLMEndpoint
	},
	model string,
	envelope llmprotocol.Envelope,
) error {
	if !hasDynamoRequestExtension(envelope) {
		return nil
	}
	endpoints := cfg.GetEndpointsForModel(model)
	for _, endpoint := range endpoints {
		if !strings.EqualFold(strings.TrimSpace(endpoint.Type), "dynamo") {
			return unsupportedDynamoBackendError(model)
		}
	}
	if len(endpoints) == 0 {
		return unsupportedDynamoBackendError(model)
	}
	return nil
}

func hasDynamoRequestExtension(envelope llmprotocol.Envelope) bool {
	return envelope.Dynamo != nil &&
		(envelope.Dynamo.RequestNVExt != nil || envelope.Dynamo.RequestTopLevelCacheSalt != nil)
}

func unsupportedDynamoBackendError(model string) error {
	return llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"unsupported_dynamo_nvext_backend",
		fmt.Sprintf("model %q is not backed exclusively by Dynamo endpoints", model),
		nil,
	)
}
