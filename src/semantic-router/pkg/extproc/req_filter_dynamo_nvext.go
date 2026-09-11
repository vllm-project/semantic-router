package extproc

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var dynamoRoutingHeaderNames = []string{
	headers.DynamoWorkerInstanceID, headers.DynamoPrefillInstanceID,
	headers.DynamoDPRank, headers.DynamoPrefillDPRank,
	headers.DynamoRequestPriority, headers.DynamoRequestStrictPriority,
	headers.DynamoTenantID, headers.DynamoWorkerInstanceIDLegacy,
	headers.DynamoPrefillInstanceIDLegacy, headers.DynamoDPRankLegacy,
	headers.DynamoDataParallelRankLegacy, headers.DynamoPrefillDPRankLegacy,
}

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
	ctx *RequestContext,
	envelope llmprotocol.Envelope,
) error {
	if !hasDynamoRequestExtension(ctx, envelope) {
		return nil
	}
	if !modelHasOnlyDynamoBackends(cfg, model) {
		return unsupportedDynamoBackendError(model)
	}
	return nil
}

func hasDynamoRequestExtension(ctx *RequestContext, envelope llmprotocol.Envelope) bool {
	if envelope.Dynamo != nil &&
		(envelope.Dynamo.RequestNVExt != nil || envelope.Dynamo.RequestTopLevelCacheSalt != nil) {
		return true
	}
	return hasDynamoRoutingHeader(ctx)
}

func hasDynamoRoutingHeader(ctx *RequestContext) bool {
	for _, name := range dynamoRoutingHeaderNames {
		if strings.TrimSpace(headerValueCI(ctx, name)) != "" {
			return true
		}
	}
	return false
}

// snapshotDynamoRoutingHeaders copies only the documented, validated Dynamo
// routing inputs needed to preserve shadow request semantics. Other client
// headers remain outside the shadow trust boundary.
func snapshotDynamoRoutingHeaders(ctx *RequestContext) map[string]string {
	var result map[string]string
	for _, name := range dynamoRoutingHeaderNames {
		value := headerValueCI(ctx, name)
		if strings.TrimSpace(value) == "" {
			continue
		}
		if result == nil {
			result = make(map[string]string)
		}
		result[name] = value
	}
	return result
}

func modelHasOnlyDynamoBackends(
	cfg interface {
		GetEndpointsForModel(string) []config.VLLMEndpoint
	},
	model string,
) bool {
	endpoints := cfg.GetEndpointsForModel(model)
	if len(endpoints) == 0 {
		return false
	}
	for _, endpoint := range endpoints {
		if !strings.EqualFold(strings.TrimSpace(endpoint.Type), "dynamo") {
			return false
		}
	}
	return true
}

func validateDynamoResponseBackend(ctx *RequestContext, envelope llmprotocol.Envelope) error {
	if envelope.Dynamo == nil || envelope.Dynamo.ResponseNVExt == nil {
		return nil
	}
	if ctx != nil && ctx.AllowDynamoExtensions {
		return nil
	}
	return unexpectedDynamoResponseError(ctx)
}

func validateDynamoResponseEvents(ctx *RequestContext, events []llmprotocol.Event) error {
	for _, event := range events {
		if event.DynamoRequestID && (ctx == nil || !ctx.AllowDynamoExtensions) {
			return unexpectedDynamoStreamEventError(ctx)
		}
		if event.DynamoNVExt != nil && (ctx == nil || !ctx.AllowDynamoExtensions) {
			return unexpectedDynamoResponseError(ctx)
		}
	}
	return nil
}

func unexpectedDynamoStreamEventError(ctx *RequestContext) error {
	model := ""
	backend := ""
	if ctx != nil {
		model = ctx.RequestModel
		backend = ctx.UpstreamBackendName
	}
	return llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"unexpected_dynamo_request_id_backend",
		fmt.Sprintf("model %q returned a Dynamo request_id SSE event from non-Dynamo backend %q", model, backend),
		nil,
	)
}

func unsupportedDynamoBackendError(model string) error {
	return llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"unsupported_dynamo_nvext_backend",
		fmt.Sprintf("model %q is not backed exclusively by Dynamo endpoints", model),
		nil,
	)
}

func unexpectedDynamoResponseError(ctx *RequestContext) error {
	model := ""
	backend := ""
	if ctx != nil {
		model = ctx.RequestModel
		backend = ctx.UpstreamBackendName
	}
	return llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"unexpected_dynamo_nvext_backend",
		fmt.Sprintf("model %q returned Dynamo nvext from non-Dynamo backend %q", model, backend),
		nil,
	)
}
