package protocolcodec

import (
	"errors"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func upstreamSemanticValidationError(err error) error {
	var protocolError *llmprotocol.ProtocolError
	if errors.As(err, &protocolError) {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			protocolError.Code,
			protocolError.Message,
			err,
		)
	}
	return llmprotocol.NewError(
		llmprotocol.ErrorUpstreamUnavailable,
		"invalid_upstream_semantics",
		"upstream response contains an invalid semantic value",
		err,
	)
}

func decodeProviderErrorCategory(values ...string) llmprotocol.ErrorCategory {
	for _, value := range values {
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "invalid_request", "invalid_request_error", "bad_request", "validation_error", "request_too_large":
			return llmprotocol.ErrorInvalidRequest
		case "authentication", "authentication_error", "unauthorized":
			return llmprotocol.ErrorAuthentication
		case "permission", "permission_error", "permission_denied", "forbidden":
			return llmprotocol.ErrorPermission
		case "not_found", "not_found_error":
			return llmprotocol.ErrorNotFound
		case "conflict", "conflict_error":
			return llmprotocol.ErrorConflict
		case "rate_limited", "rate_limit_error", "too_many_requests":
			return llmprotocol.ErrorRateLimited
		case "upstream_timeout", "timeout", "timeout_error", "request_timeout":
			return llmprotocol.ErrorUpstreamTimeout
		case "upstream_unavailable", "api_error", "overloaded_error", "server_error":
			return llmprotocol.ErrorUpstreamUnavailable
		}
	}
	return llmprotocol.ErrorUpstreamUnavailable
}
