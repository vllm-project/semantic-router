package responsestore

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"

// ApplyListOptions enforces consistent limit constraints across all store implementations.
func ApplyListOptions(responses []*responseapi.StoredResponse, opts ListOptions) []*responseapi.StoredResponse {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if limit > MaxListLimit {
		limit = MaxListLimit
	}
	if len(responses) > limit {
		responses = responses[:limit]
	}
	return responses
}

// ApplyConvListOptions enforces consistent limit constraints across all store implementations.
func ApplyConvListOptions(convs []*responseapi.StoredConversation, opts ListOptions) []*responseapi.StoredConversation {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if limit > MaxListLimit {
		limit = MaxListLimit
	}
	if len(convs) > limit {
		convs = convs[:limit]
	}
	return convs
}
