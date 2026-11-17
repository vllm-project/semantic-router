package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// fallbackRequestHeaders are the headers of a fallback attempt the Router
// sends itself, besides its credential, which the connector sets last: the
// backend's static headers, with the client's value of each header the
// policy forwards laid over them.
func fallbackRequestHeaders(ctx *RequestContext, profile *config.ProviderProfile, orch *fallback.Orchestrator) map[string]string {
	out := make(map[string]string)
	if profile != nil {
		for name, value := range profile.ExtraHeaders {
			out[name] = value
		}
	}
	for _, name := range forwardedFallbackHeaders(ctx, orch) {
		for static := range out {
			if strings.EqualFold(static, name) {
				delete(out, static)
			}
		}
		out[name] = headerValueCI(ctx, name)
	}
	return out
}

// forwardedFallbackHeaders names, in lower case and in the policy's order,
// the client headers a fallback attempt the Router sends itself copies: each
// header in the policy's forward_request_headers that the client sent.
func forwardedFallbackHeaders(ctx *RequestContext, orch *fallback.Orchestrator) []string {
	if orch == nil {
		return nil
	}
	var names []string
	for _, listed := range orch.Policy().ForwardRequestHeaders {
		name := strings.ToLower(strings.TrimSpace(listed))
		if headerValueCI(ctx, name) != "" {
			names = append(names, name)
		}
	}
	return names
}
