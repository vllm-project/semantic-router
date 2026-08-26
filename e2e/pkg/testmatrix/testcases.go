package testmatrix

// RouterSmoke is the smallest shared router check that heavy environments reuse.
var RouterSmoke = []string{
	"chat-completions-request",
}

// BaselineRouterContract is the canonical full router contract owned by the kubernetes baseline profile.
var BaselineRouterContract = []string{
	"chat-completions-request",
	"anthropic-messages-request",
	"anthropic-messages-protocol-headers",
	"anthropic-messages-response-shape",
	"anthropic-messages-streaming",
	"apiserver-runtime-config-endpoints",
	"apiserver-classification-endpoints",
	"chat-completions-stress-request",
	"domain-classify",
	"semantic-cache",
	// NLI polarity tier of the semantic cache (issue #2751)
	"semantic-cache-polarity",
	"pii-detection",
	"jailbreak-detection",
	"decision-priority-selection",
	"plugin-chain-execution",
	"tool-selection",
	"rule-condition-logic",
	"decision-fallback-behavior",
	"plugin-config-variations",
	"chat-completions-progressive-stress",
	"anthropic-passthrough-openai-regression",
	// Retention directive response-header contract (issue #2009)
	"retention-directive",
	// Looper aggregate latency/token-usage response-header contract (issue #2694)
	"looper-latency-token-headers",
	// Entrypoint virtual names select routing recipes (issue #2331)
	"entrypoint-recipe-routing",
	// json_schema response_format survives auto-routing model rewrite (issue #3024)
	"chat-completions-structured-output",
	// Session observability
	"session-telemetry-metrics",
	"session-pricing-chat-completions",
	"session-pricing-response-api",
}

// DashboardContract is the canonical E2E contract for the dashboard API surface.
var DashboardContract = []string{
	// Core API
	"dashboard-health",
	"dashboard-status",
	// Config endpoints
	"dashboard-config-read",
	"dashboard-deploy-preview",
	"dashboard-config-versions",
	"dashboard-deploy-invalid-yaml",
	// Evaluation endpoints (tasks/CRUD require CGO — only datasets works without it)
	"dashboard-eval-datasets",
	// Workflow persistence survives dashboard pod restart (requires dashboard PVC)
	"dashboard-restart-recovery",
	// Security Policy RBAC + ratelimit apply
	"security-policy-apply",
}

// AnthropicShimContract is the test suite that exercises the Anthropic-
// shaped backend (llama.cpp + anthropic-shim). These tests require the
// anthropic-shim profile and will not run correctly against the baseline
// OpenAI-shaped backends because they assert on Anthropic-specific
// behaviour such as cache-token synthesis and stop-reason mapping.
var AnthropicShimContract = []string{
	"anthropic-messages-cache-cycle",
	"anthropic-messages-stop-sequence",
	// /v1/responses streaming must emit Response API SSE on Anthropic-format
	// backends instead of leaking chat.completion.chunk frames (issue #3013)
	"anthropic-response-api-streaming",
}

// Combine preserves order while removing duplicate testcase names.
func Combine(groups ...[]string) []string {
	size := 0
	for _, group := range groups {
		size += len(group)
	}

	combined := make([]string, 0, size)
	seen := make(map[string]struct{}, size)
	for _, group := range groups {
		for _, name := range group {
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			combined = append(combined, name)
		}
	}

	return combined
}
