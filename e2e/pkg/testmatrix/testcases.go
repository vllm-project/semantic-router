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
	"pii-detection",
	"jailbreak-detection",
	"decision-priority-selection",
	"plugin-chain-execution",
	"tool-selection",
	"rule-condition-logic",
	"decision-fallback-behavior",
	"plugin-config-variations",
	"chat-completions-progressive-stress",
	"protocol-codec-openai-regression",
	// Retention directive response-header contract (issue #2009)
	"retention-directive",
	// Looper aggregate latency/token-usage response-header contract (issue #2694)
	"looper-latency-token-headers",
	// Entrypoint virtual names select routing recipes (issue #2331)
	"entrypoint-recipe-routing",
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
	// Evaluation endpoints (tasks/CRUD require CGO — only datasets works without it)
	"dashboard-eval-datasets",
	// Router-native identity, access, quota, usage, and credential lifecycle
	"dashboard-managed-access-lifecycle",
	// Workflow persistence survives dashboard pod restart (requires dashboard PVC)
	"dashboard-restart-recovery",
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
