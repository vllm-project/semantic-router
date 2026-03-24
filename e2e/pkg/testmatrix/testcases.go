package testmatrix

// RouterSmoke is the smallest shared router check that heavy environments reuse.
var RouterSmoke = []string{
	"chat-completions-request",
}

// BaselineRouterContract is the canonical full router contract owned by ai-gateway.
var BaselineRouterContract = []string{
	"chat-completions-request",
	"apiserver-runtime-config-endpoints",
	"apiserver-classification-endpoints",
	"chat-completions-stress-request",
	"domain-classify",
	"semantic-cache",
	"pii-detection",
	"jailbreak-detection",
	"decision-priority-selection",
	"plugin-chain-execution",
	"rule-condition-logic",
	"decision-fallback-behavior",
	"plugin-config-variations",
	"chat-completions-progressive-stress",
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
