package routercontract

import (
	"net/http"
	"strings"
)

// ManagementPolicy is the deliberate Dashboard gateway subset of the Router
// API. It owns browser RBAC and readonly semantics, not the Router API catalog.
// A POST preview or inference probe is not a persistent mutation.
type ManagementPolicy struct {
	Method      string
	Path        string
	Permissions []string
	Mutation    bool
}

const (
	GatewayPrefix  = "/api/router"
	configRead     = "config.read"
	configWrite    = "config.write"
	evaluationRun  = "evaluation.run"
	replayRead     = "replay.read"
	feedbackSubmit = "feedback.submit"
)

var managementPolicies = buildManagementPolicies()

func buildManagementPolicies() []ManagementPolicy {
	var policies []ManagementPolicy
	addAll := func(method string, permissions []string, mutation bool, paths ...string) {
		for _, path := range paths {
			policies = append(policies, ManagementPolicy{Method: method, Path: GatewayPrefix + path, Permissions: append([]string(nil), permissions...), Mutation: mutation})
		}
	}
	add := func(method, permission string, mutation bool, paths ...string) {
		addAll(method, []string{permission}, mutation, paths...)
	}
	for _, method := range []string{http.MethodGet, http.MethodHead} {
		add(method, configRead, false, "/api/v1", "/openapi.json", "/docs", "/v1/models")
	}
	add(http.MethodGet, configRead, false,
		"/api/v1/config/hash", "/api/v1/config/schema",
		"/api/v1/inventory/models", "/api/v1/inventory/classifier", "/api/v1/inventory/embedding-models",
		"/api/v1/plugins", "/api/v1/plugins/{type}", "/api/v1/plugins/{type}/bindings",
		"/api/v1/diagnostics/models",
		"/api/v1/storage/response-cache/capabilities", "/api/v1/storage/response-cache/health", "/api/v1/storage/response-cache/stats",
		"/api/v1/plugins/context_compression/capabilities", "/api/v1/plugins/context_compression/health",
		"/api/v1/observability/plugins/context_compression/stats",
		"/api/v1/storage/knowledge-bases", "/api/v1/storage/knowledge-bases/{name}",
		"/api/v1/storage/knowledge-bases/{name}/map/metadata", "/api/v1/storage/knowledge-bases/{name}/map/data.ndjson")
	add(http.MethodGet, replayRead, false,
		"/api/v1/observability/replays", "/api/v1/observability/replays/aggregate", "/api/v1/observability/replays/trajectory", "/api/v1/observability/replays/{id}", "/api/v1/observability/audit")
	add(http.MethodPost, feedbackSubmit, true, "/api/v1/observability/outcomes")
	add(http.MethodPost, configRead, false,
		"/api/v1/plugins/system_prompt/preview", "/api/v1/plugins/request_params/preview", "/api/v1/plugins/header_mutation/preview", "/api/v1/plugins/fast_response/preview")
	add(http.MethodPost, evaluationRun, false,
		"/api/v1/routing/preview", "/api/v1/plugins/context_compression/preview",
		"/api/v1/plugins/response_jailbreak/preview", "/api/v1/plugins/hallucination/preview",
		"/api/v1/storage/response-cache/test",
		"/api/v1/diagnostics/models/labels", "/api/v1/diagnostics/models/label-scores", "/api/v1/diagnostics/models/tokens", "/api/v1/diagnostics/models/embeddings", "/api/v1/diagnostics/models/rerank",
		"/api/v1/diagnostics/classify/intent", "/api/v1/diagnostics/classify/pii", "/api/v1/diagnostics/classify/security", "/api/v1/diagnostics/classify/fact-check", "/api/v1/diagnostics/classify/user-feedback", "/api/v1/diagnostics/classify/combined", "/api/v1/diagnostics/classify/batch",
		"/api/v1/diagnostics/nli", "/api/v1/diagnostics/embeddings", "/api/v1/diagnostics/similarity", "/api/v1/diagnostics/similarity/batch")
	addAll(http.MethodPost, []string{evaluationRun, configRead}, false, "/api/v1/plugins/rag/preview", "/api/v1/plugins/tools/preview", "/api/v1/plugins/tool_selection/preview")
	add(http.MethodPost, configWrite, true, "/api/v1/storage/response-cache/invalidate", "/api/v1/storage/response-cache/flush", "/api/v1/storage/context-recovery/invalidate", "/api/v1/storage/knowledge-bases")
	add(http.MethodPut, configWrite, true, "/api/v1/storage/knowledge-bases/{name}")
	add(http.MethodDelete, configWrite, true, "/api/v1/storage/knowledge-bases/{name}")
	return policies
}

// ManagementPolicies returns a copy for contract tests and gateway inspection.
func ManagementPolicies() []ManagementPolicy {
	policies := append([]ManagementPolicy(nil), managementPolicies...)
	for i := range policies {
		policies[i].Permissions = append([]string(nil), policies[i].Permissions...)
	}
	return policies
}

// LookupManagement requires an exact method and a complete path template match.
// Parameters match one nonempty segment; no prefix grants or path normalization
// can widen the authorized surface. The caller passes URL.Path (already decoded).
func LookupManagement(method, path string) (ManagementPolicy, bool) {
	for _, policy := range managementPolicies {
		if policy.Method == method && matchManagementPath(policy.Path, path) {
			policy.Permissions = append([]string(nil), policy.Permissions...)
			return policy, true
		}
	}
	return ManagementPolicy{}, false
}

func matchManagementPath(template, path string) bool {
	parts, want := strings.Split(path, "/"), strings.Split(template, "/")
	if len(parts) != len(want) {
		return false
	}
	for i, part := range parts {
		if strings.HasPrefix(want[i], "{") && strings.HasSuffix(want[i], "}") {
			if part == "" || part == "." || part == ".." || strings.ContainsAny(part, "\\%") {
				return false
			}
		} else if part != want[i] {
			return false
		}
	}
	return true
}
