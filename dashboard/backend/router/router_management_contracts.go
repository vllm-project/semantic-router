package router

import (
	"net/http"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

const (
	maxRouterProbeBodyBytes    = 4 << 20
	maxRouterMutationBodyBytes = 16 << 20
)

const knowledgeBasePathPrefix = routercontract.GatewayPrefix + "/api/v1/storage/knowledge-bases"

func isKnowledgeBasePath(path string) bool {
	return path == knowledgeBasePathPrefix || strings.HasPrefix(path, knowledgeBasePathPrefix+"/")
}

func isGatewayProxyPath(path string) bool {
	return !isKnowledgeBasePath(path)
}

// routerManagementContracts projects the shared Router management gateway
// allowlist into route contracts, one per path, so the same declaration that
// decides what the Dashboard forwards also decides who may call it. The
// gateway templates use the ServeMux wildcard syntax already.
func routerManagementContracts(include func(path string) bool) []auth.RouteContract {
	byPath := map[string][]auth.RoutePolicy{}
	for _, policy := range routercontract.ManagementPolicies() {
		if !include(policy.Path) {
			continue
		}
		byPath[policy.Path] = append(byPath[policy.Path], managementRoutePolicy(policy))
	}
	paths := make([]string, 0, len(byPath))
	for path := range byPath {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	contracts := make([]auth.RouteContract, 0, len(paths))
	for _, path := range paths {
		contracts = append(contracts, auth.Route(path, byPath[path]...))
	}
	return contracts
}

func managementRoutePolicy(policy routercontract.ManagementPolicy) auth.RoutePolicy {
	primary, extra := policy.Permissions[0], policy.Permissions[1:]
	owner, sensitivity := managementRouteOwner(primary)
	var routePolicy auth.RoutePolicy
	switch {
	case policy.Mutation:
		routePolicy = auth.MutationPolicy(policy.Method, primary, managementAuditAction(policy.Path), sensitivity, owner, maxRouterMutationBodyBytes)
	case policy.Method == http.MethodGet || policy.Method == http.MethodHead:
		routePolicy = auth.ReadPolicy(policy.Method, primary, sensitivity, owner)
	default:
		routePolicy = auth.BoundedPolicy(policy.Method, primary, sensitivity, owner, maxRouterProbeBodyBytes)
	}
	routePolicy.Permissions = append(routePolicy.Permissions, extra...)
	return routePolicy
}

func managementRouteOwner(permission string) (auth.ResourceOwner, auth.Sensitivity) {
	switch permission {
	case auth.PermReplayRead:
		return auth.ResourceOwnerReplay, auth.SensitivitySecret
	case auth.PermFeedbackSubmit:
		return auth.ResourceOwnerFeedback, auth.SensitivitySensitive
	case auth.PermEvalRun:
		return auth.ResourceOwnerEvaluation, auth.SensitivitySensitive
	default:
		return auth.ResourceOwnerConfig, auth.SensitivitySensitive
	}
}

// managementAuditAction derives a stable action such as
// "router.storage.response_cache.flush" from the gateway template.
func managementAuditAction(path string) string {
	rest := strings.TrimPrefix(path, routercontract.GatewayPrefix+"/api/v1/")
	rest = strings.TrimPrefix(rest, routercontract.GatewayPrefix+"/")
	segments := make([]string, 0, 6)
	for _, segment := range strings.Split(rest, "/") {
		if segment == "" || strings.HasPrefix(segment, "{") {
			continue
		}
		segments = append(segments, strings.ReplaceAll(segment, "-", "_"))
	}
	return "router." + strings.Join(segments, ".")
}
