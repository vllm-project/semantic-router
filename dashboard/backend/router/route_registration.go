package router

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

// routeRegistrar lets focused handler tests use a plain ServeMux. Production
// passes PolicyMux, which refuses any registration without a complete policy.
type routeRegistrar interface {
	Handle(string, http.Handler)
	HandleFunc(string, func(http.ResponseWriter, *http.Request))
	ServeHTTP(http.ResponseWriter, *http.Request)
}

func registerRoute(mux routeRegistrar, contract auth.RouteContract, handler http.Handler) {
	if policies, ok := mux.(*auth.PolicyMux); ok {
		policies.HandlePolicy(contract, handler)
		return
	}
	mux.Handle(contract.Pattern, handler)
}

func registerRouteFunc(mux routeRegistrar, contract auth.RouteContract, handler http.HandlerFunc) {
	registerRoute(mux, contract, handler)
}

func registerRouteGroup(mux routeRegistrar, contracts []auth.RouteContract, handler http.Handler) {
	if policies, ok := mux.(*auth.PolicyMux); ok {
		policies.HandleGroup(contracts, handler)
		return
	}
	for _, contract := range contracts {
		mux.Handle(contract.Pattern, handler)
	}
}

const knowledgeBaseGatewayPrefix = "/api/router/api/v1/storage/knowledge-bases"

func managementRouteContracts(includeKnowledgeBases bool) []auth.RouteContract {
	byPath := make(map[string]*auth.RouteContract)
	order := make([]string, 0)
	for _, management := range routercontract.ManagementPolicies() {
		knowledgeBase := strings.HasPrefix(management.Path, knowledgeBaseGatewayPrefix)
		if knowledgeBase != includeKnowledgeBases {
			continue
		}
		contract, ok := byPath[management.Path]
		if !ok {
			contract = &auth.RouteContract{Pattern: management.Path}
			byPath[management.Path] = contract
			order = append(order, management.Path)
		}
		owner := auth.ResourceOwnerConfig
		sensitivity := auth.SensitivitySensitive
		switch management.Permissions[0] {
		case auth.PermReplayRead:
			owner, sensitivity = auth.ResourceOwnerReplay, auth.SensitivitySecret
		case auth.PermFeedbackSubmit:
			owner, sensitivity = auth.ResourceOwnerFeedback, auth.SensitivitySecret
		case auth.PermEvalRun:
			owner = auth.ResourceOwnerEvaluation
		}
		var policy auth.RoutePolicy
		if management.Mutation {
			policy = auth.MutationPolicy(management.Method, management.Permissions[0], "router.management."+strings.ToLower(management.Method), sensitivity, owner, 16<<20)
		} else {
			policy = auth.ReadPolicy(management.Method, management.Permissions[0], sensitivity, owner)
			if management.Method == http.MethodPost {
				policy.MaxBodyBytes = 16 << 20
			}
		}
		policy.AdditionalPermissions = append([]string(nil), management.Permissions[1:]...)
		contract.Policies = append(contract.Policies, policy)
	}
	contracts := make([]auth.RouteContract, 0, len(order))
	for _, path := range order {
		contracts = append(contracts, *byPath[path])
	}
	return contracts
}

func registerKnowledgeBaseRoutes(mux routeRegistrar, cfg *config.Config, store *recipe.Store) {
	handler := handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store)
	registerRouteGroup(mux, managementRouteContracts(true), handler)
}
