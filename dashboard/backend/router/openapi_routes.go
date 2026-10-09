package router

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const dashboardOpenAPIPath = "/openapi.json"

var dashboardOpenAPIInfo = apicontract.Info{
	Title:       "vLLM Semantic Router Dashboard API",
	Description: "Dashboard API for router configuration, setup, recipes, evaluation, and operational diagnostics",
	Version:     "1.0.0",
}

// openAPIExemptPrefixes lists registered namespaces that are not Dashboard API
// contracts. Each entry is reviewed: adding a prefix hides routes from the
// published document and from client generation.
var openAPIExemptPrefixes = map[string]string{
	"/embedded/": "embedded third-party UIs (Grafana, Jaeger, Prometheus, WizMap) proxied as HTML and upstream APIs",
	"/metrics/":  "Prometheus exposition proxy, not a JSON API",
}

func openAPIExemption(pattern string) (string, bool) {
	for prefix, reason := range openAPIExemptPrefixes {
		if strings.HasPrefix(pattern, prefix) {
			return reason, true
		}
	}
	return "", false
}

func dashboardOpenAPIRoutes(contracts []auth.RouteContract) []apicontract.Route {
	routes := make([]apicontract.Route, 0, len(contracts))
	for _, contract := range contracts {
		if _, exempt := openAPIExemption(contract.Pattern); exempt {
			continue
		}
		for _, policy := range contract.Policies {
			routes = append(routes, apicontract.Route{
				Pattern:               contract.Pattern,
				Method:                policy.Method,
				Permission:            policy.Permission,
				AdditionalPermissions: policy.AdditionalPermissions,
				Sensitivity:           string(policy.Sensitivity),
				ResourceOwner:         string(policy.ResourceOwner),
				AuditMode:             string(policy.AuditMode),
				AuditAction:           policy.AuditAction,
				Public:                policy.Public,
				CSRF:                  policy.CSRFRequired(),
				MaxBodyBytes:          policy.MaxBodyBytes,
				ProxyUpstream:         policy.ProxyUpstream,
				Operation:             policy.Operation,
			})
		}
	}
	return routes
}

func dashboardOpenAPISpec(contracts []auth.RouteContract) (apicontract.Spec, error) {
	return apicontract.Build(dashboardOpenAPIInfo, auth.SessionCookieName, dashboardOpenAPIRoutes(contracts))
}

type routeContractSource interface {
	Contracts() []auth.RouteContract
}

// registerOpenAPIRoute serves the document for the routes this process
// actually registered, so disabled features are absent. It renders on first
// request, after Setup has sealed the registry.
func registerOpenAPIRoute(mux routeRegistrar, source routeContractSource) {
	var (
		once     sync.Once
		document []byte
		buildErr error
	)
	contract := auth.ProtectedRoute(dashboardOpenAPIPath, auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet).
		Describe(http.MethodGet, apicontract.Operation{
			ID:          "getDashboardOpenAPI",
			Summary:     "Dashboard OpenAPI document",
			Description: "Describes the routes registered by this Dashboard process, including their route policies.",
			Responses: map[int]apicontract.Response{
				http.StatusOK:                  apicontract.JSONResponse[apicontract.Spec]("OpenAPI 3.0 document"),
				http.StatusInternalServerError: apicontract.TextResponse("The route registration could not be rendered"),
			},
		})
	registerRouteFunc(mux, contract, func(w http.ResponseWriter, r *http.Request) {
		once.Do(func() {
			spec, err := dashboardOpenAPISpec(source.Contracts())
			if err != nil {
				buildErr = err
				return
			}
			document, buildErr = json.Marshal(spec)
		})
		if buildErr != nil {
			log.Printf("Dashboard OpenAPI document unavailable: %v", buildErr)
			http.Error(w, "OpenAPI document unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(document)
	})
}
