package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

const dashboardCapabilitiesPath = "/api/v1/capabilities"

func capabilitiesResponseSchema() apicontract.Schema {
	schema := apicontract.SchemaFor[handlers.CapabilitiesResponse]()
	capability := *schema.Properties["capabilities"].Items
	states := make([]string, 0, len(handlers.CapabilityStates))
	for _, state := range handlers.CapabilityStates {
		states = append(states, string(state))
	}
	setEnum(&capability, "state", states)
	setEnum(&capability, "reason", handlers.CapabilityReasons)
	capabilities := schema.Properties["capabilities"]
	capabilities.Items = &capability
	schema.Properties["capabilities"] = capabilities
	return schema
}

func setEnum(schema *apicontract.Schema, property string, values []string) {
	field := schema.Properties[property]
	field.Enum = values
	schema.Properties[property] = field
}

var capabilitiesOperation = apicontract.Operation{
	ID:      "listCapabilities",
	Summary: "Dashboard capability states for the caller",
	Description: "state is the first failing dimension in the order enabled, configured, authorized, reachable, " +
		"degraded; reason is a stable code for it. Health checks are cached, so checkedAt can lag by one refresh interval.",
	Responses: map[int]apicontract.Response{
		http.StatusOK: {
			Description: "One entry per capability",
			Content:     map[string]apicontract.Media{"application/json": {Schema: ptr(capabilitiesResponseSchema())}},
		},
	},
}

func ptr[T any](value T) *T { return &value }

func registerCapabilityRoutes(mux routeRegistrar, cfg *config.Config, setupResolver *setupmode.Resolver, prober *handlers.CapabilityProber) {
	contract := auth.ProtectedRoute(dashboardCapabilitiesPath, auth.PermSessionRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet).
		Describe(http.MethodGet, capabilitiesOperation)
	registerRouteFunc(mux, contract, handlers.CapabilitiesHandler(cfg, setupResolver, prober))
	log.Printf("Capabilities API endpoint registered: %s", dashboardCapabilitiesPath)
}
