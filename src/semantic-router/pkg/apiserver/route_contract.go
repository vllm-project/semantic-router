//go:build !windows && cgo

package apiserver

// APIPlane separates traffic contracts even though the current process serves
// them from one HTTP listener. The public inference data plane remains behind
// Envoy; this catalog describes the Router's management and diagnostic API.
type APIPlane string

const (
	APIPlaneInfrastructure APIPlane = "infrastructure"
	APIPlaneManagement     APIPlane = "management"
	APIPlaneDiagnostic     APIPlane = "diagnostic"
	APIPlaneData           APIPlane = "data"
)

// APIAudience identifies the intended caller for progressive discovery.
type APIAudience string

const (
	APIAudienceAgent    APIAudience = "agent"
	APIAudienceOperator APIAudience = "operator"
	APIAudienceClient   APIAudience = "client"
	APIAudienceInternal APIAudience = "internal"
)

// APIStability is the compatibility promise made by one operation.
type APIStability string

const (
	APIStabilityStable       APIStability = "stable"
	APIStabilityExperimental APIStability = "experimental"
)

// APIVisibility controls whether an operation belongs in the compact primary
// surface or only in expanded discovery.
type APIVisibility string

const (
	APIVisibilityPrimary  APIVisibility = "primary"
	APIVisibilityAdvanced APIVisibility = "advanced"
)

// PluginOperationContract associates an operation with its canonical plugin.
// Shared storage and observability routes may serve multiple plugins.
type PluginOperationContract struct {
	Plugin string `json:"plugin"`
	Mode   string `json:"mode"`
}

// EndpointContract is shared by runtime discovery, OpenAPI, and agent tooling.
type EndpointContract struct {
	PluginOperations []PluginOperationContract `json:"plugin_operations,omitempty"`
	Capability       string                    `json:"capability"`
	Plane            APIPlane                  `json:"plane"`
	Audiences        []APIAudience             `json:"audiences"`
	Stability        APIStability              `json:"stability"`
	Visibility       APIVisibility             `json:"visibility"`
	Deprecated       bool                      `json:"deprecated"`
}

type capabilityDefinition struct {
	Name        string
	Description string
}

var capabilityRegistry = []capabilityDefinition{
	{Name: "system", Description: "Health, readiness, and API contract discovery."},
	{Name: "config", Description: "Validate, inspect, apply, version, and roll back Router configuration and Recipes."},
	{Name: "routing", Description: "Preview routing behavior without invoking a generation backend."},
	{Name: "inventory", Description: "Inspect configured and loaded model and classifier resources."},
	{Name: "observability", Description: "Inspect routing replays, metrics, and management audit; submit outcome evidence."},
	{Name: "storage", Description: "Manage Router-owned knowledge bases, memories, files, vector stores, cache partitions, and context recovery."},
	{Name: "plugins", Description: "Discover recipe-scoped plugin bindings, dependencies, and typed behavior previews."},
	{Name: "diagnostics", Description: "Inspect and invoke recipe-scoped prepared models, classifiers, embeddings, NLI, and rerankers."},
}

func routeContract(
	capability string,
	plane APIPlane,
	visibility APIVisibility,
	audiences ...APIAudience,
) EndpointContract {
	return EndpointContract{
		Capability: capability,
		Plane:      plane,
		Audiences:  append([]APIAudience(nil), audiences...),
		Stability:  APIStabilityStable,
		Visibility: visibility,
	}
}

func applyRouteContract(routes []apiRoute, contract EndpointContract) []apiRoute {
	for i := range routes {
		operations := routes[i].PluginOperations
		routes[i].EndpointMetadata.EndpointContract = contract
		routes[i].PluginOperations = operations
	}
	return routes
}
