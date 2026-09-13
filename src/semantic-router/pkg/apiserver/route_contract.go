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

// EndpointContract is the semantic contract used by runtime discovery,
// OpenAPI generation, the website reference, and agent tooling.
type EndpointContract struct {
	Capability string        `json:"capability"`
	Plane      APIPlane      `json:"plane"`
	Audiences  []APIAudience `json:"audiences"`
	Stability  APIStability  `json:"stability"`
	Visibility APIVisibility `json:"visibility"`
	Deprecated bool          `json:"deprecated"`
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
	{Name: "observability", Description: "Inspect routing replays and metrics, and submit outcome evidence."},
	{Name: "storage", Description: "Manage Router-owned knowledge bases, memories, files, and vector stores."},
	{Name: "response-cache", Description: "Inspect and manage the response-cache service."},
	{Name: "context-compression", Description: "Inspect, preview, and manage context compression."},
	{Name: "diagnostics", Description: "Invoke low-level classifiers, embeddings, NLI, and similarity diagnostics."},
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
		routes[i].EndpointMetadata.EndpointContract = contract
	}
	return routes
}
