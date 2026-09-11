//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

const (
	apiDiscoveryViewIndex      = "index"
	apiDiscoveryViewOperations = "operations"
)

// APIOverviewResponse is the progressive discovery response served by
// GET /api/v1. The default response is intentionally compact; callers request
// one capability or view=operations before loading individual operations.
type APIOverviewResponse struct {
	Service      string              `json:"service"`
	Version      string              `json:"version"`
	Description  string              `json:"description"`
	Capabilities []APICapabilityInfo `json:"capabilities"`
	Endpoints    []EndpointInfo      `json:"endpoints,omitempty"`
	Links        map[string]string   `json:"links"`
}

// APICapabilityInfo summarizes one cohesive API surface.
type APICapabilityInfo struct {
	Name              string `json:"name"`
	Description       string `json:"description"`
	OperationCount    int    `json:"operation_count"`
	PrimaryOperations int    `json:"primary_operations"`
	Discovery         string `json:"discovery"`
	OpenAPI           string `json:"openapi"`
}

type routeSelection struct {
	Capability string
	Audience   APIAudience
	Plane      APIPlane
	Visibility APIVisibility
}

// handleAPIOverview handles progressive API discovery. No query parameters
// returns only the capability directory. Selecting a capability/filter, or
// view=operations, returns the matching operations as well.
func (s *ClassificationAPIServer) handleAPIOverview(w http.ResponseWriter, r *http.Request) {
	selection, err := parseRouteSelection(r.URL.Query())
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_API_FILTER", err.Error())
		return
	}
	view := strings.TrimSpace(r.URL.Query().Get("view"))
	if view == "" {
		view = apiDiscoveryViewIndex
	}
	if view != apiDiscoveryViewIndex && view != apiDiscoveryViewOperations {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_API_VIEW", "view must be index or operations")
		return
	}

	routes := filterRoutes(apiRoutes(), selection)
	response := APIOverviewResponse{
		Service:      "Semantic Router Management API",
		Version:      "v1",
		Description:  "Versioned management, routing-preview, observability, storage, and diagnostic contracts",
		Capabilities: capabilityIndex(routes),
		Links: map[string]string{
			"capability":        apiRootPath + "?capability={capability}",
			"operations":        apiRootPath + "?view=operations",
			"openapi_spec":      "/openapi.json",
			"openapi_filtered":  "/openapi.json?capability={capability}&audience={audience}",
			"openapi_operation": "/openapi.json?path={path}&method={method}",
			"swagger_ui":        "/docs",
			"config_schema":     apiConfigSchemaPath,
			"models":            apiInventoryModelsPath,
			"health":            "/health",
			"ready":             "/ready",
		},
	}

	if view == apiDiscoveryViewOperations || !selection.empty() {
		response.Endpoints = endpointInfo(routes)
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func endpointInfo(routes []apiRoute) []EndpointInfo {
	endpoints := make([]EndpointInfo, 0, len(routes))
	for _, route := range routes {
		endpoints = append(endpoints, EndpointInfo{
			Path:             route.Path,
			Method:           route.Method,
			Description:      route.Description,
			Permission:       route.Permission,
			Sensitivity:      route.Sensitivity,
			EndpointContract: route.EndpointContract,
		})
	}
	return endpoints
}

func capabilityIndex(routes []apiRoute) []APICapabilityInfo {
	counts := make(map[string]int)
	primary := make(map[string]int)
	for _, route := range routes {
		counts[route.Capability]++
		if route.Visibility == APIVisibilityPrimary {
			primary[route.Capability]++
		}
	}

	result := make([]APICapabilityInfo, 0, len(capabilityRegistry))
	for _, capability := range capabilityRegistry {
		if counts[capability.Name] == 0 {
			continue
		}
		encoded := url.QueryEscape(capability.Name)
		result = append(result, APICapabilityInfo{
			Name:              capability.Name,
			Description:       capability.Description,
			OperationCount:    counts[capability.Name],
			PrimaryOperations: primary[capability.Name],
			Discovery:         apiRootPath + "?capability=" + encoded,
			OpenAPI:           "/openapi.json?capability=" + encoded,
		})
	}
	return result
}

func parseRouteSelection(query url.Values) (routeSelection, error) {
	selection := routeSelection{
		Capability: strings.TrimSpace(query.Get("capability")),
		Audience:   APIAudience(strings.TrimSpace(query.Get("audience"))),
		Plane:      APIPlane(strings.TrimSpace(query.Get("plane"))),
		Visibility: APIVisibility(strings.TrimSpace(query.Get("visibility"))),
	}
	if selection.Capability != "" && !knownCapability(selection.Capability) {
		return routeSelection{}, fmt.Errorf("unknown capability %q", selection.Capability)
	}
	if selection.Audience != "" && !oneOfAudience(selection.Audience) {
		return routeSelection{}, fmt.Errorf("unknown audience %q", selection.Audience)
	}
	if selection.Plane != "" && !oneOfPlane(selection.Plane) {
		return routeSelection{}, fmt.Errorf("unknown plane %q", selection.Plane)
	}
	if selection.Visibility != "" && !oneOfVisibility(selection.Visibility) {
		return routeSelection{}, fmt.Errorf("unknown visibility %q", selection.Visibility)
	}
	return selection, nil
}

func (selection routeSelection) empty() bool {
	return selection.Capability == "" && selection.Audience == "" && selection.Plane == "" && selection.Visibility == ""
}

func filterRoutes(routes []apiRoute, selection routeSelection) []apiRoute {
	filtered := make([]apiRoute, 0, len(routes))
	for _, route := range routes {
		if selection.Capability != "" && route.Capability != selection.Capability {
			continue
		}
		if selection.Audience != "" && !containsAudience(route.Audiences, selection.Audience) {
			continue
		}
		if selection.Plane != "" && route.Plane != selection.Plane {
			continue
		}
		if selection.Visibility != "" && route.Visibility != selection.Visibility {
			continue
		}
		filtered = append(filtered, route)
	}
	return filtered
}

func knownCapability(value string) bool {
	for _, capability := range capabilityRegistry {
		if capability.Name == value {
			return true
		}
	}
	return false
}

func containsAudience(values []APIAudience, want APIAudience) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func oneOfAudience(value APIAudience) bool {
	switch value {
	case APIAudienceAgent, APIAudienceOperator, APIAudienceClient, APIAudienceInternal:
		return true
	default:
		return false
	}
}

func oneOfPlane(value APIPlane) bool {
	switch value {
	case APIPlaneInfrastructure, APIPlaneManagement, APIPlaneDiagnostic, APIPlaneData:
		return true
	default:
		return false
	}
}

func oneOfVisibility(value APIVisibility) bool {
	return value == APIVisibilityPrimary || value == APIVisibilityAdvanced
}

// handleOpenAPISpec serves the complete OpenAPI 3.0 specification or a valid
// path/operation/capability subset for progressive agent discovery.
func (s *ClassificationAPIServer) handleOpenAPISpec(w http.ResponseWriter, r *http.Request) {
	selectedPath := strings.TrimSpace(r.URL.Query().Get("path"))
	selectedMethod := strings.ToUpper(strings.TrimSpace(r.URL.Query().Get("method")))
	selection, err := parseRouteSelection(r.URL.Query())
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_OPENAPI_FILTER", err.Error())
		return
	}

	if selectedMethod != "" && selectedPath == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_OPENAPI_FILTER", "method requires path")
		return
	}
	if selectedPath != "" && !selection.empty() {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_OPENAPI_FILTER", "path selection cannot be combined with capability, audience, plane, or visibility")
		return
	}

	spec := s.generateOpenAPISpecForRoutes(filterRoutes(apiRoutes(), selection))
	if selectedPath != "" {
		path, ok := spec.Paths[selectedPath]
		if !ok {
			s.writeErrorResponse(w, http.StatusNotFound, "OPENAPI_PATH_NOT_FOUND", "requested API path is not registered")
			return
		}
		if selectedMethod != "" {
			operation := selectOpenAPIOperation(path, selectedMethod)
			if operation == nil {
				s.writeErrorResponse(w, http.StatusNotFound, "OPENAPI_OPERATION_NOT_FOUND", "requested method is not registered for this API path")
				return
			}
			path = OpenAPIPath{}
			assignOpenAPIOperation(&path, selectedMethod, operation)
		}
		spec.Paths = map[string]OpenAPIPath{selectedPath: path}
	}
	s.writeJSONResponse(w, http.StatusOK, spec)
}

// handleSwaggerUI serves the generated OpenAPI contract at /docs.
func (s *ClassificationAPIServer) handleSwaggerUI(w http.ResponseWriter, _ *http.Request) {
	html := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Router API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css">
    <style>body { margin: 0; padding: 0; }</style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
                plugins: [SwaggerUIBundle.plugins.DownloadUrl],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(html))
}
