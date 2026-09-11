//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strings"
)

// APIOverviewResponse represents the response for GET /api/v1
type APIOverviewResponse struct {
	Service     string            `json:"service"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	Endpoints   []EndpointInfo    `json:"endpoints"`
	TaskTypes   []TaskTypeInfo    `json:"task_types"`
	Links       map[string]string `json:"links"`
}

// taskTypeRegistry is a centralized registry of all supported task types
var taskTypeRegistry = []TaskTypeInfo{
	{Name: "intent", Description: "Intent/category classification (default for batch endpoint)"},
	{Name: "pii", Description: "Personally Identifiable Information detection"},
	{Name: "security", Description: "Jailbreak and security threat detection"},
	{Name: "all", Description: "All classification types combined"},
}

// handleAPIOverview handles GET /api/v1 for API discovery
func (s *ClassificationAPIServer) handleAPIOverview(w http.ResponseWriter, _ *http.Request) {
	// Build endpoints list from registry.
	routes := apiRoutes()
	endpoints := make([]EndpointInfo, 0, len(routes))
	for _, route := range routes {
		endpoints = append(endpoints, EndpointInfo{
			Path:        route.Path,
			Method:      route.Method,
			Description: route.Description,
			Permission:  route.Permission,
			Sensitivity: route.Sensitivity,
		})
	}

	response := APIOverviewResponse{
		Service:     "Semantic Router Apiserver",
		Version:     "v1",
		Description: "HTTP router apiserver for classification utilities, config management, and service introspection",
		Endpoints:   endpoints,
		TaskTypes:   taskTypeRegistry,
		Links: map[string]string{
			"documentation":     "https://vllm-project.github.io/semantic-router/",
			"openapi_spec":      "/openapi.json",
			"openapi_path":      "/openapi.json?path={path}",
			"openapi_operation": "/openapi.json?path={path}&method={method}",
			"swagger_ui":        "/docs",
			"config_schema":     "/config/router/schema",
			"models_info":       "/info/models",
			"health":            "/health",
			"ready":             "/ready",
		},
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleOpenAPISpec serves the complete OpenAPI 3.0 specification or a valid
// path/operation subset for progressive agent discovery.
func (s *ClassificationAPIServer) handleOpenAPISpec(w http.ResponseWriter, r *http.Request) {
	spec := s.generateOpenAPISpec()
	selectedPath := strings.TrimSpace(r.URL.Query().Get("path"))
	selectedMethod := strings.ToUpper(strings.TrimSpace(r.URL.Query().Get("method")))

	if selectedMethod != "" && selectedPath == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_OPENAPI_FILTER", "method requires path")
		return
	}
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

// handleSwaggerUI serves the Swagger UI at /docs
func (s *ClassificationAPIServer) handleSwaggerUI(w http.ResponseWriter, _ *http.Request) {
	// Serve a simple HTML page that loads Swagger UI from CDN
	html := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Router API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
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
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
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
