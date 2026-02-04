//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
)

// OpenAPI 3.0 spec structures

// OpenAPISpec represents an OpenAPI 3.0 specification
type OpenAPISpec struct {
	OpenAPI    string                 `json:"openapi"`
	Info       OpenAPIInfo            `json:"info"`
	Servers    []OpenAPIServer        `json:"servers"`
	Paths      map[string]OpenAPIPath `json:"paths"`
	Components OpenAPIComponents      `json:"components,omitempty"`
}

// OpenAPIInfo contains API metadata
type OpenAPIInfo struct {
	Title       string `json:"title"`
	Description string `json:"description"`
	Version     string `json:"version"`
}

// OpenAPIServer describes a server
type OpenAPIServer struct {
	URL         string `json:"url"`
	Description string `json:"description"`
}

// OpenAPIPath represents operations for a path
type OpenAPIPath struct {
	Get    *OpenAPIOperation `json:"get,omitempty"`
	Post   *OpenAPIOperation `json:"post,omitempty"`
	Put    *OpenAPIOperation `json:"put,omitempty"`
	Delete *OpenAPIOperation `json:"delete,omitempty"`
}

// OpenAPIOperation describes an API operation
type OpenAPIOperation struct {
	Summary     string                     `json:"summary"`
	Description string                     `json:"description,omitempty"`
	OperationID string                     `json:"operationId,omitempty"`
	Responses   map[string]OpenAPIResponse `json:"responses"`
	RequestBody *OpenAPIRequestBody        `json:"requestBody,omitempty"`
}

// OpenAPIResponse describes a response
type OpenAPIResponse struct {
	Description string                  `json:"description"`
	Content     map[string]OpenAPIMedia `json:"content,omitempty"`
}

// OpenAPIRequestBody describes a request body
type OpenAPIRequestBody struct {
	Description string                  `json:"description,omitempty"`
	Required    bool                    `json:"required,omitempty"`
	Content     map[string]OpenAPIMedia `json:"content"`
}

// OpenAPIMedia describes media type content
type OpenAPIMedia struct {
	Schema *OpenAPISchema `json:"schema,omitempty"`
}

// OpenAPISchema describes a schema
type OpenAPISchema struct {
	Type       string                   `json:"type,omitempty"`
	Properties map[string]OpenAPISchema `json:"properties,omitempty"`
	Items      *OpenAPISchema           `json:"items,omitempty"`
	Ref        string                   `json:"$ref,omitempty"`
}

// OpenAPIComponents contains reusable components
type OpenAPIComponents struct {
	Schemas map[string]OpenAPISchema `json:"schemas,omitempty"`
}

// APIOverviewResponse represents the response for GET /api/v1
type APIOverviewResponse struct {
	Service     string            `json:"service"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	Endpoints   []EndpointInfo    `json:"endpoints"`
	TaskTypes   []TaskTypeInfo    `json:"task_types"`
	Links       map[string]string `json:"links"`
}

// endpointRegistry is a centralized registry of all API endpoints with their metadata
var endpointRegistry = []EndpointMetadata{
	{Path: "/health", Method: "GET", Description: "Health check endpoint"},
	{Path: "/api/v1", Method: "GET", Description: "API discovery and documentation"},
	{Path: "/openapi.json", Method: "GET", Description: "OpenAPI 3.0 specification"},
	{Path: "/docs", Method: "GET", Description: "Interactive Swagger UI documentation"},
	{Path: "/api/v1/classify/intent", Method: "POST", Description: "Classify user queries into routing categories"},
	{Path: "/api/v1/classify/pii", Method: "POST", Description: "Detect personally identifiable information in text"},
	{Path: "/api/v1/classify/security", Method: "POST", Description: "Detect jailbreak attempts and security threats"},
	{Path: "/api/v1/classify/fact-check", Method: "POST", Description: "Classify if text needs fact-checking"},
	{Path: "/api/v1/classify/user-feedback", Method: "POST", Description: "Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different)"},
	{Path: "/api/v1/classify/combined", Method: "POST", Description: "Perform combined classification (intent, PII, and security)"},
	{Path: "/api/v1/classify/batch", Method: "POST", Description: "Batch classification with configurable task_type parameter"},
	{Path: "/info/models", Method: "GET", Description: "Get information about loaded models"},
	{Path: "/info/classifier", Method: "GET", Description: "Get classifier information and status"},
	{Path: "/v1/models", Method: "GET", Description: "OpenAI-compatible model listing"},
	{Path: "/metrics/classification", Method: "GET", Description: "Get classification metrics and statistics"},
	{Path: "/config/classification", Method: "GET", Description: "Get classification configuration"},
	{Path: "/config/classification", Method: "PUT", Description: "Update classification configuration"},
	{Path: "/config/system-prompts", Method: "GET", Description: "Get system prompt configuration (requires explicit enablement)"},
	{Path: "/config/system-prompts", Method: "PUT", Description: "Update system prompt configuration (requires explicit enablement)"},
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
	// Build endpoints list from registry, filtering out disabled endpoints
	endpoints := make([]EndpointInfo, 0, len(endpointRegistry))
	for _, metadata := range endpointRegistry {
		// Filter out system prompt endpoints if they are disabled
		if !s.enableSystemPromptAPI && (metadata.Path == "/config/system-prompts") {
			continue
		}
		endpoints = append(endpoints, EndpointInfo(metadata))
	}

	response := APIOverviewResponse{
		Service:     "Semantic Router Classification API",
		Version:     "v1",
		Description: "API for intent classification, PII detection, and security analysis",
		Endpoints:   endpoints,
		TaskTypes:   taskTypeRegistry,
		Links: map[string]string{
			"documentation": "https://vllm-project.github.io/semantic-router/",
			"openapi_spec":  "/openapi.json",
			"swagger_ui":    "/docs",
			"models_info":   "/info/models",
			"health":        "/health",
		},
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// generateOpenAPISpec generates an OpenAPI 3.0 specification from the endpoint registry
func (s *ClassificationAPIServer) generateOpenAPISpec() OpenAPISpec {
	spec := OpenAPISpec{
		OpenAPI: "3.0.0",
		Info: OpenAPIInfo{
			Title:       "Semantic Router Classification API",
			Description: "API for intent classification, PII detection, and security analysis",
			Version:     "v1",
		},
		Servers: []OpenAPIServer{
			{
				URL:         "/",
				Description: "Classification API Server",
			},
		},
		Paths: make(map[string]OpenAPIPath),
	}

	// Generate paths from endpoint registry
	for _, endpoint := range endpointRegistry {
		// Filter out system prompt endpoints if they are disabled
		if !s.enableSystemPromptAPI && endpoint.Path == "/config/system-prompts" {
			continue
		}

		path, ok := spec.Paths[endpoint.Path]
		if !ok {
			path = OpenAPIPath{}
		}

		operation := &OpenAPIOperation{
			Summary:     endpoint.Description,
			Description: endpoint.Description,
			OperationID: fmt.Sprintf("%s_%s", endpoint.Method, endpoint.Path),
			Responses: map[string]OpenAPIResponse{
				"200": {
					Description: "Successful response",
					Content: map[string]OpenAPIMedia{
						"application/json": {
							Schema: &OpenAPISchema{
								Type: "object",
							},
						},
					},
				},
				"400": {
					Description: "Bad request",
					Content: map[string]OpenAPIMedia{
						"application/json": {
							Schema: &OpenAPISchema{
								Type: "object",
								Properties: map[string]OpenAPISchema{
									"error": {
										Type: "object",
										Properties: map[string]OpenAPISchema{
											"code":      {Type: "string"},
											"message":   {Type: "string"},
											"timestamp": {Type: "string"},
										},
									},
								},
							},
						},
					},
				},
			},
		}

		// Add request body for POST and PUT methods
		if endpoint.Method == "POST" || endpoint.Method == "PUT" {
			operation.RequestBody = &OpenAPIRequestBody{
				Required: true,
				Content: map[string]OpenAPIMedia{
					"application/json": {
						Schema: &OpenAPISchema{
							Type: "object",
						},
					},
				},
			}
		}

		// Map operation to the appropriate method
		switch endpoint.Method {
		case "GET":
			path.Get = operation
		case "POST":
			path.Post = operation
		case "PUT":
			path.Put = operation
		case "DELETE":
			path.Delete = operation
		}

		spec.Paths[endpoint.Path] = path
	}

	return spec
}

// handleOpenAPISpec serves the OpenAPI 3.0 specification at /openapi.json
func (s *ClassificationAPIServer) handleOpenAPISpec(w http.ResponseWriter, _ *http.Request) {
	spec := s.generateOpenAPISpec()
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
                url: "/openapi.json",
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
