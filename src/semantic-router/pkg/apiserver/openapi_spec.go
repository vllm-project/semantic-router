//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"strings"
)

// generateOpenAPISpec generates an OpenAPI 3.0 specification from the route catalog.
func (s *ClassificationAPIServer) generateOpenAPISpec() OpenAPISpec {
	return s.generateOpenAPISpecForRoutes(apiRoutes())
}

func (s *ClassificationAPIServer) generateOpenAPISpecForRoutes(routes []apiRoute) OpenAPISpec {
	spec := newOpenAPISpec()
	seenTags := make(map[string]bool)
	for _, route := range routes {
		path := spec.Paths[route.Path]
		assignOpenAPIOperation(&path, route.Method, buildOpenAPIOperation(route))
		spec.Paths[route.Path] = path
		if !seenTags[route.Capability] {
			seenTags[route.Capability] = true
			spec.Tags = append(spec.Tags, OpenAPITag{Name: route.Capability, Description: capabilityDescription(route.Capability)})
		}
	}

	return spec
}

func newOpenAPISpec() OpenAPISpec {
	return OpenAPISpec{
		OpenAPI: "3.0.0",
		Info: OpenAPIInfo{
			Title:       "Semantic Router Apiserver",
			Description: "Router management API for configuration, plugin operations, prepared-model diagnostics, storage, and observability",
			Version:     "v1",
		},
		Servers: []OpenAPIServer{
			{
				URL:         "/",
				Description: "Router Apiserver",
			},
		},
		Paths: make(map[string]OpenAPIPath),
		Components: OpenAPIComponents{
			SecuritySchemes: map[string]OpenAPISecurityScheme{
				"bearerAuth": { // #nosec G101 -- OpenAPI security scheme metadata, not a credential.
					Type:         "http",
					Scheme:       "bearer",
					BearerFormat: "opaque management token",
					Description:  "Required when global.services.management_api.auth.mode is bearer.",
				},
			},
		},
	}
}

func buildOpenAPIOperation(route apiRoute) *OpenAPIOperation {
	operation := &OpenAPIOperation{
		Summary:          route.Description,
		Description:      route.Description,
		OperationID:      openAPIOperationID(route.Method, route.Path),
		Tags:             []string{route.Capability},
		Deprecated:       route.Deprecated,
		Parameters:       append(openAPIPathParameters(route.Path), route.Parameters...),
		Security:         openAPIOperationSecurity(route),
		Permission:       route.Permission,
		Sensitivity:      route.Sensitivity,
		AuditAction:      route.AuditAction,
		Plane:            route.Plane,
		Audiences:        append([]APIAudience(nil), route.Audiences...),
		Stability:        route.Stability,
		PluginOperations: append([]PluginOperationContract(nil), route.PluginOperations...),
		Visibility:       route.Visibility,
		Responses:        routeOpenAPIResponses(route),
	}

	if route.RequestBody.Kind != requestBodyNone {
		operation.Responses["413"] = openAPIErrorResponse("Request body too large")
		operation.RequestBody = buildOpenAPIRequestBody(route.RequestBody)
	}
	return operation
}

func capabilityDescription(name string) string {
	for _, capability := range capabilityRegistry {
		if capability.Name == name {
			return capability.Description
		}
	}
	return ""
}

func openAPIOperationSecurity(route apiRoute) []OpenAPISecurityRequirement {
	if route.Permission == PermHealthRead {
		return nil
	}
	return []OpenAPISecurityRequirement{
		{},
		{"bearerAuth": {}},
	}
}

func openAPIOperationID(method, path string) string {
	operationPath := strings.Trim(path, "/")
	if operationPath == "" {
		operationPath = "root"
	}

	replacer := strings.NewReplacer(
		"/", "_",
		"{", "",
		"}", "",
		"-", "_",
		".", "_",
	)
	return strings.ToLower(method) + "_" + replacer.Replace(operationPath)
}

func openAPIPathParameters(path string) []OpenAPIParameter {
	segments := strings.Split(path, "/")
	parameters := make([]OpenAPIParameter, 0)
	seen := make(map[string]struct{})

	for _, segment := range segments {
		if !strings.HasPrefix(segment, "{") || !strings.HasSuffix(segment, "}") {
			continue
		}

		name := strings.TrimSuffix(strings.TrimPrefix(segment, "{"), "}")
		if name == "" {
			continue
		}
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}

		parameters = append(parameters, OpenAPIParameter{
			Name:        name,
			In:          "path",
			Description: fmt.Sprintf("%s path parameter", name),
			Required:    true,
			Schema:      OpenAPISchema{Type: "string"},
		})
	}

	return parameters
}

func buildOpenAPIRequestBody(body apiRequestBody) *OpenAPIRequestBody {
	return &OpenAPIRequestBody{
		Description: requestBodyDescription(body),
		Required:    body.Required,
		Content:     requestBodyMedia(body),
	}
}

func requestBodyDescription(body apiRequestBody) string {
	if body.Description != "" {
		return fmt.Sprintf("%s Limit: %d bytes.", body.Description, body.LimitBytes)
	}
	return fmt.Sprintf("%s request payload. Limit: %d bytes.", body.Kind, body.LimitBytes)
}

func openAPIErrorResponse(description string) OpenAPIResponse {
	return OpenAPIResponse{Description: description, Headers: map[string]OpenAPIHeader{
		managementRequestIDHeader: {Description: "Correlation identifier for this management request.", Schema: OpenAPISchema{Type: "string"}},
	}, Content: map[string]OpenAPIMedia{
		"application/json": {Schema: openAPIRequestSchemaFor[managementErrorResponse]()},
	}}
}

func requestBodyMedia(body apiRequestBody) map[string]OpenAPIMedia {
	switch body.Kind {
	case requestBodyMultipart:
		return map[string]OpenAPIMedia{
			string(requestBodyMultipart): {
				Schema: &OpenAPISchema{
					Type: "object",
					Properties: map[string]OpenAPISchema{
						"file":    {Type: "string", Format: "binary"},
						"purpose": {Type: "string"},
					},
				},
			},
		}
	default:
		schema := body.Schema
		if schema == nil {
			schema = &OpenAPISchema{Type: "object"}
		}
		return map[string]OpenAPIMedia{
			string(requestBodyJSON): {Schema: schema},
		}
	}
}

func assignOpenAPIOperation(path *OpenAPIPath, method string, operation *OpenAPIOperation) {
	switch method {
	case "GET":
		path.Get = operation
	case "POST":
		path.Post = operation
	case "PATCH":
		path.Patch = operation
	case "PUT":
		path.Put = operation
	case "DELETE":
		path.Delete = operation
	}
}

func selectOpenAPIOperation(path OpenAPIPath, method string) *OpenAPIOperation {
	switch method {
	case "GET":
		return path.Get
	case "POST":
		return path.Post
	case "PATCH":
		return path.Patch
	case "PUT":
		return path.Put
	case "DELETE":
		return path.Delete
	default:
		return nil
	}
}
