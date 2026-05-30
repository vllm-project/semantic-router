//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"strings"
)

// generateOpenAPISpec generates an OpenAPI 3.0 specification from the route catalog.
func (s *ClassificationAPIServer) generateOpenAPISpec() OpenAPISpec {
	spec := newOpenAPISpec()
	for _, route := range apiRoutes() {
		path := spec.Paths[route.Path]
		assignOpenAPIOperation(&path, route.Method, buildOpenAPIOperation(route))
		spec.Paths[route.Path] = path
	}

	return spec
}

func newOpenAPISpec() OpenAPISpec {
	return OpenAPISpec{
		OpenAPI: "3.0.0",
		Info: OpenAPIInfo{
			Title:       "Semantic Router Apiserver",
			Description: "HTTP router apiserver for classification utilities, config management, and service introspection",
			Version:     "v1",
		},
		Servers: []OpenAPIServer{
			{
				URL:         "/",
				Description: "Router Apiserver",
			},
		},
		Paths: make(map[string]OpenAPIPath),
	}
}

func buildOpenAPIOperation(route apiRoute) *OpenAPIOperation {
	operation := &OpenAPIOperation{
		Summary:     route.Description,
		Description: route.Description,
		OperationID: openAPIOperationID(route.Method, route.Path),
		Parameters:  openAPIPathParameters(route.Path),
		Responses: map[string]OpenAPIResponse{
			"200": openAPIObjectResponse("Successful response"),
			"400": openAPIErrorResponse("Bad request"),
		},
	}

	if route.RequestBody.Kind != requestBodyNone {
		operation.Responses["413"] = openAPIErrorResponse("Request body too large")
		operation.RequestBody = buildOpenAPIRequestBody(route.RequestBody)
	}

	return operation
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
		Content:     requestBodyMedia(body.Kind),
	}
}

func requestBodyDescription(body apiRequestBody) string {
	if body.Description != "" {
		return fmt.Sprintf("%s Limit: %d bytes.", body.Description, body.LimitBytes)
	}
	return fmt.Sprintf("%s request payload. Limit: %d bytes.", body.Kind, body.LimitBytes)
}

func openAPIObjectResponse(description string) OpenAPIResponse {
	return OpenAPIResponse{
		Description: description,
		Content:     openAPIObjectMedia(),
	}
}

func openAPIErrorResponse(description string) OpenAPIResponse {
	return OpenAPIResponse{
		Description: description,
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
	}
}

func openAPIObjectMedia() map[string]OpenAPIMedia {
	return map[string]OpenAPIMedia{
		"application/json": {
			Schema: &OpenAPISchema{
				Type: "object",
			},
		},
	}
}

func requestBodyMedia(kind requestBodyKind) map[string]OpenAPIMedia {
	switch kind {
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
		return openAPIObjectMedia()
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
