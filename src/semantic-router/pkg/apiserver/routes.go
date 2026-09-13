//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
)

type apiRouteHandler func(*ClassificationAPIServer, http.ResponseWriter, *http.Request)

type requestBodyKind string

const (
	requestBodyNone      requestBodyKind = ""
	requestBodyJSON      requestBodyKind = "application/json"
	requestBodyMultipart requestBodyKind = "multipart/form-data"
)

type apiRequestBody struct {
	Kind        requestBodyKind
	Required    bool
	LimitBytes  int64
	Description string
	Schema      *OpenAPISchema
}

type apiRoute struct {
	EndpointMetadata
	Handler     apiRouteHandler
	RequestBody apiRequestBody
	Permission  RoutePermission
	Sensitivity RouteSensitivity
	AuditAction RouteAuditAction
}

func (r apiRoute) pattern() string {
	return fmt.Sprintf("%s %s", r.Method, r.Path)
}

func (r apiRoute) bind(s *ClassificationAPIServer) http.HandlerFunc {
	handler := func(w http.ResponseWriter, req *http.Request) {
		r.Handler(s, w, req)
	}
	return s.wrapRouteHandler(r, handler)
}

func jsonBodyFor[T any]() apiRequestBody {
	return jsonBodyWithLimitFor[T](defaultJSONRequestBodyLimit)
}

func strictJSONBodyFor[T any]() apiRequestBody {
	return strictJSONBodyWithLimitFor[T](defaultJSONRequestBodyLimit)
}

func jsonBodyWithLimitFor[T any](limit int64) apiRequestBody {
	return apiRequestBody{
		Kind:       requestBodyJSON,
		Required:   true,
		LimitBytes: limit,
		Schema:     openAPIRequestSchemaFor[T](),
	}
}

func strictJSONBodyWithLimitFor[T any](limit int64) apiRequestBody {
	body := jsonBodyWithLimitFor[T](limit)
	closeOpenAPIObjectSchemas(body.Schema)
	return body
}

// closeOpenAPIObjectSchemas mirrors json.Decoder.DisallowUnknownFields for
// reflected struct objects while preserving explicitly open map schemas.
func closeOpenAPIObjectSchemas(schema *OpenAPISchema) {
	if schema == nil {
		return
	}
	if schema.Type == "object" && schema.AdditionalProperties == nil && len(schema.Properties) > 0 {
		schema.AdditionalProperties = false
	}
	for name, property := range schema.Properties {
		closeOpenAPIObjectSchemas(&property)
		schema.Properties[name] = property
	}
	closeOpenAPIObjectSchemas(schema.Items)
	if additional, ok := schema.AdditionalProperties.(*OpenAPISchema); ok {
		closeOpenAPIObjectSchemas(additional)
	}
}

func multipartBody(limit int64, description string) apiRequestBody {
	return apiRequestBody{
		Kind:        requestBodyMultipart,
		Required:    true,
		LimitBytes:  limit,
		Description: description,
	}
}

func queryParameter(name, description, valueType string, enum ...string) OpenAPIParameter {
	return OpenAPIParameter{
		Name:        name,
		In:          "query",
		Description: description,
		Schema:      OpenAPISchema{Type: valueType, Enum: enum},
	}
}

func requiredQueryParameter(name, description, valueType string, enum ...string) OpenAPIParameter {
	parameter := queryParameter(name, description, valueType, enum...)
	parameter.Required = true
	return parameter
}

func headerParameter(name, description string, required bool) OpenAPIParameter {
	return OpenAPIParameter{
		Name:        name,
		In:          "header",
		Description: description,
		Required:    required,
		Schema:      OpenAPISchema{Type: "string"},
	}
}

func apiEndpointMetadata() []EndpointMetadata {
	routes := apiRoutes()
	metadata := make([]EndpointMetadata, 0, len(routes))
	for _, route := range routes {
		metadata = append(metadata, route.EndpointMetadata)
	}
	return metadata
}

func apiRoutes() []apiRoute {
	return appendAPIRoutes(
		make([]apiRoute, 0, 64),
		applyRouteContract(apiHealthRoutes(), routeContract("system", APIPlaneInfrastructure, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiRoutingRoutes(), routeContract("routing", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiClassifyRoutes(), routeContract("diagnostics", APIPlaneDiagnostic, APIVisibilityAdvanced, APIAudienceOperator, APIAudienceInternal)),
		applyRouteContract(apiInventoryRoutes(), routeContract("inventory", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiOpenAIDataRoutes(), routeContract("inventory", APIPlaneData, APIVisibilityPrimary, APIAudienceAgent, APIAudienceClient)),
		applyRouteContract(apiObservabilityRoutes(), routeContract("observability", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiRouterReplayRoutes(), routeContract("observability", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiResponseCacheRoutes(), routeContract("response-cache", APIPlaneManagement, APIVisibilityAdvanced, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiContextCompressionRoutes(), routeContract("context-compression", APIPlaneManagement, APIVisibilityAdvanced, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiConfigRoutes(), routeContract("config", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiKnowledgeBaseRoutes(), routeContract("storage", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiMemoryRoutes(), routeContract("storage", APIPlaneManagement, APIVisibilityAdvanced, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiVectorStoreRoutes(), routeContract("storage", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
		applyRouteContract(apiFileRoutes(), routeContract("storage", APIPlaneManagement, APIVisibilityPrimary, APIAudienceAgent, APIAudienceOperator)),
	)
}
