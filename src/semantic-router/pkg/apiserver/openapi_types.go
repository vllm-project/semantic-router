//go:build !windows && cgo

package apiserver

// OpenAPISpec represents an OpenAPI 3.0 specification.
type OpenAPISpec struct {
	OpenAPI    string                 `json:"openapi"`
	Info       OpenAPIInfo            `json:"info"`
	Servers    []OpenAPIServer        `json:"servers"`
	Tags       []OpenAPITag           `json:"tags,omitempty"`
	Paths      map[string]OpenAPIPath `json:"paths"`
	Components OpenAPIComponents      `json:"components,omitempty"`
}

// OpenAPITag describes one capability group.
type OpenAPITag struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// OpenAPIInfo contains API metadata.
type OpenAPIInfo struct {
	Title       string `json:"title"`
	Description string `json:"description"`
	Version     string `json:"version"`
}

// OpenAPIServer describes a server.
type OpenAPIServer struct {
	URL         string `json:"url"`
	Description string `json:"description"`
}

// OpenAPIPath represents operations for a path.
type OpenAPIPath struct {
	Get    *OpenAPIOperation `json:"get,omitempty"`
	Post   *OpenAPIOperation `json:"post,omitempty"`
	Patch  *OpenAPIOperation `json:"patch,omitempty"`
	Put    *OpenAPIOperation `json:"put,omitempty"`
	Delete *OpenAPIOperation `json:"delete,omitempty"`
}

// OpenAPIOperation describes an API operation.
type OpenAPIOperation struct {
	Summary     string                       `json:"summary"`
	Description string                       `json:"description,omitempty"`
	OperationID string                       `json:"operationId,omitempty"`
	Tags        []string                     `json:"tags,omitempty"`
	Deprecated  bool                         `json:"deprecated,omitempty"`
	Parameters  []OpenAPIParameter           `json:"parameters,omitempty"`
	Security    []OpenAPISecurityRequirement `json:"security,omitempty"`
	Responses   map[string]OpenAPIResponse   `json:"responses"`
	RequestBody *OpenAPIRequestBody          `json:"requestBody,omitempty"`
	Permission  RoutePermission              `json:"x-vllm-sr-permission"`
	Sensitivity RouteSensitivity             `json:"x-vllm-sr-sensitivity"`
	AuditAction RouteAuditAction             `json:"x-vllm-sr-audit-action,omitempty"`
	Plane       APIPlane                     `json:"x-vllm-sr-plane"`
	Audiences   []APIAudience                `json:"x-vllm-sr-audiences"`
	Stability   APIStability                 `json:"x-vllm-sr-stability"`
	Visibility  APIVisibility                `json:"x-vllm-sr-visibility"`
}

// OpenAPISecurityRequirement names one authentication scheme accepted by an
// operation. An empty requirement alongside bearerAuth means authentication is
// runtime-configurable: anonymous when disabled, bearer when enabled.
type OpenAPISecurityRequirement map[string][]string

// OpenAPIParameter describes an operation parameter.
type OpenAPIParameter struct {
	Name        string        `json:"name"`
	In          string        `json:"in"`
	Description string        `json:"description,omitempty"`
	Required    bool          `json:"required,omitempty"`
	Schema      OpenAPISchema `json:"schema"`
}

// OpenAPIResponse describes a response.
type OpenAPIResponse struct {
	Description string                  `json:"description"`
	Content     map[string]OpenAPIMedia `json:"content,omitempty"`
}

// OpenAPIRequestBody describes a request body.
type OpenAPIRequestBody struct {
	Description string                  `json:"description,omitempty"`
	Required    bool                    `json:"required,omitempty"`
	Content     map[string]OpenAPIMedia `json:"content"`
}

// OpenAPIMedia describes media type content.
type OpenAPIMedia struct {
	Schema *OpenAPISchema `json:"schema,omitempty"`
}

// OpenAPISchema describes a schema.
type OpenAPISchema struct {
	Type                 string                   `json:"type,omitempty"`
	Format               string                   `json:"format,omitempty"`
	Description          string                   `json:"description,omitempty"`
	Enum                 []string                 `json:"enum,omitempty"`
	Properties           map[string]OpenAPISchema `json:"properties,omitempty"`
	Required             []string                 `json:"required,omitempty"`
	Items                *OpenAPISchema           `json:"items,omitempty"`
	AdditionalProperties any                      `json:"additionalProperties,omitempty"`
	Ref                  string                   `json:"$ref,omitempty"`
}

// OpenAPIComponents contains reusable components.
type OpenAPIComponents struct {
	Schemas         map[string]OpenAPISchema         `json:"schemas,omitempty"`
	SecuritySchemes map[string]OpenAPISecurityScheme `json:"securitySchemes,omitempty"`
}

// OpenAPISecurityScheme describes an authentication mechanism understood by
// standard OpenAPI clients and Swagger UI.
type OpenAPISecurityScheme struct {
	Type         string `json:"type"`
	Scheme       string `json:"scheme"`
	BearerFormat string `json:"bearerFormat,omitempty"`
	Description  string `json:"description,omitempty"`
}
